import pm4py
import openai
import neo4j
import pandas as pd
import os
import time
import traceback
from dotenv import load_dotenv
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import HybridCypherRetriever
from neo4j_graphrag.types import RetrieverResult, RetrieverResultItem
from collections import defaultdict, Counter

# Load environment variables
load_dotenv()

PROCESS_MINING_CONTEXT = """You are a process mining expert. 

IMPORTANT: Always check if the user is asking a simple factual question about the process data. If so, provide ONLY the direct answer based on the data.

For simple questions like:
- "How many times does X follow Y?"
- "What activities come after X?"
- "What is the flow from start to finish?"
- "Which activities are start/end activities?"

Provide only the factual answer from the process mining data without additional analysis.

Only provide detailed process mining analysis when the user explicitly asks for:
- Recommendations or improvements
- Bottleneck analysis
- Optimization suggestions
- Process efficiency insights
- Root cause analysis

Your expertise includes:
- Activity flow analysis and process discovery
- Bottleneck identification and root cause analysis
- Process efficiency optimization recommendations
- Manufacturing workflow understanding
- Business process reengineering principles

Guidelines for analysis (only when explicitly requested):
1. Focus on actionable process insights
2. Identify potential bottlenecks or inefficiencies
3. Suggest process improvements when relevant
4. Use process mining terminology appropriately
5. Consider both current state and optimization opportunities
6. Provide specific, data-driven recommendations

Always structure your responses to be helpful for process analysts and manufacturing engineers."""

EXAMPLE_QUESTIONS = [
    "What is the complete flow from start to finish?",
    "Which activities follow Material Preparation?",
    "Where are the potential bottlenecks in this process?",
    "How can we optimize the CNC Programming step?",
    "What improvements would reduce cycle time?",
    "How does quality inspection impact the process?",
    "What are the possible paths through the process?",
    "Which activities could be parallelized?",
    "What are the different process variants?",
    "Which cases follow similar patterns?"
]

def show_help():
    print("\nEXAMPLE QUESTIONS:")
    print("=" * 50)
    for i, question in enumerate(EXAMPLE_QUESTIONS, 1):
        print(f"   {i}. '{question}'")
    print("\n" + "=" * 50)

def prepare_pm4py_log(df):
    log_df = df.copy()
    log_df = log_df.rename(columns={
        'case_id': 'case:concept:name',
        'activity': 'concept:name',
        'timestamp': 'time:timestamp'
    })
    log = pm4py.convert_to_event_log(log_df)
    return log

def discover_process_model(log):
    print("Discovering process model...")
    dfg, start_activities, end_activities = pm4py.discover_dfg(log)
    print(f"Process discovered:")
    print(f"   - DFG edges: {len(dfg)}")
    print(f"   - Start activities: {list(start_activities.keys())}")
    print(f"   - End activities: {list(end_activities.keys())}")
    return dfg, start_activities, end_activities

def generate_activity_based_chunks(dfg, start_activities, end_activities):
    print("Generating activity-based process model chunks...")
    chunks = []
    all_activities = set()
    for (src, tgt), count in dfg.items():
        all_activities.add(src)
        all_activities.add(tgt)
    for activity in all_activities:
        incoming = [(src, count) for (src, tgt), count in dfg.items() if tgt == activity]
        outgoing = [(tgt, count) for (src, tgt), count in dfg.items() if src == activity]
        total_incoming = sum(count for _, count in incoming)
        total_outgoing = sum(count for _, count in outgoing)
        if activity in start_activities:
            execution_count = total_outgoing if total_outgoing > 0 else start_activities[activity]
        elif activity in end_activities:
            execution_count = total_incoming if total_incoming > 0 else end_activities[activity]
        else:
            execution_count = max(total_incoming, total_outgoing) if total_incoming > 0 or total_outgoing > 0 else 0
        text = f"{activity} is an activity in this process. "
        text += f"This activity executed {execution_count} times. "
        if activity in start_activities:
            text += f"{activity} is a starting activity that begins the process ({start_activities[activity]} cases start here). "
        if activity in end_activities:
            text += f"{activity} is an ending activity that concludes the process ({end_activities[activity]} cases end here). "
        if incoming:
            incoming_desc = [f"{src} ({count} times)" for src, count in incoming]
            text += f"{activity} is preceded by: {', '.join(incoming_desc)}. "
        if outgoing:
            outgoing_desc = [f"{tgt} ({count} times)" for tgt, count in outgoing]
            text += f"{activity} is followed by: {', '.join(outgoing_desc)}. "
        activity_model = {
            "activity": activity,
            "incoming": incoming,
            "outgoing": outgoing,
            "is_start": activity in start_activities,
            "is_end": activity in end_activities,
            "start_frequency": start_activities.get(activity, 0),
            "end_frequency": end_activities.get(activity, 0),
            "execution_count": execution_count
        }
        chunks.append({
            "text": text.strip(),
            "type": "activity_chunk",
            "activity_name": activity,
            "source": "activity_based_chunking",
            "data": activity_model
        })
    return chunks

def extract_case_variants(df, min_cases_per_variant=1):
    """Extract different case variants (process patterns) from the event log"""
    print(f"Extracting case variants (min cases per variant: {min_cases_per_variant})...")
    
    # Group by case and get activity sequences
    case_sequences = {}
    case_metadata = {}
    
    for case_id, group in df.groupby('case_id'):
        # Sort by timestamp to get correct order
        sorted_group = group.sort_values('timestamp')
        sequence = tuple(sorted_group['activity'].tolist())
        case_sequences[case_id] = sequence
        
        # Collect case metadata
        case_metadata[case_id] = {
            'num_activities': len(sequence),
            'unique_activities': len(set(sequence))
        }
    
    # Group cases by their activity sequences (variants)
    variant_groups = defaultdict(list)
    for case_id, sequence in case_sequences.items():
        variant_groups[sequence].append(case_id)
    
    # Filter variants by minimum case count
    frequent_variants = {variant: cases for variant, cases in variant_groups.items() 
                        if len(cases) >= min_cases_per_variant}
    
    # Sort variants by frequency (number of cases)
    sorted_variants = sorted(frequent_variants.items(), key=lambda x: len(x[1]), reverse=True)
    
    print(f"   Found {len(sorted_variants)} case variants")
    print("   Top 10 most frequent variants:")
    for i, (variant, cases) in enumerate(sorted_variants[:10], 1):
        variant_str = " → ".join(variant)
        print(f"   {i}. {variant_str} ({len(cases)} cases)")
    
    # Calculate variant statistics
    variant_stats = []
    for variant, cases in sorted_variants:
        stats = {
            'variant': variant,
            'cases': cases,
            'frequency': len(cases),
            'avg_activities': sum(case_metadata[c]['num_activities'] for c in cases) / len(cases),
            'avg_unique_activities': sum(case_metadata[c]['unique_activities'] for c in cases) / len(cases)
        }
        
        variant_stats.append(stats)
    
    return variant_stats

def generate_case_based_chunks(dfg, start_activities, end_activities, variant_stats):
    """Generate chunks based on case variants and their characteristics"""
    print("Generating case-based process model chunks...")
    chunks = []
    
    total_variants = len(variant_stats)
    total_cases = sum(stats['frequency'] for stats in variant_stats)
    # Find longest and shortest variant lengths
    lengths = [len(stats['variant']) for stats in variant_stats]
    max_length = max(lengths)
    min_length = min(lengths)
    
    for i, stats in enumerate(variant_stats):
        variant = stats['variant']
        cases = stats['cases']
        frequency = stats['frequency']
        variant_length = len(variant)
        variant_str = " → ".join(variant)
        text = f"Process variant '{variant_str}' represents a distinct execution pattern found in {frequency} cases ({frequency/total_cases*100:.1f}% of all cases). "
        text += f"This variant consists of {variant_length} activities. "
        
        # Add explicit longest/shortest info
        if variant_length == max_length:
            text += "This is the longest variant among all discovered variants. "
        if variant_length == min_length:
            text += "This is the shortest variant among all discovered variants. "
        
        if variant[0] in start_activities:
            text += f"This variant begins the process with {variant[0]}. "
        if variant[-1] in end_activities:
            text += f"This variant concludes the process with {variant[-1]}. "
        text += "The complete execution sequence is: "
        for j, activity in enumerate(variant):
            if j == 0:
                text += f"starting with {activity}"
            elif j == len(variant) - 1:
                text += f", and ending with {activity}"
            else:
                text += f", followed by {activity}"
        text += ". "
        text += f"On average, cases following this variant have {stats['avg_activities']:.1f} total activities and {stats['avg_unique_activities']:.1f} unique activities. "
        rank = i + 1
        percentage = (frequency / total_cases) * 100
        text += f"This is the {rank} most common variant out of {total_variants} variants identified, representing {percentage:.1f}% of all process executions. "
        example_cases = cases[:3]
        text += f"Example cases following this variant include: {', '.join(example_cases)}. "
        example_cases = cases[:3]
        text += f"Example cases following this variant include: {', '.join(example_cases)}. "
        if rank == 1:
            text += f"This is the most common process execution pattern (rank {rank} of {total_variants}). "
        elif rank == total_variants:
            text += f"This is the least common process execution pattern (rank {rank} of {total_variants}). "
        else:
            text += f"This is a mid-ranked variant (rank {rank} of {total_variants}). "
        unique_activities = set(variant)
        common_activities = set()
        for other_stats in variant_stats:
            if other_stats != stats:
                common_activities.update(set(other_stats['variant']))
        variant_specific = unique_activities - common_activities
        if variant_specific:
            text += f"This variant includes unique activities not found in other common variants: {', '.join(variant_specific)}. "
        variant_model = {
            "variant": variant,
            "variant_string": variant_str,
            "cases": cases,
            "frequency": frequency,
            "percentage": percentage,
            "length": variant_length,
            "rank": rank,
            "starts_process": variant[0] in start_activities,
            "ends_process": variant[-1] in end_activities,
            "avg_activities": stats['avg_activities'],
            "avg_unique_activities": stats['avg_unique_activities'],
            "unique_activities": list(variant_specific)
        }
        chunks.append({
            "text": text.strip(),
            "type": "case_variant_chunk",
            "variant_string": variant_str,
            "source": "case_based_chunking",
            "data": variant_model
        })
    return chunks

def connect_neo4j(uri, user, password):
    try:
        driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            session.run("RETURN 1")
        print("Connected to Neo4j successfully!")
        return driver
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        return None

def store_chunks_in_neo4j(driver, dfg, start_activities, end_activities, activity_chunks, case_chunks, variant_stats):
    print("Storing activity-based and case-based chunks and RAG data in Neo4j...")
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        print("   - Cleared existing data")
        try:
            session.run("DROP INDEX activity_chunk_vector_index IF EXISTS")
            session.run("DROP INDEX activity_chunk_fulltext_index IF EXISTS")
            session.run("DROP INDEX case_chunk_vector_index IF EXISTS")
            session.run("DROP INDEX case_chunk_fulltext_index IF EXISTS")
            print("   - Dropped existing indexes")
        except Exception as e:
            print(f"   - Note: {e}")
        dfg_data = [{"src": s, "tgt": t, "count": c} for (s, t), c in dfg.items()]
        session.run("""
            UNWIND $rows AS r
            MERGE (a:Activity {name: r.src})
            MERGE (b:Activity {name: r.tgt})
            MERGE (a)-[f:NEXT]->(b)
            ON CREATE SET f.count = r.count
            ON MATCH SET f.count = r.count
        """, rows=dfg_data)
        print(f"   - Created {len(dfg_data)} process paths")
        if start_activities:
            session.run("""
                UNWIND $starts AS start
                MATCH (a:Activity {name: start.activity})
                SET a.is_start = true, a.start_count = start.count
            """, starts=[{"activity": act, "count": count} for act, count in start_activities.items()])
            print(f"   - Marked {len(start_activities)} start activities")
        if end_activities:
            session.run("""
                UNWIND $ends AS end
                MATCH (a:Activity {name: end.activity})
                SET a.is_end = true, a.end_count = end.count
            """, ends=[{"activity": act, "count": count} for act, count in end_activities.items()])
            print(f"   - Marked {len(end_activities)} end activities")
        
        # Store activity chunks
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("   - Warning: No OpenAI API key, RAG will not work properly")
            return
        print("   - Creating activity chunk embeddings...")
        for i, chunk in enumerate(activity_chunks):
            try:
                response = openai.embeddings.create(
                    input=chunk["text"], 
                    model="text-embedding-3-large"
                )
                embedding = response.data[0].embedding
                session.run("""
                    CREATE (ac:ActivityChunk {
                        id: $id,
                        text: $text,
                        activity_name: $activity_name,
                        type: $type,
                        source: $source,
                        embedding: $embedding
                    })
                """, 
                id=i, 
                text=chunk["text"], 
                activity_name=chunk["activity_name"],
                type=chunk["type"],
                source=chunk["source"],
                embedding=embedding)
                session.run("""
                    MATCH (ac:ActivityChunk {id: $chunk_id})
                    MATCH (a:Activity {name: $activity_name})
                    MERGE (ac)-[:DESCRIBES]->(a)
                """, chunk_id=i, activity_name=chunk["activity_name"])
                print(f"   - Created activity chunk {i+1}/{len(activity_chunks)} for '{chunk['activity_name']}'")
            except Exception as e:
                print(f"   - Error creating activity chunk {i}: {e}")
        
        # Create CaseVariant nodes for the different variants
        for i, stats in enumerate(variant_stats):
            variant = stats['variant']
            variant_str = " → ".join(variant)
            
            # Create variant node with basic properties
            variant_props = {
                'id': i,
                'variant_string': variant_str,
                'frequency': stats['frequency'],
                'length': len(variant),
                'rank': i + 1,
                'avg_activities': stats['avg_activities'],
                'avg_unique_activities': stats['avg_unique_activities']
            }
            
            session.run("""
                CREATE (cv:CaseVariant $props)
            """, props=variant_props)
            
            # Link CaseVariant to Activities in sequence
            for j, activity in enumerate(variant):
                session.run("""
                    MATCH (cv:CaseVariant {id: $variant_id})
                    MATCH (a:Activity {name: $activity})
                    MERGE (cv)-[:EXECUTES {position: $position}]->(a)
                """, variant_id=i, activity=activity, position=j)
            
            # Create Case nodes and link to variant
            for case_id in stats['cases'][:10]:  # Limit to first 10 cases to avoid too many nodes
                session.run("""
                    MATCH (cv:CaseVariant {id: $variant_id})
                    CREATE (c:Case {id: $case_id})
                    MERGE (c)-[:FOLLOWS]->(cv)
                """, variant_id=i, case_id=case_id)
        
        print(f"   - Created {len(variant_stats)} CaseVariant nodes")
        
        # Store case chunks
        print("   - Creating case variant chunk embeddings...")
        for i, chunk in enumerate(case_chunks):
            try:
                response = openai.embeddings.create(
                    input=chunk["text"], 
                    model="text-embedding-3-large"
                )
                embedding = response.data[0].embedding
                session.run("""
                    CREATE (cc:CaseChunk {
                        id: $id,
                        text: $text,
                        variant_string: $variant_string,
                        type: $type,
                        source: $source,
                        embedding: $embedding
                    })
                """, 
                id=i, 
                text=chunk["text"], 
                variant_string=chunk["variant_string"],
                type=chunk["type"],
                source=chunk["source"],
                embedding=embedding)
                session.run("""
                    MATCH (cc:CaseChunk {id: $chunk_id})
                    MATCH (cv:CaseVariant {id: $variant_id})
                    MERGE (cc)-[:DESCRIBES]->(cv)
                """, chunk_id=i, variant_id=i)
                print(f"   - Created case chunk {i+1}/{len(case_chunks)} for '{chunk['variant_string']}'")
            except Exception as e:
                print(f"   - Error creating case chunk {i}: {e}")
        
        time.sleep(2)
        try:
            session.run("""
                CREATE FULLTEXT INDEX activity_chunk_fulltext_index IF NOT EXISTS
                FOR (ac:ActivityChunk) ON EACH [ac.text, ac.activity_name]
            """)
            session.run("""
                CREATE FULLTEXT INDEX case_chunk_fulltext_index IF NOT EXISTS
                FOR (cc:CaseChunk) ON EACH [cc.text, cc.variant_string]
            """)
            print("   - Created fulltext indexes")
        except Exception as e:
            print(f"   - Warning: Fulltext index creation: {e}")
        try:
            result = session.run("MATCH (ac:ActivityChunk) RETURN count(ac) as count")
            node_count = result.single()["count"]
            if node_count > 0:
                session.run("""
                    CREATE VECTOR INDEX activity_chunk_vector_index IF NOT EXISTS
                    FOR (ac:ActivityChunk) ON (ac.embedding)
                    OPTIONS {
                        indexConfig: {
                            `vector.dimensions`: 3072,
                            `vector.similarity_function`: 'cosine'
                        }
                    }
                """)
            result = session.run("MATCH (cc:CaseChunk) RETURN count(cc) as count")
            node_count = result.single()["count"]
            if node_count > 0:
                session.run("""
                    CREATE VECTOR INDEX case_chunk_vector_index IF NOT EXISTS
                    FOR (cc:CaseChunk) ON (cc.embedding)
                    OPTIONS {
                        indexConfig: {
                            `vector.dimensions`: 3072,
                            `vector.similarity_function`: 'cosine'
                        }
                    }
                """)
            print("   - Created vector indexes")
        except Exception as e:
            print(f"   - Critical error in vector index creation: {e}")
            traceback.print_exc()

def setup_retriever(driver, chunk_type):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OpenAI API key not found. RAG requires embeddings.")
        return None
    try:
        with driver.session() as session:
            result = session.run("SHOW INDEXES YIELD name, state, type, labelsOrTypes, properties")
            all_indexes = list(result)
            vector_index_name = None
            vector_index_ready = False
            fulltext_index_name = None
            fulltext_index_ready = False
            for idx in all_indexes:
                if (idx['type'] == 'VECTOR' and 
                    chunk_type in str(idx['labelsOrTypes']) and 
                    'embedding' in str(idx['properties'])):
                    vector_index_name = idx['name']
                    vector_index_ready = idx['state'] == 'ONLINE'
                if (idx['type'] == 'FULLTEXT' and 
                    chunk_type in str(idx['labelsOrTypes'])):
                    fulltext_index_name = idx['name']
                    fulltext_index_ready = idx['state'] == 'ONLINE'
        embedder = OpenAIEmbeddings(model="text-embedding-3-large")
        def custom_result_formatter(record):
            from neo4j_graphrag.types import RetrieverResultItem
            score = record.get("score", None)
            record_metadata = record.get("metadata", {})
            metadata = {}
            if score is not None:
                metadata["score"] = score
            if record_metadata:
                metadata.update(record_metadata)
            content = record.get("text", str(record))
            return RetrieverResultItem(
                content=content,
                metadata=metadata if metadata else None,
            )
        if not (vector_index_ready and fulltext_index_ready and vector_index_name and fulltext_index_name):
            print(f"Error: Both vector and fulltext indexes are required for {chunk_type} HybridCypherRetriever")
            return None
        if chunk_type == "ActivityChunk":
            retrieval_query = """
                MATCH (node:ActivityChunk)-[:DESCRIBES]->(activity:Activity)
                RETURN node.text AS text,
                       score AS score,
                       {
                           activity_name: activity.name,
                           is_start: activity.is_start,
                           is_end: activity.is_end,
                           start_count: activity.start_count,
                           end_count: activity.end_count,
                           type: node.type,
                           source: node.source
                       } AS metadata
            """
        elif chunk_type == "CaseChunk":
            retrieval_query = """
        MATCH (node)-[:DESCRIBES]->(variant:CaseVariant)
        RETURN node.text AS text,
               score AS score,
               {
                   variant_string: variant.variant_string,
                   rank: variant.rank,
                   frequency: variant.frequency,
                   type: node.type,
                   source: node.source
               } AS metadata
    """
        else:
            raise ValueError("Unknown chunk_type")
        retriever = HybridCypherRetriever(
            driver=driver,
            vector_index_name=vector_index_name,
            fulltext_index_name=fulltext_index_name,
            retrieval_query=retrieval_query,
            embedder=embedder,
            result_formatter=custom_result_formatter
        )
        return retriever
    except Exception as e:
        print(f"Error setting up retriever for {chunk_type}: {e}")
        traceback.print_exc()
        return None

def graphrag_query_interface(rag_activity, rag_case):
    print("\n" + "="*80)
    print("PROCESS MINING EXPERT - GraphRAG Interface (Integrated)")
    print("="*80)
    print("I'm your process mining expert assistant. You can query:")
    print("1. Activity-based context (individual activities and their relationships)")
    print("2. Case-based context (process variants and execution patterns)")
    print("3. Both (merged results)")
    print("\nType 'quit' to exit, 'help' for more examples")
    print("-" * 80)
    while True:
        mode = input("\nChoose context: (1) Activity-based, (2) Case-based, (3) Both: ").strip()
        if mode.lower() in ['quit', 'exit', 'q']:
            print("\nThanks for using Process Mining Expert! Keep optimizing those processes!")
            break
        if mode.lower() in ['help', 'examples', '?']:
            show_help()
            continue
        if mode not in ['1', '2', '3']:
            print("Please enter 1, 2, or 3")
            continue
        question = input("\nProcess Mining Question: ").strip()
        if not question:
            continue
        try:
            if mode == "1":
                rag = rag_activity
                context_label = "ACTIVITY-BASED"
            elif mode == "2":
                rag = rag_case
                context_label = "CASE-BASED"
            else:
                # Both: retrieve from both and merge
                print("\nRETRIEVED CHUNKS (Activity-Based):")
                print("-" * 50)
                search_result_a = rag_activity.retriever.search(question, top_k=3)
                for i, item in enumerate(search_result_a.items, 1):
                    content = str(item.content)
                    if content.startswith('<Record text="') and content.endswith('">'):
                        actual_content = content[14:-2]
                    else:
                        actual_content = content
                    print(f"Activity Chunk {i}: {actual_content}")
                    print(f"   Metadata: {item.metadata}")
                print("\nRETRIEVED CHUNKS (Case-Based):")
                print("-" * 50)
                search_result_c = rag_case.retriever.search(question, top_k=3)
                for i, item in enumerate(search_result_c.items, 1):
                    content = str(item.content)
                    if content.startswith('<Record text="') and content.endswith('">'):
                        actual_content = content[14:-2]
                    else:
                        actual_content = content
                    print(f"Case Chunk {i}: {actual_content}")
                    print(f"   Metadata: {item.metadata}")
                # Enhanced prompt with both
                answer_a = rag_activity.search(question).answer
                answer_c = rag_case.search(question).answer
                enhanced_prompt = f"""{PROCESS_MINING_CONTEXT}

Manufacturing Process Context: You are analyzing a manufacturing process with activities like Material Preparation, CNC Programming, Turning Process, Quality Inspection, etc. You have access to both ACTIVITY-BASED and CASE-BASED chunks.

Activity-Based Retrieved Information: {answer_a}

Case-Based Retrieved Information: {answer_c}

User Question: {question}

Please provide a detailed process mining analysis based on the retrieved information from both perspectives:"""
                llm = OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0.1})
                enhanced_answer = llm.invoke(enhanced_prompt)
                print(f"\n💡 ENHANCED ANSWER (COMBINED):")
                print("-" * 50)
                print(f"{enhanced_answer.content}")
                continue
            print(f"\nRETRIEVED CHUNKS ({context_label}):")
            print("-" * 50)
            search_result = rag.retriever.search(question, top_k=5)
            for i, item in enumerate(search_result.items, 1):
                content = str(item.content)
                if content.startswith('<Record text="') and content.endswith('">'):
                    actual_content = content[14:-2]
                else:
                    actual_content = content
                print(f"Chunk {i}: {actual_content}")
                print(f"   Metadata: {item.metadata}")
            result = rag.search(question)
            enhanced_prompt = f"""{PROCESS_MINING_CONTEXT}

Manufacturing Process Context: You are analyzing a manufacturing process with activities like Material Preparation, CNC Programming, Turning Process, Quality Inspection, etc. You have access to {context_label} chunks.

Retrieved Information: {result.answer}

User Question: {question}

Please provide a detailed process mining analysis based on the retrieved information:"""
            llm = OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0.1})
            enhanced_answer = llm.invoke(enhanced_prompt)
            print(f"\n💡 ENHANCED ANSWER:")
            print("-" * 50)
            print(f"{enhanced_answer.content}")
        except Exception as e:
            print(f"\nAnalysis Error: {e}")
            print("Try rephrasing your question or type 'help' for examples")

def setup_environment():
    neo4j_uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    csv_file_path = os.getenv("CSV_FILE_PATH")
    if openai_api_key:
        openai.api_key = openai_api_key
    else:
        print("Warning: OPENAI_API_KEY not found in environment")
    if not csv_file_path:
        print("Error: CSV_FILE_PATH environment variable not set!")
        print("Please add CSV_FILE_PATH to your .env file")
        return neo4j_uri, neo4j_user, neo4j_password, openai_api_key, None
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found at {csv_file_path}")
        print("Please check the CSV_FILE_PATH in your .env file")
        return neo4j_uri, neo4j_user, neo4j_password, openai_api_key, None
    return neo4j_uri, neo4j_user, neo4j_password, openai_api_key, csv_file_path

def main():
    print("Starting Neo4j + PM4py + GraphRAG Test with Integrated Activity & Case Chunks")
    print("=" * 70)
    neo4j_uri, neo4j_user, neo4j_password, openai_api_key, csv_file_path = setup_environment()
    if not openai_api_key:
        print("Error: OpenAI API key is required for RAG functionality!")
        return
    if not csv_file_path:
        print("Error: CSV file path is required!")
        return
    print(f"Using CSV file: {csv_file_path}")
    print(f"Loading data from {csv_file_path}...")
    try:
        df = pd.read_csv(csv_file_path, sep=';')
        print(f" Loaded dataset with {len(df)} events")
        if 'case_id' in df.columns:
            df['case_id'] = df['case_id'].astype(str)
        elif 'case:concept:name' in df.columns:
            df['case:concept:name'] = df['case:concept:name'].astype(str)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'time:timestamp' in df.columns:
            df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
        print(f"   Dataset contains {df['case_id'].nunique()} unique cases")
        print(f"   Activities: {df['activity'].unique()}")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return
    log = prepare_pm4py_log(df)
    dfg, start_activities, end_activities = discover_process_model(log)
    print("Generating activity-based chunks for RAG...")
    activity_chunks = generate_activity_based_chunks(dfg, start_activities, end_activities)
    print(f"   Generated {len(activity_chunks)} activity-based chunks")
    print("Extracting case variants for RAG...")
    variant_stats = extract_case_variants(df, min_cases_per_variant=1)
    print("Generating case-based chunks for RAG...")
    case_chunks = generate_case_based_chunks(dfg, start_activities, end_activities, variant_stats)
    print(f"   Generated {len(case_chunks)} case-based chunks")
    driver = connect_neo4j(neo4j_uri, neo4j_user, neo4j_password)
    if driver:
        store_chunks_in_neo4j(driver, dfg, start_activities, end_activities, activity_chunks, case_chunks, variant_stats)
        retriever_activity = setup_retriever(driver, "ActivityChunk")
        retriever_case = setup_retriever(driver, "CaseChunk")
        if retriever_activity and retriever_case:
            rag_activity = GraphRAG(retriever=retriever_activity, llm=OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0.1}))
            rag_case = GraphRAG(retriever=retriever_case, llm=OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0.1}))
            graphrag_query_interface(rag_activity, rag_case)
        else:
            print("Failed to setup GraphRAG retrievers. Check your OpenAI API key and Neo4j indexes.")
        driver.close()
        print("\nDisconnected from Neo4j. Goodbye!")
    else:
        print("Cannot proceed without Neo4j connection.")

if __name__ == "__main__":
    main()