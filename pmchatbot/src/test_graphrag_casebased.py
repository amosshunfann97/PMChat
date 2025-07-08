import pm4py
import openai
import neo4j
import pandas as pd
import os
from dotenv import load_dotenv
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import HybridCypherRetriever
import time
import traceback
from collections import defaultdict, Counter

# Load environment variables
load_dotenv()

# Global process mining domain context - define once and reuse
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

# Global example questions for help - used by both interfaces
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
    """Display example questions - shared by both interfaces"""
    print("\nEXAMPLE QUESTIONS:")
    print("=" * 50)
    for i, question in enumerate(EXAMPLE_QUESTIONS, 1):
        print(f"   {i}. '{question}'")
    print("\n" + "=" * 50)

def prepare_pm4py_log(df):
    """Convert DataFrame to PM4py event log"""
    log_df = df.copy()
    log_df = log_df.rename(columns={
        'case_id': 'case:concept:name',
        'activity': 'concept:name',
        'timestamp': 'time:timestamp'
    })
    
    log = pm4py.convert_to_event_log(log_df)
    return log

def discover_process_model(log):
    """Discover process model using PM4py"""
    print("Discovering process model...")
    
    dfg, start_activities, end_activities = pm4py.discover_dfg(log)
    
    print(f"Process discovered:")
    print(f"   - DFG edges: {len(dfg)}")
    print(f"   - Start activities: {list(start_activities.keys())}")
    print(f"   - End activities: {list(end_activities.keys())}")
    
    return dfg, start_activities, end_activities

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
    """Generate chunks based on case variants and their characteristics (Strategy 3)"""
    print("Generating case-based process model chunks...")
    chunks = []
    
    total_variants = len(variant_stats)
    total_cases = sum(stats['frequency'] for stats in variant_stats)
    
    for i, stats in enumerate(variant_stats):
        variant = stats['variant']
        cases = stats['cases']
        frequency = stats['frequency']
        
        # Create natural language description for this variant
        variant_str = " → ".join(variant)
        text = f"Process variant '{variant_str}' represents a distinct execution pattern found in {frequency} cases ({frequency/total_cases*100:.1f}% of all cases). "
        
        # Add variant characteristics
        variant_length = len(variant)
        text += f"This variant consists of {variant_length} activities. "
        
        # Check if variant starts with a start activity
        if variant[0] in start_activities:
            text += f"This variant begins the process with {variant[0]}. "
        
        # Check if variant ends with an end activity  
        if variant[-1] in end_activities:
            text += f"This variant concludes the process with {variant[-1]}. "
        
        # Add detailed activity sequence description
        text += "The complete execution sequence is: "
        for j, activity in enumerate(variant):
            if j == 0:
                text += f"starting with {activity}"
            elif j == len(variant) - 1:
                text += f", and ending with {activity}"
            else:
                text += f", followed by {activity}"
        text += ". "
        
        # Add statistical information
        text += f"On average, cases following this variant have {stats['avg_activities']:.1f} total activities and {stats['avg_unique_activities']:.1f} unique activities. "
        
        # Add frequency and ranking context
        rank = i + 1
        percentage = (frequency / total_cases) * 100
        text += f"This is the {rank} most common variant out of {total_variants} variants identified, representing {percentage:.1f}% of all process executions. "
        
        # Add case examples
        example_cases = cases[:3]  # Show first 3 cases as examples
        text += f"Example cases following this variant include: {', '.join(example_cases)}. "
        
        # Compare with other variants
        if rank == 1:
            text += "This is the most common process execution pattern. "
        elif rank <= 3:
            text += "This represents one of the most frequent process execution patterns. "
        else:
            text += "This represents a less common but still significant process execution pattern. "
        
        # Identify unique characteristics
        unique_activities = set(variant)
        common_activities = set()
        for other_stats in variant_stats:
            if other_stats != stats:
                common_activities.update(set(other_stats['variant']))
        
        variant_specific = unique_activities - common_activities
        if variant_specific:
            text += f"This variant includes unique activities not found in other common variants: {', '.join(variant_specific)}. "
        
        # Create variant model data
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
    """Connect to Neo4j database"""
    try:
        driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            session.run("RETURN 1")
        print("Connected to Neo4j successfully!")
        return driver
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        return None

def store_case_chunks_in_neo4j(driver, dfg, start_activities, end_activities, chunks, variant_stats):
    """Store case-based chunks with embeddings for RAG"""
    print("Storing case-based chunks and RAG data in Neo4j...")
    
    with driver.session() as session:
        # Clear existing data and indexes
        session.run("MATCH (n) DETACH DELETE n")
        print("   - Cleared existing data")
        
        # Drop existing indexes to avoid conflicts
        try:
            session.run("DROP INDEX case_chunk_vector_index IF EXISTS")
            session.run("DROP INDEX case_chunk_fulltext_index IF EXISTS")
            print("   - Dropped existing indexes")
        except Exception as e:
            print(f"   - Note: {e}")
        
        # Create activity nodes and NEXT relationships (keep graph structure)
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
        
        # Mark start and end activities
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
        
        # Create CaseVariant nodes for the different variants
        for i, stats in enumerate(variant_stats):
            variant = stats['variant']
            variant_str = " → ".join(variant)
            
            # Create variant node with basic properties (no duration)
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
        
        # Create case-based chunks with embeddings
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("   - Warning: No OpenAI API key, RAG will not work properly")
            return
        
        print("   - Creating case variant chunk embeddings...")
        for i, chunk in enumerate(chunks):
            try:
                # Generate embedding for the case variant chunk
                response = openai.embeddings.create(
                    input=chunk["text"], 
                    model="text-embedding-3-large"
                )
                embedding = response.data[0].embedding
                
                # Create case variant chunk node with embedding
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
                
                # Link chunk to the corresponding CaseVariant node
                session.run("""
                    MATCH (cc:CaseChunk {id: $chunk_id})
                    MATCH (cv:CaseVariant {id: $variant_id})
                    MERGE (cc)-[:DESCRIBES]->(cv)
                """, chunk_id=i, variant_id=i)
                
                print(f"   - Created case chunk {i+1}/{len(chunks)} for '{chunk['variant_string']}'")
                
            except Exception as e:
                print(f"   - Error creating case chunk {i}: {e}")
        
        # Wait for nodes to be committed
        time.sleep(2)
        
        # Create fulltext index first
        try:
            session.run("""
                CREATE FULLTEXT INDEX case_chunk_fulltext_index IF NOT EXISTS
                FOR (cc:CaseChunk) ON EACH [cc.text, cc.variant_string]
            """)
            print("   - Created fulltext index")
        except Exception as e:
            print(f"   - Warning: Fulltext index creation: {e}")
        
        # Create vector index with improved error handling
        vector_index_created = False
        try:
            # First, check if any CaseChunk nodes exist
            result = session.run("MATCH (cc:CaseChunk) RETURN count(cc) as count")
            node_count = result.single()["count"]
            print(f"   - Found {node_count} CaseChunk nodes for indexing")
            
            if node_count == 0:
                print("   - Warning: No CaseChunk nodes found, vector index may fail")
                return
            
            # Try to create vector index
            try:
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
                print("   - Created vector index")
                vector_index_created = True
            except Exception as e:
                print(f"   Vector index creation failed: {e}")
                print("   Your Neo4j version might need a different syntax")
                vector_index_created = False
            
            if vector_index_created:
                print("   - Vector index creation completed (will be verified during GraphRAG setup)")
            else:
                print("   - Error: Could not create vector index with any method")
                
        except Exception as e:
            print(f"   - Critical error in vector index creation: {e}")
            traceback.print_exc()

def setup_case_chunk_graphrag(driver):
    """Setup GraphRAG with case-based chunks"""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OpenAI API key not found. RAG requires embeddings.")
        return None
    
    try:
        # Check if indexes exist and are online
        with driver.session() as session:
            try:
                # List ALL indexes for debugging
                result = session.run("SHOW INDEXES YIELD name, state, type, labelsOrTypes, properties")
                all_indexes = list(result)
                print("All indexes in database:")
                for idx in all_indexes:
                    print(f"   - {idx['name']}: {idx['state']} ({idx['type']}) - {idx['labelsOrTypes']} {idx['properties']}")
                
                # Look for vector index on CaseChunk.embedding
                vector_index_name = None
                vector_index_ready = False
                
                for idx in all_indexes:
                    if (idx['type'] == 'VECTOR' and 
                        'CaseChunk' in str(idx['labelsOrTypes']) and 
                        'embedding' in str(idx['properties'])):
                        vector_index_name = idx['name']
                        vector_index_ready = idx['state'] == 'ONLINE'
                        print(f"Found vector index: {vector_index_name} - {idx['state']}")
                        break
                
                # Look for fulltext index
                fulltext_index_name = None
                fulltext_index_ready = False
                
                for idx in all_indexes:
                    if (idx['type'] == 'FULLTEXT' and 
                        'CaseChunk' in str(idx['labelsOrTypes'])):
                        fulltext_index_name = idx['name']
                        fulltext_index_ready = idx['state'] == 'ONLINE'
                        print(f"Found fulltext index: {fulltext_index_name} - {idx['state']}")
                        break
                
                print(f"Vector index ready: {vector_index_ready} (name: {vector_index_name})")
                print(f"Fulltext index ready: {fulltext_index_ready} (name: {fulltext_index_name})")
                
            except Exception as e:
                print(f"Error checking indexes: {e}")
                return None
        
        # Setup embedder for query encoding
        embedder = OpenAIEmbeddings(model="text-embedding-3-large")
        
        # Use HybridCypherRetriever (requires both indexes)
        if vector_index_ready and fulltext_index_ready and vector_index_name and fulltext_index_name:
            print(f"Using HybridCypherRetriever with vector index '{vector_index_name}' and fulltext index '{fulltext_index_name}'...")
            
            # Enhanced retrieval query for case-based chunks (no duration references)
            retrieval_query = """
                MATCH (node)-[:DESCRIBES]->(variant:CaseVariant)
                
                // Get activities in this variant
                OPTIONAL MATCH (variant)-[e:EXECUTES]->(activity:Activity)
                
                // Get some example cases
                OPTIONAL MATCH (case:Case)-[:FOLLOWS]->(variant)
                
                WITH node, variant,
                     collect(DISTINCT activity.name + '(' + toString(e.position) + ')') as variant_activities,
                     collect(DISTINCT case.id)[..3] as example_cases
                
                RETURN node.text + 
                       ' [Variant Context: ' + variant.variant_string + 
                       ' (rank: ' + toString(variant.rank) + ', frequency: ' + toString(variant.frequency) + ')' +
                       CASE WHEN size(example_cases) > 0 
                            THEN ' Example cases: ' + reduce(s = '', x IN example_cases | s + x + ', ')
                            ELSE '' END + ']'
                       AS text
            """
            
            retriever = HybridCypherRetriever(
                driver=driver,
                vector_index_name=vector_index_name,
                fulltext_index_name=fulltext_index_name,
                retrieval_query=retrieval_query,
                embedder=embedder
            )
        else:
            print("Error: Both vector and fulltext indexes are required for HybridCypherRetriever")
            print("Please ensure your indexes are created and ONLINE")
            return None
        
        # Setup LLM
        llm = OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0.1})
        
        # Setup GraphRAG
        rag = GraphRAG(
            retriever=retriever,
            llm=llm
        )
        
        print("Case-based GraphRAG setup completed successfully!")
        return rag
        
    except Exception as e:
        print(f"Error setting up GraphRAG: {e}")
        traceback.print_exc()
        return None

def query_neo4j(driver):
    """Run basic queries to verify the process model and case chunks"""
    print("\nQuerying Neo4j data...")
    
    with driver.session() as session:
        # Get all activities
        result = session.run("""
            MATCH (a:Activity)
            OPTIONAL MATCH (a)-[r:NEXT]->()
            RETURN a.name as activity, 
                   a.is_start as is_start, 
                   a.is_end as is_end,
                   count(r) as outgoing_connections
            ORDER BY a.name
        """)
        
        print("\nActivities in the process:")
        for record in result:
            start_indicator = "[START]" if record["is_start"] else "       "
            end_indicator = "[END]" if record["is_end"] else "     "
            print(f"   {start_indicator} {end_indicator} {record['activity']} (connections: {record['outgoing_connections']})")
        
        # Get case variants (removed duration references)
        result = session.run("""
            MATCH (cv:CaseVariant)
            RETURN cv.variant_string as variant_string, 
                   cv.frequency as frequency,
                   cv.rank as rank
            ORDER BY cv.rank
        """)
        
        print("\nCase variants created:")
        for record in result:
            print(f"   Rank {record['rank']}: {record['variant_string']} (frequency: {record['frequency']})")
        
        # Get case chunks
        result = session.run("""
            MATCH (cc:CaseChunk)-[:DESCRIBES]->(cv:CaseVariant)
            RETURN cc.variant_string as variant_string, 
                   cc.text as full_text
            ORDER BY cc.id
        """)
        
        print("\nCase variant chunks created:")
        for record in result:
            print(f"   Variant: {record['variant_string']}")
            print(f"   Full Text: {record['full_text']}")
            print("-" * 60)  # Add separator between chunks

def graphrag_query_interface(rag):
    """Enhanced GraphRAG-powered query interface with process mining domain expertise"""
    print("\n" + "="*80)
    print("PROCESS MINING EXPERT - GraphRAG Interface (Case-Based)")
    print("="*80)
    print("I'm your process mining expert assistant using CASE-BASED chunking. I can help you understand:")
    print("Process flows and activity relationships")
    print("Different process variants and execution patterns") 
    print("Case similarities and differences")
    print("Process optimization opportunities")
    print("Activity patterns and frequencies")
    print("Root cause analysis")
    print("\nType 'quit' to exit, 'help' for more examples")
    print("-" * 80)
    
    # Main interaction loop
    while True:
        question = input("\nProcess Mining Question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nThanks for using Process Mining Expert! Keep optimizing those processes!")
            break
        
        if question.lower() in ['help', 'examples', '?']:
            show_help()  # Use the global function
            continue
        
        if question:
            try:
                print("\n🔄 Analyzing process data with domain expertise (case-based)...")
                
                # Use original question for retrieval (this goes to Neo4j search)
                result = rag.search(question)
                
                # Enhance the LLM response with process mining context
                enhanced_prompt = f"""{PROCESS_MINING_CONTEXT}

Manufacturing Process Context: You are analyzing a manufacturing process with activities like Material Preparation, CNC Programming, Turning Process, Quality Inspection, etc. You have access to CASE-BASED chunks that describe different process variants and execution patterns found across different cases.

Retrieved Information: {result.answer}

User Question: {question}

Please provide a detailed process mining analysis based on the retrieved information:"""
                
                # Get enhanced response from LLM with process mining context
                from neo4j_graphrag.llm import OpenAILLM
                llm = OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0.1})
                enhanced_answer = llm.invoke(enhanced_prompt)
                
                # Display the enhanced response
                print(f"\nAnswer: {enhanced_answer.content}")
                
            except Exception as e:
                print(f"\nAnalysis Error: {e}")
                print("Try rephrasing your question or type 'help' for examples")

def graphrag_query_interface_basic(rag):
    """Basic GraphRAG-powered query interface WITHOUT enhanced prompting"""
    print("\n" + "="*80)
    print("BASIC GRAPHRAG INTERFACE (Case-Based, No Enhanced Prompting)")
    print("="*80)
    print("This is the raw GraphRAG output without domain expertise enhancement.")
    print("Type 'quit' to exit, 'help' for examples")
    print("-" * 80)
    
    # Main interaction loop
    while True:
        question = input("\nBasic GraphRAG Question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nThanks for testing basic GraphRAG!")
            break
        
        if question.lower() in ['help', 'examples', '?']:
            show_help()  # Use the global function
            continue
        
        if question:
            try:
                print("\n🔄 Getting basic GraphRAG response (case-based)...")
                
                # Direct GraphRAG call without enhancement
                result = rag.search(question)
                
                # Display the raw GraphRAG response
                print(f"\nBasic Answer: {result.answer}")
                
            except Exception as e:
                print(f"\nError: {e}")
                print("Try rephrasing your question or type 'help' for examples")

def setup_environment():
    """Setup environment variables"""
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
    """Main function to run the test with case-based chunks"""
    print("Starting Neo4j + PM4py + GraphRAG Test with Case-Based Chunks")
    print("=" * 70)
    
    # Setup environment
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
    
    # Convert to PM4py log
    log = prepare_pm4py_log(df)
    
    # Discover process model
    dfg, start_activities, end_activities = discover_process_model(log)
    
    # Extract case variants
    print("Extracting case variants for RAG...")
    variant_stats = extract_case_variants(df, min_cases_per_variant=1)
    
    # Generate case-based chunks
    print("Generating case-based chunks for RAG...")
    chunks = generate_case_based_chunks(dfg, start_activities, end_activities, variant_stats)
    print(f"   Generated {len(chunks)} case-based chunks")
    
    # Connect to Neo4j
    driver = connect_neo4j(neo4j_uri, neo4j_user, neo4j_password)
    
    if driver:
        # Store process model with case chunks
        store_case_chunks_in_neo4j(driver, dfg, start_activities, end_activities, chunks, variant_stats)
        
        # Query the data to verify
        query_neo4j(driver)
        
        # Setup GraphRAG with case chunks
        rag = setup_case_chunk_graphrag(driver)
        
        if rag:
            # Ask user which mode they want to test
            print("\n" + "="*60)
            print("TESTING MODE SELECTION (CASE-BASED)")
            print("="*60)
            print("1. Basic GraphRAG (no enhanced prompting)")
            print("2. Enhanced GraphRAG (with process mining expertise)")
            print("3. Compare both side-by-side")
            
            while True:
                choice = input("\nSelect mode (1/2/3): ").strip()
                
                if choice == '1':
                    graphrag_query_interface_basic(rag)
                    break
                elif choice == '2':
                    graphrag_query_interface(rag)
                    break
                elif choice == '3':
                    print("\n" + "="*60)
                    print("COMPARISON MODE (CASE-BASED)")
                    print("="*60)
                    print("Enter your question to see both basic and enhanced responses")
                    
                    while True:
                        question = input("\nComparison Question (or 'quit'): ").strip()
                        
                        if question.lower() in ['quit', 'exit', 'q']:
                            break
                        
                        if question:
                            try:
                                print("\n" + "-"*40)
                                print("BASIC GRAPHRAG RESPONSE:")
                                print("-"*40)
                                basic_result = rag.search(question)
                                print(f"{basic_result.answer}")
                                
                                print("\n" + "-"*40)
                                print("ENHANCED RESPONSE:")
                                print("-"*40)
                                
                                # Process mining context (same as before)
                                enhanced_prompt = f"""{PROCESS_MINING_CONTEXT}

Manufacturing Process Context: You are analyzing a manufacturing process with activities like Material Preparation, CNC Programming, Turning Process, Quality Inspection, etc. You have access to CASE-BASED chunks that describe different process variants and execution patterns found across different cases.

Retrieved Information: {basic_result.answer}

User Question: {question}

Please provide a detailed process mining analysis based on the retrieved information:"""
                                
                                from neo4j_graphrag.llm import OpenAILLM
                                llm = OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0.1})
                                enhanced_answer = llm.invoke(enhanced_prompt)
                                print(f"{enhanced_answer.content}")
                                
                                print("\n" + "="*60)
                                
                            except Exception as e:
                                print(f"Error: {e}")
                    break
                else:
                    print("Please enter 1, 2, or 3")
        else:
            print("Failed to setup GraphRAG. Check your OpenAI API key.")
        
        # Close connection
        driver.close()
        print("\nDisconnected from Neo4j. Goodbye!")
    else:
        print("Cannot proceed without Neo4j connection.")

if __name__ == "__main__":
    main()