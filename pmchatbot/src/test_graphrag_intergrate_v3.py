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
from sentence_transformers import SentenceTransformer

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

def generate_activity_based_chunks(dfg, start_activities, end_activities, activity_case_map):
    print("Generating activity-based process model chunks...")
    chunks = []
    all_activities = set()
    for (src, tgt), count in dfg.items():
        all_activities.add(src)
        all_activities.add(tgt)

    execution_counts = {}
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
        execution_counts[activity] = execution_count

    max_count = max(execution_counts.values())
    min_count = min(execution_counts.values())

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
        text = f"{activity} is an activity in this workflow. "
        text += f"This activity executed {execution_count} times. "
        
        if execution_count == max_count:
            text += f"This activity has the highest frequency among all activities. "
        if execution_count == min_count:
            text += f"This activity has the lowest frequency among all activities. "
        if activity in start_activities:
            text += f"{activity} is a starting activity that begins the workflow ({start_activities[activity]} cases start here). "
        if activity in end_activities:
            text += f"{activity} is an ending activity that concludes the workflow ({end_activities[activity]} cases end here). "
        if incoming:
            incoming_desc = [f"{src} ({count} times)" for src, count in incoming]
            text += f"{activity} is preceded by: {', '.join(incoming_desc)}. "
        if outgoing:
            outgoing_desc = [f"{tgt} ({count} times)" for tgt, count in outgoing]
            text += f"{activity} is followed by: {', '.join(outgoing_desc)}. "
        case_ids = activity_case_map.get(activity, [])
        text += f"This activity appears in {len(case_ids)} cases. "
        if case_ids:
            text += f"Related Case IDs: {', '.join(case_ids)}. "
        activity_model = {
            "activity": activity,
            "incoming": incoming,
            "outgoing": outgoing,
            "is_start": activity in start_activities,
            "is_end": activity in end_activities,
            "start_frequency": start_activities.get(activity, 0),
            "end_frequency": end_activities.get(activity, 0),
            "execution_count": execution_count,
            "case_ids": case_ids
        }
        chunks.append({
            "text": text.strip(),
            "type": "activity_chunk",
            "activity_name": activity,
            "source": "activity_based_chunking",
            "data": activity_model
        })
    return chunks

def extract_process_paths(df, min_frequency=1):
    """Extract 2-activity process paths from the event log"""
    print(f"Extracting 2-activity process paths (min frequency: {min_frequency})...")
    
    # Group by case and get activity sequences
    case_sequences = []
    for case_id, group in df.groupby('case_id'):
        # Sort by timestamp to get correct order
        sequence = group.sort_values('timestamp')['activity'].tolist()
        case_sequences.append(sequence)
    
    # Extract only 2-activity paths (direct transitions)
    path_frequencies = defaultdict(int)
    
    for sequence in case_sequences:
        # Extract only paths of length 2 (direct activity transitions)
        for i in range(len(sequence) - 1):
            path = tuple(sequence[i:i + 2])  # Only 2-activity paths
            path_frequencies[path] += 1
    
    # Filter by minimum frequency
    frequent_paths = {path: freq for path, freq in path_frequencies.items() 
                     if freq >= min_frequency}
    
    # Sort by frequency
    sorted_paths = sorted(frequent_paths.items(), key=lambda x: x[1], reverse=True)
    
    print(f"   Found {len(sorted_paths)} frequent 2-activity paths")
    print("   Top 10 most frequent 2-activity transitions:")
    for i, (path, freq) in enumerate(sorted_paths[:10], 1):
        path_str = " → ".join(path)
        print(f"   {i}. {path_str} (frequency: {freq})")
    
    return sorted_paths

def generate_process_based_chunks(dfg, start_activities, end_activities, frequent_paths):
    """Generate chunks based on process paths and their context"""
    print("Generating process-based process model chunks...")
    chunks = []
    
    # Find highest and lowest frequency
    if frequent_paths:
        frequencies = [freq for _, freq in frequent_paths]
        max_freq = max(frequencies)
        min_freq = min(frequencies)
    else:
        max_freq = min_freq = None

    for i, (path, frequency) in enumerate(frequent_paths):
        path_str = " → ".join(path)
        text = f"Process path '{path_str}' is a process that occurs {frequency} times. "
        path_length = len(path)
        if path[0] in start_activities:
            text += f"This path begins the process starting from {path[0]}. "
        if path[-1] in end_activities:
            text += f"This path concludes the process ending at {path[-1]}. "
        text += "The complete execution sequence is: "
        for j, activity in enumerate(path):
            if j == 0:
                text += f"starting with {activity}"
            elif j == len(path) - 1:
                text += f", and ending with {activity}"
            else:
                text += f", followed by {activity}"
        text += ". "
        predecessor_activities = set()
        successor_activities = set()
        for (src, tgt), count in dfg.items():
            if tgt == path[0]:
                predecessor_activities.add(src)
            if src == path[-1]:
                successor_activities.add(tgt)
        if predecessor_activities:
            pred_list = list(predecessor_activities)
            text += f"This path is typically preceded by: {', '.join(pred_list)}. "
        if successor_activities:
            succ_list = list(successor_activities)
            text += f"This path is typically followed by: {', '.join(succ_list)}. "
        total_paths = len(frequent_paths)
        rank = i + 1
        text += f"This is the {rank} most frequent process path out of {total_paths} processes identified. "

        if frequency == max_freq:
            text += f"This path has the highest frequency among all process paths. "
        if frequency == min_freq:
            text += f"This path has the lowest frequency among all process paths. "
        process_model = {
            "path": path,
            "path_string": path_str,
            "frequency": frequency,
            "length": path_length,
            "rank": rank,
            "starts_process": path[0] in start_activities,
            "ends_process": path[-1] in end_activities,
            "predecessors": list(predecessor_activities),
            "successors": list(successor_activities)
        }
        chunks.append({
            "text": text.strip(),
            "type": "process_chunk",
            "path_string": path_str,
            "source": "process_based_chunking",
            "data": process_model
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

def generate_variant_based_chunks(dfg, start_activities, end_activities, variant_stats):
    """Generate chunks based on case variants and their characteristics"""
    print("Generating variant-based process model chunks...")
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
        text = f"Process variant '{variant_str}' represents a distinct execution pattern found in {frequency} cases ({frequency/total_cases*100:.1f}% of all cases). (→ means 'followed by'). "
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
        text += f"On average, cases following this variant have {stats['avg_activities']:.1f} total activities and {stats['avg_unique_activities']:.1f} unique activities. "
        rank = i + 1
        percentage = (frequency / total_cases) * 100
        text += f"This variant representing {percentage:.1f}% of all workflow. "
        example_cases = cases
        text += f"Cases following this variant: {', '.join(example_cases)}. "
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
            "type": "variant_chunk",
            "variant_string": variant_str,
            "source": "variant_based_chunking",
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

def store_chunks_in_neo4j(driver, dfg, start_activities, end_activities, activity_chunks, process_chunks, variant_chunks, frequent_paths, variant_stats, local_embedder):
    print("Storing activity-based, process-based, and variant-based chunks and RAG data in Neo4j...")
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        print("   - Cleared existing data")
        try:
            session.run("DROP INDEX activity_chunk_vector_index IF EXISTS")
            session.run("DROP INDEX activity_chunk_fulltext_index IF EXISTS")
            session.run("DROP INDEX process_chunk_vector_index IF EXISTS")
            session.run("DROP INDEX process_chunk_fulltext_index IF EXISTS")
            session.run("DROP INDEX case_chunk_vector_index IF EXISTS")
            session.run("DROP INDEX case_chunk_fulltext_index IF EXISTS")
            session.run("DROP INDEX variant_chunk_vector_index IF EXISTS")
            session.run("DROP INDEX variant_chunk_fulltext_index IF EXISTS")
            print("   - Dropped existing indexes")
        except Exception as e:
            print(f"   - Note: {e}")
        
        # Store DFG and activities (unchanged)
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
        
        # Store activity chunks with LOCAL EMBEDDINGS
        print("   - Creating activity chunk embeddings with local model...")
        for i, chunk in enumerate(activity_chunks):
            try:
                # Use local embedder instead of OpenAI
                embedding = local_embedder.encode([chunk["text"]])[0].tolist()
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
        
        # Create ProcessPath nodes (unchanged)
        for i, (path, frequency) in enumerate(frequent_paths):
            path_str = " → ".join(path)
            session.run("""
                CREATE (pp:ProcessPath {
                    id: $id,
                    path_string: $path_string,
                    frequency: $frequency,
                    length: $length,
                    rank: $rank
                })
            """, 
            id=i,
            path_string=path_str,
            frequency=frequency,
            length=len(path),
            rank=i + 1)
            
            for j, activity in enumerate(path):
                session.run("""
                    MATCH (pp:ProcessPath {id: $path_id})
                    MATCH (a:Activity {name: $activity})
                    MERGE (pp)-[:CONTAINS {position: $position}]->(a)
                """, path_id=i, activity=activity, position=j)
        
        print(f"   - Created {len(frequent_paths)} ProcessPath nodes")
        
        # Store process chunks with LOCAL EMBEDDINGS
        print("   - Creating process chunk embeddings with local model...")
        for i, chunk in enumerate(process_chunks):
            try:
                # Use local embedder instead of OpenAI
                embedding = local_embedder.encode([chunk["text"]])[0].tolist()
                session.run("""
                    CREATE (pc:ProcessChunk {
                        id: $id,
                        text: $text,
                        path_string: $path_string,
                        type: $type,
                        source: $source,
                        embedding: $embedding
                    })
                """, 
                id=i, 
                text=chunk["text"], 
                path_string=chunk["path_string"],
                type=chunk["type"],
                source=chunk["source"],
                embedding=embedding)
                session.run("""
                    MATCH (pc:ProcessChunk {id: $chunk_id})
                    MATCH (pp:ProcessPath {id: $path_id})
                    MERGE (pc)-[:DESCRIBES]->(pp)
                """, chunk_id=i, path_id=i)
                print(f"   - Created process chunk {i+1}/{len(process_chunks)} for '{chunk['path_string']}'")
            except Exception as e:
                print(f"   - Error creating process chunk {i}: {e}")
        
        # Create CaseVariant nodes (unchanged)
        for i, stats in enumerate(variant_stats):
            variant = stats['variant']
            variant_str = " → ".join(variant)
            
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
            
            for j, activity in enumerate(variant):
                session.run("""
                    MATCH (cv:CaseVariant {id: $variant_id})
                    MATCH (a:Activity {name: $activity})
                    MERGE (cv)-[:EXECUTES {position: $position}]->(a)
                """, variant_id=i, activity=activity, position=j)
            
            for case_id in stats['cases'][:10]:
                session.run("""
                    MATCH (cv:CaseVariant {id: $variant_id})
                    CREATE (c:Case {id: $case_id})
                    MERGE (c)-[:FOLLOWS]->(cv)
                """, variant_id=i, case_id=case_id)
        
        print(f"   - Created {len(variant_stats)} CaseVariant nodes")
        
        # Store variant chunks with LOCAL EMBEDDINGS
        print("   - Creating variant-based chunk embeddings with local model...")
        for i, chunk in enumerate(variant_chunks):
            try:
                # Use local embedder instead of OpenAI
                embedding = local_embedder.encode([chunk["text"]])[0].tolist()
                session.run("""
                    CREATE (vc:VariantChunk {
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
                    MATCH (vc:VariantChunk {id: $chunk_id})
                    MATCH (cv:CaseVariant {id: $variant_id})
                    MERGE (vc)-[:DESCRIBES]->(cv)
                """, chunk_id=i, variant_id=i)
                print(f"   - Created variant chunk {i+1}/{len(variant_chunks)} for '{chunk['variant_string']}'")
            except Exception as e:
                print(f"   - Error creating variant chunk {i}: {e}")
        
        # Create indexes
        time.sleep(2)
        try:
            session.run("""
                CREATE FULLTEXT INDEX activity_chunk_fulltext_index IF NOT EXISTS
                FOR (ac:ActivityChunk) ON EACH [ac.text, ac.activity_name]
            """)
            session.run("""
                CREATE FULLTEXT INDEX process_chunk_fulltext_index IF NOT EXISTS
                FOR (pc:ProcessChunk) ON EACH [pc.text, pc.path_string]
            """)
            session.run("""
                CREATE FULLTEXT INDEX variant_chunk_fulltext_index IF NOT EXISTS
                FOR (vc:VariantChunk) ON EACH [vc.text, vc.variant_string]
            """)
            print("   - Created fulltext indexes")
        except Exception as e:
            print(f"   - Warning: Fulltext index creation: {e}")
        
        try:
            # Create vector indexes with 1024 dimensions for E5-large
            result = session.run("MATCH (ac:ActivityChunk) RETURN count(ac) as count")
            node_count = result.single()["count"]
            if node_count > 0:
                session.run("""
                    CREATE VECTOR INDEX activity_chunk_vector_index IF NOT EXISTS
                    FOR (ac:ActivityChunk) ON (ac.embedding)
                    OPTIONS {
                        indexConfig: {
                            `vector.dimensions`: 1024,
                            `vector.similarity_function`: 'cosine'
                        }
                    }
                """)
            
            result = session.run("MATCH (pc:ProcessChunk) RETURN count(pc) as count")
            node_count = result.single()["count"]
            if node_count > 0:
                session.run("""
                    CREATE VECTOR INDEX process_chunk_vector_index IF NOT EXISTS
                    FOR (pc:ProcessChunk) ON (pc.embedding)
                    OPTIONS {
                        indexConfig: {
                            `vector.dimensions`: 1024,
                            `vector.similarity_function`: 'cosine'
                        }
                    }
                """)
            
            result = session.run("MATCH (vc:VariantChunk) RETURN count(vc) as count")
            node_count = result.single()["count"]
            if node_count > 0:
                session.run("""
                    CREATE VECTOR INDEX variant_chunk_vector_index IF NOT EXISTS
                    FOR (vc:VariantChunk) ON (vc.embedding)
                    OPTIONS {
                        indexConfig: {
                            `vector.dimensions`: 1024,
                            `vector.similarity_function`: 'cosine'
                        }
                    }
                """)
            print("   - Created vector indexes")
        except Exception as e:
            print(f"   - Critical error in vector index creation: {e}")
            traceback.print_exc()

def force_clean_neo4j_indexes(driver):
    """Force clean all indexes in Neo4j"""
    print("Force cleaning all Neo4j indexes...")
    with driver.session() as session:
        try:
            # Get all existing indexes
            result = session.run("SHOW INDEXES")
            indexes = list(result)
            
            for index in indexes:
                index_name = index.get('name')
                if index_name and ('chunk' in index_name.lower() or 'vector' in index_name.lower() or 'fulltext' in index_name.lower()):
                    try:
                        session.run(f"DROP INDEX `{index_name}` IF EXISTS")
                        print(f"   - Force dropped index: {index_name}")
                    except Exception as e:
                        print(f"   - Could not drop {index_name}: {e}")
            
            # Wait for cleanup
            time.sleep(5)
            print("   - All indexes cleaned successfully")
        except Exception as e:
            print(f"   - Error cleaning indexes: {e}")

def build_activity_case_map(df):
    activity_case_map = defaultdict(set)
    for _, row in df.iterrows():
        activity_case_map[row['activity']].add(str(row['case_id']))
    # Sort the case IDs for each activity
    return {k: sorted(list(v), key=lambda x: int(x) if x.isdigit() else x) for k, v in activity_case_map.items()}

def get_local_embedder():
    """Initialize local embedding model"""
    model_path = os.getenv("EMBEDDING_MODEL_PATH", "intfloat/multilingual-e5-large")
    print(f"Loading local embedding model from: {model_path}")
    try:
        local_embedder = SentenceTransformer(model_path, device="cuda", trust_remote_code=True)
        print("Local embedding model loaded successfully on CUDA")
        return local_embedder
    except Exception as e:
        print(f"Error loading local model, falling back to CPU: {e}")
        local_embedder = SentenceTransformer(model_path, device="cpu", trust_remote_code=True)
        return local_embedder

# Replace the setup_retriever function
def setup_retriever(driver, chunk_type, local_embedder):
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
        
        # Create local embedder wrapper
        class LocalEmbeddings:
            def __init__(self, model):
                self.model = model
            
            def embed_query(self, text):
                return self.model.encode([text])[0].tolist()
            
            def embed_documents(self, texts):
                return [self.model.encode([text])[0].tolist() for text in texts]
        
        embedder = LocalEmbeddings(local_embedder)
        
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
        elif chunk_type == "ProcessChunk":
            retrieval_query = """
                MATCH (node)-[:DESCRIBES]->(path:ProcessPath)
                RETURN node.text AS text,
                       score AS score,
                       {
                           path_string: path.path_string,
                           rank: path.rank,
                           frequency: path.frequency,
                           type: node.type,
                           source: node.source
                       } AS metadata
            """
        elif chunk_type == "VariantChunk":
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

def graphrag_query_interface(rag_activity, rag_process, rag_variant):
    print("\n" + "="*80)
    print("PROCESS MINING EXPERT - GraphRAG Interface (Integrated)")
    print("="*80)
    print("I'm your process mining expert assistant. You can query:")
    print("1. Activity-based context (individual activities and their relationships)")
    print("2. Process-based context (2-activity sequences and transitions)")
    print("3. Variant-based context (process variants and execution patterns)")
    print("4. All combined (merged results)")
    print("\nType 'quit' to exit, 'help' for more examples")
    print("-" * 80)
    while True:
        mode = input("\nChoose context: (1) Activity, (2) Process, (3) Variant, (4) All: ").strip()
        if mode.lower() in ['quit', 'exit', 'q']:
            print("\nThanks for using Process Mining Expert! Keep optimizing those processes!")
            break
        if mode.lower() in ['help', 'examples', '?']:
            show_help()
            continue
        if mode not in ['1', '2', '3', '4']:
            print("Please enter 1, 2, 3, or 4")
            continue
        question = input("\nProcess Mining Question: ").strip()
        if not question:
            continue
        try:
            if mode == "1":
                rag = rag_activity
                context_label = "ACTIVITY-BASED"
            elif mode == "2":
                rag = rag_process
                context_label = "PROCESS-BASED"
            elif mode == "3":
                rag = rag_variant
                context_label = "VARIANT-BASED"
            else:
                # All: retrieve from all three and merge
                print("\nRETRIEVED CHUNKS (Activity-Based):")
                print("-" * 50)
                search_result_a = rag_activity.retriever.search(question, top_k=5)
                for i, item in enumerate(search_result_a.items, 1):
                    content = str(item.content)
                    if content.startswith('<Record text="') and content.endswith('">'):
                        actual_content = content[14:-2]
                    else:
                        actual_content = content
                    print(f"Activity Chunk {i}: {actual_content}")
                    print(f"   Metadata: {item.metadata}")
                print("\nRETRIEVED CHUNKS (Process-Based):")
                print("-" * 50)
                search_result_p = rag_process.retriever.search(question, top_k=5)
                for i, item in enumerate(search_result_p.items, 1):
                    content = str(item.content)
                    if content.startswith('<Record text="') and content.endswith('">'):
                        actual_content = content[14:-2]
                    else:
                        actual_content = content
                    print(f"Process Chunk {i}: {actual_content}")
                    print(f"   Metadata: {item.metadata}")
                print("\nRETRIEVED CHUNKS (Variant-Based):")
                print("-" * 50)
                search_result_v = rag_variant.retriever.search(question, top_k=5)
                for i, item in enumerate(search_result_v.items, 1):
                    content = str(item.content)
                    if content.startswith('<Record text="') and content.endswith('">'):
                        actual_content = content[14:-2]
                    else:
                        actual_content = content
                    print(f"Variant Chunk {i}: {actual_content}")
                    print(f"   Metadata: {item.metadata}")
                # Enhanced prompt with all three
                answer_a = rag_activity.search(question).answer
                answer_p = rag_process.search(question).answer
                answer_v = rag_variant.search(question).answer
                enhanced_prompt = f"""{PROCESS_MINING_CONTEXT}

Manufacturing Process Context: You are analyzing a manufacturing process with activities like Material Preparation, CNC Programming, Turning Process, Quality Inspection, etc. You have access to ACTIVITY-BASED, PROCESS-BASED, and VARIANT-BASED chunks.

Activity-Based Retrieved Information: {answer_a}

Process-Based Retrieved Information: {answer_p}

Variant-Based Retrieved Information: {answer_v}

User Question: {question}

Please provide a detailed process mining analysis based on the retrieved information from all three perspectives:"""
                llm = OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0.1})
                enhanced_answer = llm.invoke(enhanced_prompt)
                print(f"\n💡 ENHANCED ANSWER (ALL COMBINED):")
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
    print("Starting Neo4j + PM4py + GraphRAG Test with LOCAL EMBEDDING MODEL")
    print("=" * 70)
    
    # Initialize local embedder first
    local_embedder = get_local_embedder()
    
    neo4j_uri, neo4j_user, neo4j_password, openai_api_key, csv_file_path = setup_environment()
    
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
    activity_chunks = generate_activity_based_chunks(dfg, start_activities, end_activities, build_activity_case_map(df))
    print(f"   Generated {len(activity_chunks)} activity-based chunks")
    
    print("Extracting frequent process paths for RAG...")
    frequent_paths = extract_process_paths(df, min_frequency=1)
    
    print("Generating process-based chunks for RAG...")
    process_chunks = generate_process_based_chunks(dfg, start_activities, end_activities, frequent_paths)
    print(f"   Generated {len(process_chunks)} process-based chunks")
    
    print("Extracting case variants for RAG...")
    variant_stats = extract_case_variants(df, min_cases_per_variant=1)
    
    print("Generating variant-based chunks for RAG...")
    variant_chunks = generate_variant_based_chunks(dfg, start_activities, end_activities, variant_stats)
    print(f"   Generated {len(variant_chunks)} variant-based chunks")
    
    driver = connect_neo4j(neo4j_uri, neo4j_user, neo4j_password)
    if driver:
        # Force clean indexes first
        force_clean_neo4j_indexes(driver)
        
        # Then store chunks
        store_chunks_in_neo4j(driver, dfg, start_activities, end_activities, activity_chunks, process_chunks, variant_chunks, frequent_paths, variant_stats, local_embedder)
        
        # Pass local_embedder to retriever setup
        retriever_activity = setup_retriever(driver, "ActivityChunk", local_embedder)
        retriever_process = setup_retriever(driver, "ProcessChunk", local_embedder)
        retriever_variant = setup_retriever(driver, "VariantChunk", local_embedder)
        
        if retriever_activity and retriever_process and retriever_variant:
            rag_activity = GraphRAG(retriever=retriever_activity, llm=OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0.1}))
            rag_process = GraphRAG(retriever=retriever_process, llm=OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0.1}))
            rag_variant = GraphRAG(retriever=retriever_variant, llm=OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0.1}))
            graphrag_query_interface(rag_activity, rag_process, rag_variant)
        else:
            print("Failed to setup GraphRAG retrievers. Check your local embedding model and Neo4j indexes.")
        driver.close()
        print("\nDisconnected from Neo4j. Goodbye!")
    else:
        print("Cannot proceed without Neo4j connection.")

if __name__ == "__main__":
    main()