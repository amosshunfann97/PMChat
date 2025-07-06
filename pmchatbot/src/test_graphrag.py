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

# Load environment variables
load_dotenv()

def create_sample_data():
    """Create sample process mining data for testing"""
    data = {
        'case_id': ['Case_1', 'Case_1', 'Case_1', 'Case_2', 'Case_2', 'Case_2', 
                    'Case_3', 'Case_3', 'Case_3', 'Case_4', 'Case_4', 'Case_4', 
                    'Case_5', 'Case_5', 'Case_5'],
        'activity': [
            'Submit Invoice', 'Review Invoice', 'Approve Invoice',
            'Submit Invoice', 'Review Invoice', 'Reject Invoice',
            'Submit Invoice', 'Review Invoice', 'Approve Invoice',
            'Submit Invoice', 'Review Invoice', 'Approve Invoice',
            'Submit Invoice', 'Review Invoice', 'Reject Invoice'
        ],
        'timestamp': [
            '2024-01-01 09:00:00', '2024-01-01 10:00:00', '2024-01-01 11:00:00',
            '2024-01-02 09:00:00', '2024-01-02 10:30:00', '2024-01-02 11:30:00',
            '2024-01-03 09:00:00', '2024-01-03 10:15:00', '2024-01-03 11:15:00',
            '2024-01-04 09:00:00', '2024-01-04 10:45:00', '2024-01-04 11:45:00',
            '2024-01-05 09:00:00', '2024-01-05 10:20:00', '2024-01-05 11:20:00'
        ]
    }
    
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.to_csv("events.csv", index=False)
    print("Created events.csv with sample data")
    return df

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

def generate_activity_based_chunks(dfg, start_activities, end_activities):
    """Generate chunks based on individual activities and their context (Strategy 1)"""
    print("Generating activity-based process model chunks...")
    chunks = []
    
    # Get all unique activities
    all_activities = set()
    for (src, tgt), count in dfg.items():
        all_activities.add(src)
        all_activities.add(tgt)
    
    for activity in all_activities:
        # Get incoming and outgoing edges for this activity
        incoming = [(src, count) for (src, tgt), count in dfg.items() if tgt == activity]
        outgoing = [(tgt, count) for (src, tgt), count in dfg.items() if src == activity]
        
        # Create natural language description for this activity
        text = f"{activity} is an activity in this process. "
        
        # Add start/end information
        if activity in start_activities:
            text += f"{activity} is a starting activity that begins the process ({start_activities[activity]} cases start here). "
        
        if activity in end_activities:
            text += f"{activity} is an ending activity that concludes the process ({end_activities[activity]} cases end here). "
        
        # Add incoming connections
        if incoming:
            incoming_desc = [f"{src} ({count} times)" for src, count in incoming]
            text += f"{activity} is preceded by: {', '.join(incoming_desc)}. "
        
        # Add outgoing connections
        if outgoing:
            outgoing_desc = [f"{tgt} ({count} times)" for tgt, count in outgoing]
            text += f"{activity} is followed by: {', '.join(outgoing_desc)}. "
        
        # Add behavioral context
        total_incoming = sum(count for _, count in incoming)
        total_outgoing = sum(count for _, count in outgoing)
        
        if total_incoming > 0 and total_outgoing > 0:
            text += f"This activity processes {total_incoming} incoming flows and produces {total_outgoing} outgoing flows. "
        
        # Create activity model data for potential graph queries
        activity_model = {
            "activity": activity,
            "incoming": incoming,
            "outgoing": outgoing,
            "is_start": activity in start_activities,
            "is_end": activity in end_activities,
            "start_frequency": start_activities.get(activity, 0),
            "end_frequency": end_activities.get(activity, 0)
        }
        
        chunks.append({
            "text": text.strip(),
            "type": "activity_chunk",
            "activity_name": activity,
            "source": "activity_based_chunking",
            "data": activity_model
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

def store_activity_chunks_in_neo4j(driver, dfg, start_activities, end_activities, chunks):
    """Store activity-based chunks with embeddings for RAG"""
    print("Storing activity-based chunks and RAG data in Neo4j...")
    
    with driver.session() as session:
        # Clear existing data and indexes
        session.run("MATCH (n) DETACH DELETE n")
        print("   - Cleared existing data")
        
        # Drop existing indexes to avoid conflicts
        try:
            session.run("DROP INDEX activity_chunk_vector_index IF EXISTS")
            session.run("DROP INDEX activity_chunk_fulltext_index IF EXISTS")
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
        
        # Create activity-based chunks with embeddings
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("   - Warning: No OpenAI API key, RAG will not work properly")
            return
        
        print("   - Creating activity chunk embeddings...")
        for i, chunk in enumerate(chunks):
            try:
                # Generate embedding for the activity chunk
                response = openai.embeddings.create(
                    input=chunk["text"], 
                    model="text-embedding-3-large"
                )
                embedding = response.data[0].embedding
                
                # Create activity chunk node with embedding
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
                
                # Link chunk to the corresponding activity node
                session.run("""
                    MATCH (ac:ActivityChunk {id: $chunk_id})
                    MATCH (a:Activity {name: $activity_name})
                    MERGE (ac)-[:DESCRIBES]->(a)
                """, chunk_id=i, activity_name=chunk["activity_name"])
                
                print(f"   - Created activity chunk {i+1}/{len(chunks)} for '{chunk['activity_name']}'")
                
            except Exception as e:
                print(f"   - Error creating activity chunk {i}: {e}")
        
        # Wait for nodes to be committed
        import time
        time.sleep(2)
        
        # Create fulltext index first
        try:
            session.run("""
                CREATE FULLTEXT INDEX activity_chunk_fulltext_index IF NOT EXISTS
                FOR (ac:ActivityChunk) ON EACH [ac.text, ac.activity_name]
            """)
            print("   - Created fulltext index")
        except Exception as e:
            print(f"   - Warning: Fulltext index creation: {e}")
        
        # Create vector index with improved error handling
        vector_index_created = False
        try:
            # First, check if any ActivityChunk nodes exist
            result = session.run("MATCH (ac:ActivityChunk) RETURN count(ac) as count")
            node_count = result.single()["count"]
            print(f"   - Found {node_count} ActivityChunk nodes for indexing")
            
            if node_count == 0:
                print("   - Warning: No ActivityChunk nodes found, vector index may fail")
                return
            
            # Try different vector index creation approaches
            try:
                # Method 1: Standard syntax
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
                print("   - Created vector index (standard syntax)")
                vector_index_created = True
            except Exception as e1:
                print(f"   - Standard syntax failed: {e1}")
                
                # Method 2: Alternative syntax without backticks
                try:
                    session.run("""
                        CREATE VECTOR INDEX activity_chunk_vector_index IF NOT EXISTS
                        FOR (ac:ActivityChunk) ON (ac.embedding)
                        OPTIONS {
                            indexConfig: {
                                "vector.dimensions": 3072,
                                "vector.similarity_function": "cosine"
                            }
                        }
                    """)
                    print("   - Created vector index (alternative syntax)")
                    vector_index_created = True
                except Exception as e2:
                    print(f"   - Alternative syntax failed: {e2}")
                    
                    # Method 3: Basic syntax without options
                    try:
                        session.run("""
                            CREATE VECTOR INDEX activity_chunk_vector_index IF NOT EXISTS
                            FOR (ac:ActivityChunk) ON (ac.embedding)
                        """)
                        print("   - Created vector index (basic syntax)")
                        vector_index_created = True
                    except Exception as e3:
                        print(f"   - Basic syntax failed: {e3}")
            
            if vector_index_created:
                print("   - Vector index creation completed (will be verified during GraphRAG setup)")
            else:
                print("   - Error: Could not create vector index with any method")
                
        except Exception as e:
            print(f"   - Critical error in vector index creation: {e}")
            import traceback
            traceback.print_exc()

def setup_activity_chunk_graphrag(driver):
    """Setup GraphRAG with activity-based chunks"""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OpenAI API key not found. RAG requires embeddings.")
        return None
    
    try:
        # Check if indexes exist and are online with detailed debugging
        with driver.session() as session:
            try:
                # List ALL indexes for debugging
                result = session.run("SHOW INDEXES YIELD name, state, type, labelsOrTypes, properties")
                all_indexes = list(result)
                print("All indexes in database:")
                for idx in all_indexes:
                    print(f"   - {idx['name']}: {idx['state']} ({idx['type']}) - {idx['labelsOrTypes']} {idx['properties']}")
                
                # Look for any vector index on ActivityChunk.embedding
                vector_index_name = None
                vector_index_ready = False
                
                for idx in all_indexes:
                    if (idx['type'] == 'VECTOR' and 
                        'ActivityChunk' in str(idx['labelsOrTypes']) and 
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
                        'ActivityChunk' in str(idx['labelsOrTypes'])):
                        fulltext_index_name = idx['name']
                        fulltext_index_ready = idx['state'] == 'ONLINE'
                        print(f"Found fulltext index: {fulltext_index_name} - {idx['state']}")
                        break
                
                print(f"Vector index ready: {vector_index_ready} (name: {vector_index_name})")
                print(f"Fulltext index ready: {fulltext_index_ready} (name: {fulltext_index_name})")
                
            except Exception as e:
                print(f"Error checking indexes: {e}")
                vector_index_ready = False
                fulltext_index_ready = False
                vector_index_name = None
                fulltext_index_name = None
        
        # Setup embedder for query encoding
        embedder = OpenAIEmbeddings(model="text-embedding-3-large")
        
        # Try to use HybridCypherRetriever if both indexes are available
        if vector_index_ready and fulltext_index_ready and vector_index_name and fulltext_index_name:
            print(f"Using HybridCypherRetriever with vector index '{vector_index_name}' and fulltext index '{fulltext_index_name}'...")
            
            # Enhanced retrieval query
            retrieval_query = """
                MATCH (node)-[:DESCRIBES]->(activity:Activity)
                
                // Get related activities for context
                OPTIONAL MATCH (activity)-[r1:NEXT]->(next:Activity)
                OPTIONAL MATCH (prev:Activity)-[r2:NEXT]->(activity)
                
                WITH node, activity,
                     collect(DISTINCT next.name + '(' + toString(r1.count) + ')') as next_activities,
                     collect(DISTINCT prev.name + '(' + toString(r2.count) + ')') as prev_activities
                
                RETURN node.text + 
                       ' [Graph Context: ' + activity.name + 
                       CASE WHEN size(next_activities) > 0 
                            THEN ' leads to: ' + reduce(s = '', x IN next_activities[..3] | s + x + ', ')
                            ELSE '' END +
                       CASE WHEN size(prev_activities) > 0 
                            THEN ' follows: ' + reduce(s = '', x IN prev_activities[..3] | s + x + ', ')
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
            
        elif vector_index_ready and vector_index_name:
            print(f"Using VectorCypherRetriever with vector index '{vector_index_name}'...")
            
            from neo4j_graphrag.retrievers import VectorCypherRetriever
            
            retrieval_query = """
                MATCH (node)-[:DESCRIBES]->(activity:Activity)
                
                OPTIONAL MATCH (activity)-[r1:NEXT]->(next:Activity)
                OPTIONAL MATCH (prev:Activity)-[r2:NEXT]->(activity)
                
                WITH node, activity,
                     collect(DISTINCT next.name + '(' + toString(r1.count) + ')') as next_activities,
                     collect(DISTINCT prev.name + '(' + toString(r2.count) + ')') as prev_activities
                
                RETURN node.text + 
                       ' [Graph Context: ' + activity.name + 
                       CASE WHEN size(next_activities) > 0 
                            THEN ' leads to: ' + reduce(s = '', x IN next_activities[..3] | s + x + ', ')
                            ELSE '' END +
                       CASE WHEN size(prev_activities) > 0 
                            THEN ' follows: ' + reduce(s = '', x IN prev_activities[..3] | s + x + ', ')
                            ELSE '' END + ']'
                       AS text
            """
            
            retriever = VectorCypherRetriever(
                driver=driver,
                index_name=vector_index_name,
                retrieval_query=retrieval_query,
                embedder=embedder
            )
            
        else:
            print("Vector index not ready. Let's try to fix this...")
            
            # Check if there are nodes with embeddings
            with driver.session() as session:
                result = session.run("""
                    MATCH (ac:ActivityChunk) 
                    WHERE ac.embedding IS NOT NULL 
                    RETURN count(ac) as count, 
                           size(ac.embedding) as embedding_size 
                    LIMIT 1
                """)
                record = result.single()
                
                if record and record['count'] > 0:
                    print(f"   - Found {record['count']} nodes with embeddings of size {record['embedding_size']}")
                    
                    # Drop ALL existing vector indexes on ActivityChunk.embedding
                    try:
                        # Get all vector indexes on ActivityChunk.embedding
                        result = session.run("""
                            SHOW INDEXES YIELD name, type, labelsOrTypes, properties
                            WHERE type = 'VECTOR' 
                            AND 'ActivityChunk' IN labelsOrTypes 
                            AND 'embedding' IN properties
                            RETURN name
                        """)
                        
                        existing_vector_indexes = [record['name'] for record in result]
                        print(f"   - Found existing vector indexes: {existing_vector_indexes}")
                        
                        for index_name in existing_vector_indexes:
                            try:
                                session.run(f"DROP INDEX `{index_name}`")
                                print(f"   - Dropped index: {index_name}")
                            except Exception as e:
                                print(f"   - Failed to drop index {index_name}: {e}")
                        
                        import time
                        time.sleep(3)
                        
                        # Create new vector index with a unique name
                        new_index_name = "activity_chunk_vector_index"
                        session.run(f"""
                            CREATE VECTOR INDEX `{new_index_name}`
                            FOR (ac:ActivityChunk) ON (ac.embedding)
                            OPTIONS {{
                                indexConfig: {{
                                    `vector.dimensions`: {record['embedding_size']},
                                    `vector.similarity_function`: 'cosine'
                                }}
                            }}
                        """)
                        print(f"   - Created vector index: {new_index_name}")
                        
                        # Wait for it to come online
                        vector_index_created = False
                        for i in range(10):
                            time.sleep(3)
                            result = session.run(f"""
                                SHOW INDEXES 
                                YIELD name, state 
                                WHERE name = '{new_index_name}'
                                RETURN state
                            """)
                            index_state = result.single()
                            if index_state and index_state['state'] == 'ONLINE':
                                print(f"   - Vector index '{new_index_name}' is now ONLINE")
                                vector_index_created = True
                                break
                            else:
                                print(f"   - Vector index state: {index_state['state'] if index_state else 'NOT_FOUND'}")
                        
                        if vector_index_created:
                            print(f"Successfully created and using vector index: {new_index_name}")
                            from neo4j_graphrag.retrievers import VectorCypherRetriever
                            
                            retrieval_query = """
                                MATCH (node)-[:DESCRIBES]->(activity:Activity)
                                
                                OPTIONAL MATCH (activity)-[r1:NEXT]->(next:Activity)
                                OPTIONAL MATCH (prev:Activity)-[r2:NEXT]->(activity)
                                
                                WITH node, activity,
                                     collect(DISTINCT next.name + '(' + toString(r1.count) + ')') as next_activities,
                                     collect(DISTINCT prev.name + '(' + toString(r2.count) + ')') as prev_activities
                                
                                RETURN node.text + 
                                       ' [Graph Context: ' + activity.name + 
                                       CASE WHEN size(next_activities) > 0 
                                            THEN ' leads to: ' + reduce(s = '', x IN next_activities[..3] | s + x + ', ')
                                            ELSE '' END +
                                       CASE WHEN size(prev_activities) > 0 
                                            THEN ' follows: ' + reduce(s = '', x IN prev_activities[..3] | s + x + ', ')
                                            ELSE '' END + ']'
                                       AS text
                            """
                            
                            retriever = VectorCypherRetriever(
                                driver=driver,
                                index_name=new_index_name,
                                retrieval_query=retrieval_query,
                                embedder=embedder
                            )
                        else:
                            print("Failed to create working vector index")
                            return None
                            
                    except Exception as e:
                        print(f"   - Failed to recreate vector index: {e}")
                        import traceback
                        traceback.print_exc()
                        return None
                else:
                    print("   - No ActivityChunk nodes with embeddings found")
                    return None
        
        # Setup LLM
        llm = OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0.1})
        
        # Setup GraphRAG
        rag = GraphRAG(
            retriever=retriever,
            llm=llm
        )
        
        print("GraphRAG setup completed successfully!")
        return rag
        
    except Exception as e:
        print(f"Error setting up GraphRAG: {e}")
        import traceback
        traceback.print_exc()
        return None

def query_neo4j(driver):
    """Run basic queries to verify the process model and activity chunks"""
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
        
        # Get activity chunks
        result = session.run("""
            MATCH (ac:ActivityChunk)-[:DESCRIBES]->(a:Activity)
            RETURN ac.activity_name as activity, 
                   substring(ac.text, 0, 80) + '...' as chunk_preview
            ORDER BY ac.id
        """)
        
        print("\nActivity chunks created:")
        for record in result:
            print(f"   Activity: {record['activity']}")
            print(f"   Preview: {record['chunk_preview']}")
            print()

def graphrag_query_interface(rag):
    """GraphRAG-powered query interface with activity chunks"""
    print("\nGraphRAG Query Interface with Activity Chunks (type 'quit' to exit):")
    print("Ask natural language questions about the process activities:")
    print("Examples:")
    print("  - 'What does the Submit Invoice activity do?'")
    print("  - 'What happens after Review Invoice?'")
    print("  - 'Which activities start the process?'")
    print("  - 'How does the approval process work?'")
    print("  - 'What are the different outcomes after reviewing?'")
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if question:
            try:
                print("Processing with activity-based chunks and graph retrieval...")
                result = rag.search(question)
                print(f"\nAnswer: {result.answer}")
            except Exception as e:
                print(f"Error: {e}")

def setup_environment():
    """Setup environment variables"""
    neo4j_uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if openai_api_key:
        openai.api_key = openai_api_key
    else:
        print("Warning: OPENAI_API_KEY not found in environment")
    
    return neo4j_uri, neo4j_user, neo4j_password, openai_api_key

def main():
    """Main function to run the test with activity-based chunks"""
    print("Starting Neo4j + PM4py + GraphRAG Test with Activity-Based Chunks")
    print("=" * 70)
    
    # Setup environment
    neo4j_uri, neo4j_user, neo4j_password, openai_api_key = setup_environment()
    
    if not openai_api_key:
        print("Error: OpenAI API key is required for RAG functionality!")
        return
    
    csv_file_path = r"C:\Users\shunf\RoadToMaster\PMChat\running_example_manufacturing.csv"
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
    
    # Generate activity-based chunks
    print("Generating activity-based chunks for RAG...")
    chunks = generate_activity_based_chunks(dfg, start_activities, end_activities)
    print(f"   Generated {len(chunks)} activity-based chunks")
    
    # Connect to Neo4j
    driver = connect_neo4j(neo4j_uri, neo4j_user, neo4j_password)
    
    if driver:
        # Store process model with activity chunks
        store_activity_chunks_in_neo4j(driver, dfg, start_activities, end_activities, chunks)
        
        # Query the data to verify
        query_neo4j(driver)
        
        # Setup GraphRAG with activity chunks
        rag = setup_activity_chunk_graphrag(driver)
        
        if rag:
            # Use GraphRAG interface with activity chunks
            graphrag_query_interface(rag)
        else:
            print("Failed to setup GraphRAG. Check your OpenAI API key.")
        
        # Close connection
        driver.close()
        print("\nDisconnected from Neo4j. Goodbye!")
    else:
        print("Cannot proceed without Neo4j connection.")

if __name__ == "__main__":
    main()