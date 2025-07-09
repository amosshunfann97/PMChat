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
        
        # Calculate total execution count
        total_incoming = sum(count for _, count in incoming)
        total_outgoing = sum(count for _, count in outgoing)
        
        # For activities, execution count is typically the max of incoming or outgoing
        # For start activities, use outgoing count; for end activities, use incoming count
        if activity in start_activities:
            execution_count = total_outgoing if total_outgoing > 0 else start_activities[activity]
        elif activity in end_activities:
            execution_count = total_incoming if total_incoming > 0 else end_activities[activity]
        else:
            execution_count = max(total_incoming, total_outgoing) if total_incoming > 0 or total_outgoing > 0 else 0
        
        # Create natural language description for this activity
        text = f"{activity} is an activity in this process. "
        
        # Add execution count as second sentence - FOR ALL ACTIVITIES
        text += f"This activity executed {execution_count} times. "
        
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
        
        # Create activity model data for potential graph queries
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

def setup_activity_chunk_graphrag(driver):
    """Setup GraphRAG with activity-based chunks"""
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
                
                # Look for vector index on ActivityChunk.embedding
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
                return None
        
        # Setup embedder for query encoding
        embedder = OpenAIEmbeddings(model="text-embedding-3-large")
        
        # Custom formatter to extract scores and metadata - SAME AS CASE-BASED VERSION
        def custom_result_formatter(record):
            from neo4j_graphrag.types import RetrieverResultItem
            
            # Extract score from the record
            score = record.get("score", None)
            
            # Extract additional metadata from the record
            record_metadata = record.get("metadata", {})
            
            # Create comprehensive metadata
            metadata = {}
            if score is not None:
                metadata["score"] = score
            
            # Add custom metadata if available
            if record_metadata:
                metadata.update(record_metadata)
            
            # Extract content
            content = record.get("text", str(record))
            
            return RetrieverResultItem(
                content=content,
                metadata=metadata if metadata else None,
            )
        
        # Use HybridCypherRetriever (requires both indexes)
        if vector_index_ready and fulltext_index_ready and vector_index_name and fulltext_index_name:
            print(f"Using HybridCypherRetriever with vector index '{vector_index_name}' and fulltext index '{fulltext_index_name}'...")
            
            # Enhanced retrieval query for activity-based chunks - SAME PATTERN AS CASE-BASED
            retrieval_query = """
                MATCH (node)-[:DESCRIBES]->(activity:Activity)
                
                OPTIONAL MATCH (activity)-[r1:NEXT]->(next:Activity)
                OPTIONAL MATCH (prev:Activity)-[r2:NEXT]->(activity)
                
                WITH node, activity, score,  // ← Include score from the base hybrid search
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
                       AS text,
                       score AS score,  // ← Return the score
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
            
            retriever = HybridCypherRetriever(
                driver=driver,
                vector_index_name=vector_index_name,
                fulltext_index_name=fulltext_index_name,
                retrieval_query=retrieval_query,
                embedder=embedder,
                result_formatter=custom_result_formatter 
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
        
        print("Activity-based GraphRAG setup completed successfully!")
        return rag
        
    except Exception as e:
        print(f"Error setting up GraphRAG: {e}")
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
                   ac.text as full_text
            ORDER BY ac.id
        """)
        
        print("\nActivity chunks created:")
        for record in result:
            print(f"   Activity: {record['activity']}")
            print(f"   Full Text: {record['full_text']}")
            print("-" * 60)  # Add separator between chunks

def graphrag_query_interface(rag):
    """Enhanced GraphRAG-powered query interface with process mining domain expertise"""
    print("\n" + "="*80)
    print("PROCESS MINING EXPERT - GraphRAG Interface (Activity-Based)")
    print("="*80)
    print("I'm your process mining expert assistant using ACTIVITY-BASED chunking. I can help you understand:")
    print("Process flows and activity relationships")
    print("Bottlenecks and process inefficiencies") 
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
            show_help()
            continue
        
        if question:
            try:
                print("\nAnalyzing process data with domain expertise (activity-based)...")
                
                # First, show which chunks were retrieved
                print("\nRETRIEVED CHUNKS:")
                print("-" * 50)
                try:
                    search_result = rag.retriever.search(question, top_k=5)
                    
                    # Extract items from the RetrieverResult
                    items = search_result.items if hasattr(search_result, 'items') else []
                    
                    # Extract query metadata (contains query vector and other retrieval info)
                    query_metadata = search_result.metadata if hasattr(search_result, 'metadata') else {}
                    
                    # Display the retrieved items
                    for i, item in enumerate(items, 1):
                        # Extract content from Neo4j Record format
                        content = str(item.content)
                        
                        # Clean up Neo4j Record wrapper format
                        if content.startswith('<Record text="') and content.endswith('">'):
                            actual_content = content[14:-2]  # Remove '<Record text="' and '">'
                        else:
                            actual_content = content
                        
                        print(f"Chunk {i}: {actual_content[:150]}...")
                        
                        # Individual items metadata
                        print(f"   Individual Metadata: {item.metadata}")
                        
                        # Similarity scores are not exposed by HybridCypherRetriever
                        print(f"   Similarity Score: Not available (HybridCypherRetriever limitation)")
                        
                        print()
                    
                    # Show query-level metadata if available
                    if query_metadata:
                        print("📊 QUERY METADATA:")
                        print(f"   Query Vector Dimension: {len(query_metadata.get('query_vector', []))}")
                        print(f"   Available Metadata Keys: {list(query_metadata.keys())}")
                        print()
                        
                except Exception as retrieval_error:
                    print(f"Could not display retrieval details: {retrieval_error}")
                    print("Proceeding with query...")
                
                # Use original question for retrieval (this goes to Neo4j search)
                result = rag.search(question)
                
                # Enhance the LLM response with process mining context
                enhanced_prompt = f"""{PROCESS_MINING_CONTEXT}

Manufacturing Process Context: You are analyzing a manufacturing process with activities like Material Preparation, CNC Programming, Turning Process, Quality Inspection, etc. You have access to ACTIVITY-BASED chunks that describe individual activities and their relationships.

Retrieved Information: {result.answer}

User Question: {question}

Please provide a detailed process mining analysis based on the retrieved information:"""
                
                # Get enhanced response from LLM with process mining context
                from neo4j_graphrag.llm import OpenAILLM
                llm = OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0.1})
                enhanced_answer = llm.invoke(enhanced_prompt)
                
                # Display the enhanced response
                print(f"\n💡 ENHANCED ANSWER:")
                print("-" * 50)
                print(f"{enhanced_answer.content}")
                
            except Exception as e:
                print(f"\nAnalysis Error: {e}")
                print("Try rephrasing your question or type 'help' for examples")

def graphrag_query_interface_basic(rag):
    """Basic GraphRAG-powered query interface WITHOUT enhanced prompting"""
    print("\n" + "="*80)
    print("BASIC GRAPHRAG INTERFACE (Activity-Based, No Enhanced Prompting)")
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
            show_help()
            continue
        
        if question:
            try:
                print("\n🔄 Getting basic GraphRAG response (activity-based)...")
                
                # First, show which chunks were retrieved
                print("\n📋 RETRIEVED CHUNKS:")
                print("-" * 50)
                try:
                    search_result = rag.retriever.search(question, top_k=5)
                    
                    # Extract items from the RetrieverResult
                    items = search_result.items if hasattr(search_result, 'items') else []
                    
                    # Extract query metadata (contains query vector and other retrieval info)
                    query_metadata = search_result.metadata if hasattr(search_result, 'metadata') else {}
                    
                    # Display the retrieved items
                    for i, item in enumerate(items, 1):
                        # Extract content from Neo4j Record format
                        content = str(item.content)
                        
                        # Clean up Neo4j Record wrapper format
                        if content.startswith('<Record text="') and content.endswith('">'):
                            actual_content = content[14:-2]  # Remove '<Record text="' and '">'
                        else:
                            actual_content = content
                        
                        print(f"Chunk {i}: {actual_content[:150]}...")
                        
                        # Individual items metadata
                        print(f"   Individual Metadata: {item.metadata}")
                        
                        print()
                    
                    # Show query-level metadata if available
                    if query_metadata:
                        print("📊 QUERY METADATA:")
                        print(f"   Query Vector Dimension: {len(query_metadata.get('query_vector', []))}")
                        print(f"   Available Metadata Keys: {list(query_metadata.keys())}")
                        print()
                        
                except Exception as retrieval_error:
                    print(f"Could not display retrieval details: {retrieval_error}")
                    print("Proceeding with query...")
                
                # Direct GraphRAG call without enhancement
                result = rag.search(question)
                
                # Display the raw GraphRAG response
                print(f"\n🤖 BASIC ANSWER:")
                print("-" * 50)
                print(f"{result.answer}")
                
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
    """Main function to run the test with activity-based chunks"""
    print("Starting Neo4j + PM4py + GraphRAG Test with Activity-Based Chunks")
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
            # Ask user which mode they want to test
            print("\n" + "="*60)
            print("TESTING MODE SELECTION (ACTIVITY-BASED)")
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
                    print("COMPARISON MODE (ACTIVITY-BASED)")
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

Manufacturing Process Context: You are analyzing a manufacturing process with activities like Material Preparation, CNC Programming, Turning Process, Quality Inspection, etc. You have access to ACTIVITY-BASED chunks that describe individual activities and their relationships.

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