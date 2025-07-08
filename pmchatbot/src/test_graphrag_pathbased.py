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
from collections import defaultdict, deque

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
    "Which activities could be parallelized?"
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

def generate_path_based_chunks(dfg, start_activities, end_activities, frequent_paths):
    """Generate chunks based on process paths and their context (Strategy 2)"""
    print("Generating path-based process model chunks...")
    chunks = []
    
    for i, (path, frequency) in enumerate(frequent_paths):
        # Create natural language description for this path
        path_str = " → ".join(path)
        text = f"Process path '{path_str}' is a common sequence in this process that occurs {frequency} times. "
        
        # Add path characteristics
        path_length = len(path)
        text += f"This is a {path_length}-step sequence. "
        
        # Check if path starts with a start activity
        if path[0] in start_activities:
            text += f"This path begins the process starting from {path[0]}. "
        
        # Check if path ends with an end activity  
        if path[-1] in end_activities:
            text += f"This path concludes the process ending at {path[-1]}. "
        
        # Add context about individual steps in the path
        text += "The sequence involves: "
        for j, activity in enumerate(path):
            if j == 0:
                text += f"starting with {activity}"
            elif j == len(path) - 1:
                text += f", and ending with {activity}"
            else:
                text += f", then {activity}"
        text += ". "
        
        # Add information about what comes before and after this path
        predecessor_activities = set()
        successor_activities = set()
        
        # Find activities that lead to this path
        for (src, tgt), count in dfg.items():
            if tgt == path[0]:  # Activities that lead to the start of this path
                predecessor_activities.add(src)
            if src == path[-1]:  # Activities that follow the end of this path
                successor_activities.add(tgt)
        
        if predecessor_activities:
            pred_list = list(predecessor_activities)
            text += f"This path is typically preceded by: {', '.join(pred_list)}. "
        
        if successor_activities:
            succ_list = list(successor_activities)
            text += f"This path is typically followed by: {', '.join(succ_list)}. "
        
        # Add frequency context
        total_paths = len(frequent_paths)
        rank = i + 1
        text += f"This is the {rank} most frequent path out of {total_paths} common paths identified. "
        
        # Create path model data
        path_model = {
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
            "type": "path_chunk",
            "path_string": path_str,
            "source": "path_based_chunking",
            "data": path_model
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

def store_path_chunks_in_neo4j(driver, dfg, start_activities, end_activities, chunks, frequent_paths):
    """Store path-based chunks with embeddings for RAG"""
    print("Storing path-based chunks and RAG data in Neo4j...")
    
    with driver.session() as session:
        # Clear existing data and indexes
        session.run("MATCH (n) DETACH DELETE n")
        print("   - Cleared existing data")
        
        # Drop existing indexes to avoid conflicts
        try:
            session.run("DROP INDEX path_chunk_vector_index IF EXISTS")
            session.run("DROP INDEX path_chunk_fulltext_index IF EXISTS")
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
        
        # Create ProcessPath nodes for the frequent paths
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
            
            # Link ProcessPath to Activities
            for j, activity in enumerate(path):
                session.run("""
                    MATCH (pp:ProcessPath {id: $path_id})
                    MATCH (a:Activity {name: $activity})
                    MERGE (pp)-[:CONTAINS {position: $position}]->(a)
                """, path_id=i, activity=activity, position=j)
        
        print(f"   - Created {len(frequent_paths)} ProcessPath nodes")
        
        # Create path-based chunks with embeddings
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("   - Warning: No OpenAI API key, RAG will not work properly")
            return
        
        print("   - Creating path chunk embeddings...")
        for i, chunk in enumerate(chunks):
            try:
                # Generate embedding for the path chunk
                response = openai.embeddings.create(
                    input=chunk["text"], 
                    model="text-embedding-3-large"
                )
                embedding = response.data[0].embedding
                
                # Create path chunk node with embedding
                session.run("""
                    CREATE (pc:PathChunk {
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
                
                # Link chunk to the corresponding ProcessPath node
                session.run("""
                    MATCH (pc:PathChunk {id: $chunk_id})
                    MATCH (pp:ProcessPath {id: $path_id})
                    MERGE (pc)-[:DESCRIBES]->(pp)
                """, chunk_id=i, path_id=i)
                
                print(f"   - Created path chunk {i+1}/{len(chunks)} for '{chunk['path_string']}'")
                
            except Exception as e:
                print(f"   - Error creating path chunk {i}: {e}")
        
        # Wait for nodes to be committed
        time.sleep(2)
        
        # Create fulltext index first
        try:
            session.run("""
                CREATE FULLTEXT INDEX path_chunk_fulltext_index IF NOT EXISTS
                FOR (pc:PathChunk) ON EACH [pc.text, pc.path_string]
            """)
            print("   - Created fulltext index")
        except Exception as e:
            print(f"   - Warning: Fulltext index creation: {e}")
        
        # Create vector index with improved error handling
        vector_index_created = False
        try:
            # First, check if any PathChunk nodes exist
            result = session.run("MATCH (pc:PathChunk) RETURN count(pc) as count")
            node_count = result.single()["count"]
            print(f"   - Found {node_count} PathChunk nodes for indexing")
            
            if node_count == 0:
                print("   - Warning: No PathChunk nodes found, vector index may fail")
                return
            
            # Try to create vector index
            try:
                session.run("""
                    CREATE VECTOR INDEX path_chunk_vector_index IF NOT EXISTS
                    FOR (pc:PathChunk) ON (pc.embedding)
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

def setup_path_chunk_graphrag(driver):
    """Setup GraphRAG with path-based chunks"""
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
                
                # Look for vector index on PathChunk.embedding
                vector_index_name = None
                vector_index_ready = False
                
                for idx in all_indexes:
                    if (idx['type'] == 'VECTOR' and 
                        'PathChunk' in str(idx['labelsOrTypes']) and 
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
                        'PathChunk' in str(idx['labelsOrTypes'])):
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
            
            # Enhanced retrieval query for path-based chunks
            retrieval_query = """
                MATCH (node)-[:DESCRIBES]->(path:ProcessPath)
                
                // Get activities in this path
                OPTIONAL MATCH (path)-[c:CONTAINS]->(activity:Activity)
                
                // Get context about what leads to and from this path
                OPTIONAL MATCH (first_activity:Activity)<-[:CONTAINS {position: 0}]-(path)
                OPTIONAL MATCH (last_activity:Activity)<-[:CONTAINS]-(path)
                WHERE NOT EXISTS((path)-[:CONTAINS {position: 0}]->(last_activity))
                
                OPTIONAL MATCH (prev:Activity)-[:NEXT]->(first_activity)
                OPTIONAL MATCH (last_activity)-[:NEXT]->(next:Activity)
                
                WITH node, path,
                     collect(DISTINCT activity.name + '(' + toString(c.position) + ')') as path_activities,
                     collect(DISTINCT prev.name) as predecessors,
                     collect(DISTINCT next.name) as successors
                
                RETURN node.text + 
                       ' [Path Context: ' + path.path_string + 
                       ' (rank: ' + toString(path.rank) + ', frequency: ' + toString(path.frequency) + ')' +
                       CASE WHEN size(predecessors) > 0 
                            THEN ' typically preceded by: ' + reduce(s = '', x IN predecessors[..3] | s + x + ', ')
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
        
        print("Path-based GraphRAG setup completed successfully!")
        return rag
        
    except Exception as e:
        print(f"Error setting up GraphRAG: {e}")
        traceback.print_exc()
        return None

def query_neo4j(driver):
    """Run basic queries to verify the process model and path chunks"""
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
        
        # Get process paths
        result = session.run("""
            MATCH (pp:ProcessPath)
            RETURN pp.path_string as path_string, 
                   pp.frequency as frequency,
                   pp.rank as rank
            ORDER BY pp.rank
        """)
        
        print("\nProcess paths created:")
        for record in result:
            print(f"   Rank {record['rank']}: {record['path_string']} (frequency: {record['frequency']})")
        
        # Get path chunks
        result = session.run("""
            MATCH (pc:PathChunk)-[:DESCRIBES]->(pp:ProcessPath)
            RETURN pc.path_string as path_string, 
                   pc.text as full_text
            ORDER BY pc.id
        """)
        
        print("\nPath chunks created:")
        for record in result:
            print(f"   Path: {record['path_string']}")
            print(f"   Full Text: {record['full_text']}")
            print("-" * 60)  # Add separator between chunks

def graphrag_query_interface(rag):
    """Enhanced GraphRAG-powered query interface with process mining domain expertise"""
    print("\n" + "="*80)
    print("PROCESS MINING EXPERT - GraphRAG Interface (Path-Based)")
    print("="*80)
    print("I'm your process mining expert assistant using PATH-BASED chunking. I can help you understand:")
    print("Process flows and activity relationships")
    print("Common process paths and sequences") 
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
                print("\n🔄 Analyzing process data with domain expertise (path-based)...")
                
                # First, show which chunks were retrieved
                print("\n📋 RETRIEVED CHUNKS:")
                print("-" * 50)
                try:
                    search_result = rag.retriever.search(question, top_k=5)
                    
                    # Handle the structured response format
                    if isinstance(search_result, dict):
                        # If it's a dictionary, look for 'items' key
                        items = search_result.get('items', search_result.get('results', []))
                    elif hasattr(search_result, 'items'):
                        # If it has an items attribute
                        items = search_result.items
                    elif isinstance(search_result, list):
                        # If it's directly a list
                        items = search_result
                    else:
                        # Try to iterate over it directly
                        items = list(search_result) if search_result else []
                    
                    # Display the retrieved items
                    for i, item in enumerate(items, 1):
                        if hasattr(item, 'content'):
                            # Extract the actual content from the Record wrapper
                            content = str(item.content)
                            if content.startswith('<Record text="') and content.endswith('">'):
                                # Remove the Record wrapper
                                actual_content = content[14:-2]  # Remove '<Record text="' and '">'
                            else:
                                actual_content = content
                        
                            print(f"Chunk {i}: {actual_content[:150]}...")
                            
                            if hasattr(item, 'metadata') and item.metadata:
                                print(f"   Metadata: {item.metadata}")
                            else:
                                print(f"   Metadata: None")
                                
                            # Check if there's a score attribute
                            if hasattr(item, 'score'):
                                print(f"   Score: {item.score:.4f}")
                            else:
                                print(f"   Score: Not available")
                        else:
                            print(f"Chunk {i}: {str(item)[:150]}...")
                        print()
                        
                except Exception as retrieval_error:
                    print(f"Could not display retrieval details: {retrieval_error}")
                    print(f"Raw search result type: {type(search_result)}")
                    print(f"Raw search result: {str(search_result)[:200]}...")
                    print("Proceeding with query...")
                
                # Use original question for retrieval (this goes to Neo4j search)
                result = rag.search(question)
                
                # Enhance the LLM response with process mining context
                enhanced_prompt = f"""{PROCESS_MINING_CONTEXT}

Manufacturing Process Context: You are analyzing a manufacturing process with activities like Material Preparation, CNC Programming, Turning Process, Quality Inspection, etc. You have access to PATH-BASED chunks that describe common sequences of activities.

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
    print("BASIC GRAPHRAG INTERFACE (Path-Based, No Enhanced Prompting)")
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
                print("\n🔄 Getting basic GraphRAG response (path-based)...")
                
                # First, show which chunks were retrieved
                print("\n📋 RETRIEVED CHUNKS:")
                print("-" * 50)
                try:
                    search_result = rag.retriever.search(question, top_k=5)
                    
                    # Handle the structured response format
                    if isinstance(search_result, dict):
                        # If it's a dictionary, look for 'items' key
                        items = search_result.get('items', search_result.get('results', []))
                    elif hasattr(search_result, 'items'):
                        # If it has an items attribute
                        items = search_result.items
                    elif isinstance(search_result, list):
                        # If it's directly a list
                        items = search_result
                    else:
                        # Try to iterate over it directly
                        items = list(search_result) if search_result else []
                    
                    # Display the retrieved items
                    for i, item in enumerate(items, 1):
                        if hasattr(item, 'content'):
                            # Extract the actual content from the Record wrapper
                            content = str(item.content)
                            if content.startswith('<Record text="') and content.endswith('">'):
                                # Remove the Record wrapper
                                actual_content = content[14:-2]  # Remove '<Record text="' and '">'
                            else:
                                actual_content = content
                        
                            print(f"Chunk {i}: {actual_content[:150]}...")
                            
                            if hasattr(item, 'metadata') and item.metadata:
                                print(f"   Metadata: {item.metadata}")
                            else:
                                print(f"   Metadata: None")
                                
                            # Check if there's a score attribute
                            if hasattr(item, 'score'):
                                print(f"   Score: {item.score:.4f}")
                            else:
                                print(f"   Score: Not available")
                        else:
                            print(f"Chunk {i}: {str(item)[:150]}...")
                        print()
                        
                except Exception as retrieval_error:
                    print(f"Could not display retrieval details: {retrieval_error}")
                    print(f"Raw search result type: {type(search_result)}")
                    print(f"Raw search result: {str(search_result)[:200]}...")
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
    """Main function to run the test with path-based chunks"""
    print("Starting Neo4j + PM4py + GraphRAG Test with Path-Based Chunks")
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
    
    # Extract frequent process paths
    print("Extracting frequent process paths for RAG...")
    frequent_paths = extract_process_paths(df, min_frequency=1)
    
    # Generate path-based chunks
    print("Generating path-based chunks for RAG...")
    chunks = generate_path_based_chunks(dfg, start_activities, end_activities, frequent_paths)
    print(f"   Generated {len(chunks)} path-based chunks")
    
    # Connect to Neo4j
    driver = connect_neo4j(neo4j_uri, neo4j_user, neo4j_password)
    
    if driver:
        # Store process model with path chunks
        store_path_chunks_in_neo4j(driver, dfg, start_activities, end_activities, chunks, frequent_paths)
        
        # Query the data to verify
        query_neo4j(driver)
        
        # Setup GraphRAG with path chunks
        rag = setup_path_chunk_graphrag(driver)
        
        if rag:
            # Ask user which mode they want to test
            print("\n" + "="*60)
            print("TESTING MODE SELECTION (PATH-BASED)")
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
                    print("COMPARISON MODE (PATH-BASED)")
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

Manufacturing Process Context: You are analyzing a manufacturing process with activities like Material Preparation, CNC Programming, Turning Process, Quality Inspection, etc. You have access to PATH-BASED chunks that describe common sequences of activities.

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