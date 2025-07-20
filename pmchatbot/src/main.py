import openai
from config.settings import Config
from data.data_loader import load_csv_data, build_activity_case_map
from data.pm4py_processor import prepare_pm4py_log, discover_process_model
from chunking.activity_chunker import generate_activity_based_chunks
from chunking.process_chunker import extract_process_paths, generate_process_based_chunks
from chunking.variant_chunker import extract_case_variants, generate_variant_based_chunks
from embeddings.local_embedder import get_local_embedder
from database.neo4j_manager import connect_neo4j, force_clean_neo4j_indexes
from database.data_storage import store_chunks_in_neo4j
from retrieval.retriever_setup import setup_retriever
from interface.query_interface import graphrag_query_interface
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import OpenAILLM

def setup_openai():
    """Setup OpenAI API"""
    if Config.OPENAI_API_KEY:
        openai.api_key = Config.OPENAI_API_KEY
    else:
        print("Warning: OPENAI_API_KEY not found in environment")

def main():
    print("Starting Neo4j + PM4py + GraphRAG Test with LOCAL EMBEDDING MODEL and PERFORMANCE METRICS")
    print("=" * 80)
    
    # Setup
    setup_openai()
    local_embedder = get_local_embedder()
    
    try:
        # Load and process data
        df = load_csv_data()
        log = prepare_pm4py_log(df)
        dfg, start_activities, end_activities, performance_dfgs = discover_process_model(log)
        
        # Generate chunks
        print("Generating activity-based chunks for RAG...")
        activity_chunks = generate_activity_based_chunks(
            dfg, start_activities, end_activities, build_activity_case_map(df)
        )
        print(f"   Generated {len(activity_chunks)} activity-based chunks")
        
        print("Extracting frequent process paths with performance for RAG...")
        frequent_paths, path_performance = extract_process_paths(df, performance_dfgs, min_frequency=1)
        
        print("Generating process-based chunks with performance for RAG...")
        process_chunks = generate_process_based_chunks(
            dfg, start_activities, end_activities, frequent_paths, path_performance
        )
        print(f"   Generated {len(process_chunks)} process-based chunks")
        
        print("Extracting case variants with performance for RAG...")
        variant_stats = extract_case_variants(df, performance_dfgs, min_cases_per_variant=1)
        
        print("Generating variant-based chunks with performance for RAG...")
        variant_chunks = generate_variant_based_chunks(dfg, start_activities, end_activities, variant_stats)
        print(f"   Generated {len(variant_chunks)} variant-based chunks")
        
        # Database operations
        driver = connect_neo4j()
        if not driver:
            print("Cannot proceed without Neo4j connection.")
            return
        
        try:
            # Force clean indexes first
            force_clean_neo4j_indexes(driver)
            
            # Store chunks
            store_chunks_in_neo4j(
                driver, dfg, start_activities, end_activities, 
                activity_chunks, process_chunks, variant_chunks, 
                frequent_paths, variant_stats, local_embedder
            )
            
            # Setup retrievers
            retriever_activity = setup_retriever(driver, "ActivityChunk", local_embedder)
            retriever_process = setup_retriever(driver, "ProcessChunk", local_embedder)
            retriever_variant = setup_retriever(driver, "VariantChunk", local_embedder)
            
            if retriever_activity and retriever_process and retriever_variant:
                rag_activity = GraphRAG(
                    retriever=retriever_activity, 
                    llm=OpenAILLM(model_name=Config.LLM_MODEL_NAME, model_params=Config.LLM_MODEL_PARAMS)
                )
                rag_process = GraphRAG(
                    retriever=retriever_process, 
                    llm=OpenAILLM(model_name=Config.LLM_MODEL_NAME, model_params=Config.LLM_MODEL_PARAMS)
                )
                rag_variant = GraphRAG(
                    retriever=retriever_variant, 
                    llm=OpenAILLM(model_name=Config.LLM_MODEL_NAME, model_params=Config.LLM_MODEL_PARAMS)
                )
                
                # Start query interface
                graphrag_query_interface(rag_activity, rag_process, rag_variant)
            else:
                print("Failed to setup GraphRAG retrievers. Check your local embedding model and Neo4j indexes.")
        
        finally:
            driver.close()
            print("\nDisconnected from Neo4j. Goodbye!")
            
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()