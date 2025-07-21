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
from retrieval.enhanced_retriever import setup_enhanced_retriever
from interface.query_interface import graphrag_query_interface
from llm.llm_factory import get_current_model_info
import torch

config = Config()

print("DEBUG: main.py USE_RERANKER =", config.USE_RERANKER)

def setup_openai():
    """Setup OpenAI API (only if using OpenAI)"""
    if config.LLM_TYPE.lower() == "openai" and config.OPENAI_API_KEY:
        openai.api_key = config.OPENAI_API_KEY
    elif config.LLM_TYPE.lower() == "openai":
        print("Warning: OPENAI_API_KEY not found but LLM_TYPE is set to openai")

def display_model_info():
    """Display current model configuration"""
    model_info = get_current_model_info()
    print(f"LLM Configuration:")
    print(f"   - Type: {model_info['type'].upper()}")
    print(f"   - Model: {model_info['model_name']}")
    print(f"   - Temperature: {model_info['temperature']}")
    print(f"   - Endpoint: {model_info['base_url']}")

def main():
    print("Starting Neo4j + PM4py + GraphRAG Test with LOCAL EMBEDDING MODEL")
    print("=" * 80)
    
    # Display model configuration
    display_model_info()
    print("=" * 80)
    
    # Setup
    if config.LLM_TYPE.lower() == "openai":
        setup_openai()
    
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
            
            # 1. Load embedding model on GPU for chunking
            local_embedder_gpu = get_local_embedder(device="cuda")  # Modify get_local_embedder to accept device
            
            # 2. Use embedding model for chunking and retriever setup
            store_chunks_in_neo4j(
                driver, dfg, start_activities, end_activities, 
                activity_chunks, process_chunks, variant_chunks, 
                frequent_paths, variant_stats, local_embedder_gpu
            )
            # Release GPU model and clear cache
            del local_embedder_gpu
            import torch
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            # 3. Load embedding model on CPU for query embedding
            local_embedder_cpu = get_local_embedder(device="cpu")
            
            # 4. Pass CPU embedder to enhanced retrievers
            retriever_activity = setup_enhanced_retriever(driver, "ActivityChunk", local_embedder_cpu, use_reranker=None)
            retriever_process = setup_enhanced_retriever(driver, "ProcessChunk", local_embedder_cpu, use_reranker=None)
            retriever_variant = setup_enhanced_retriever(driver, "VariantChunk", local_embedder_cpu, use_reranker=None)

            if retriever_activity and retriever_process and retriever_variant:
                # 5. Release CPU embedder and clear GPU cache
                del local_embedder_cpu
                import torch
                torch.cuda.empty_cache()
                
                # 6. Use EnhancedRetriever objects directly for queries
                graphrag_query_interface(retriever_activity, retriever_process, retriever_variant)
                
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