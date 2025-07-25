import openai
from config.settings import Config
from data.data_loader import load_csv_data, build_activity_case_map, list_part_descs, filter_by_part_desc
from data.data_processor import prepare_pm4py_log, discover_process_model, extract_case_variants, extract_process_paths
from chunking.activity_chunker import generate_activity_based_chunks
from chunking.process_chunker import generate_process_based_chunks
from chunking.variant_chunker import generate_variant_based_chunks
from embeddings.local_embedder import get_local_embedder
from database.neo4j_manager import connect_neo4j, force_clean_neo4j_indexes
from database.data_storage import store_chunks_in_neo4j
from retrieval.enhanced_retriever import setup_enhanced_retriever
from interface.query_interface import graphrag_query_interface
from llm.llm_factory import get_current_model_info, get_embedding_model_name, get_reranker_model_name
import torch
from visualization.dfg_visualization import visualize_dfg, export_dfg_data
from visualization.performance_dfg_visualization import visualize_performance_dfg
from utils.logging_utils import log  # <-- Add this import

config = Config()

log(f"DEBUG: main.py USE_RERANKER = {config.USE_RERANKER}", level="debug")

def setup_openai():
    """Setup OpenAI API (only if using OpenAI)"""
    if config.LLM_TYPE.lower() == "openai" and config.OPENAI_API_KEY:
        openai.api_key = config.OPENAI_API_KEY
    elif config.LLM_TYPE.lower() == "openai":
        log("Warning: OPENAI_API_KEY not found but LLM_TYPE is set to openai", level="warning")

def display_model_info():
    """Display current model configuration"""
    model_info = get_current_model_info()
    log(f"LLM Configuration:", level="info")
    log(f"   - Type: {model_info['type'].upper()}", level="info")
    log(f"   - Model: {model_info['model_name']}", level="info")
    log(f"   - Temperature: {model_info['temperature']}", level="info")
    log(f"   - Endpoint: {model_info['base_url']}", level="info")
    log(f"   - Embedding Model: {get_embedding_model_name()}", level="info")
    log(f"   - Reranker Model: {get_reranker_model_name()}", level="info")

def main():
    log("Starting Neo4j + PM4py + GraphRAG Test with LOCAL EMBEDDING MODEL", level="info")
    log("=" * 80, level="info")
    
    # Display model configuration
    display_model_info()
    log("=" * 80, level="info")
    
    # Setup
    if config.LLM_TYPE.lower() == "openai":
        setup_openai()
    
    try:
        # Load and process data
        df = load_csv_data()
        
        # List all available part_descs and let user select one
        parts = list_part_descs(df)
        log(f"Available part_descs: {parts}", level="info")
        # Prompt user until valid input or blank
        while True:
            selected_part = input("Enter part_desc to filter (or leave blank for all): ").strip()
            if not selected_part:
                log("No filtering applied.", level="info")
                break
            if selected_part not in parts:
                log(f"Typo detected: '{selected_part}' not found in available part_descs. Please try again.", level="warning")
            else:
                df = filter_by_part_desc(df, selected_part)
                log(f"Filtered to part_desc: {selected_part} ({len(df)} events)", level="info")
                break

        event_log = prepare_pm4py_log(df)
        dfg, start_activities, end_activities, performance_dfgs = discover_process_model(event_log)

        # Visualize DFG
        visualize_dfg(dfg, start_activities, end_activities, output_path="dfg_pm4py.png")
        export_dfg_data(dfg, start_activities, end_activities, output_path="dfg_relationships.csv")
        
        # Visualize Performance DFG
        visualize_performance_dfg(performance_dfgs['mean'], start_activities, end_activities, output_path="performance_dfg_pm4py.png")
        
        # Generate chunks
        log("Generating activity-based chunks...", level="info")
        activity_chunks = generate_activity_based_chunks(
            dfg, start_activities, end_activities, build_activity_case_map(df)
        )
        log(f"   Generated {len(activity_chunks)} activity-based chunks", level="info")
        
        log("Extracting process paths...", level="info")
        frequent_paths, path_performance = extract_process_paths(dfg, performance_dfgs, min_frequency=1)
        
        log("Generating process-based chunks...", level="info")
        process_chunks = generate_process_based_chunks(
            frequent_paths, path_performance
        )
        log(f"   Generated {len(process_chunks)} process-based chunks", level="info")
        
        log("Extracting case variants...", level="info")
        variant_stats = extract_case_variants(event_log, min_cases_per_variant=1)
        
        log("Generating variant-based chunks...", level="info")
        variant_chunks = generate_variant_based_chunks(dfg, start_activities, end_activities, variant_stats)
        log(f"   Generated {len(variant_chunks)} variant-based chunks", level="info")
        
        # Database operations
        driver = connect_neo4j()
        if not driver:
            log("Cannot proceed without Neo4j connection.", level="error")
            return
        
        try:
            # Force clean indexes first
            force_clean_neo4j_indexes(driver)
            
            # 1. Load embedding model on GPU for chunking
            local_embedder_gpu = get_local_embedder(device="cuda")
            
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
                log("Failed to setup GraphRAG retrievers. Check your local embedding model and Neo4j indexes.", level="error")
        
        finally:
            driver.close()
            log("\nDisconnected from Neo4j. Goodbye!", level="info")
            
    except Exception as e:
        log(f"Error in main execution: {e}", level="error")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()