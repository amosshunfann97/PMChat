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
from utils.logging_utils import log

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
    log("Starting Neo4j + PM4py + GraphRAG locally...", level="info")
    log("=" * 80, level="info")
    display_model_info()
    log("=" * 80, level="info")

    if config.LLM_TYPE.lower() == "openai":
        setup_openai()

    driver = connect_neo4j()
    if not driver:
        log("Cannot proceed without Neo4j connection.", level="error")
        return

    try:
        while True:
            # --- PART SELECTION ---
            df = load_csv_data()
            parts = list_part_descs(df)
            log(f"Available parts: {parts}", level="info")
            while True:
                selected_part = input("Enter part's name for process discovery (or leave blank for all): ").strip()
                if not selected_part:
                    log("No filtering applied.", level="info")
                    break
                if selected_part not in parts:
                    log(f"Typo detected: '{selected_part}' not found in available parts. Please try again.", level="warning")
                else:
                    df = filter_by_part_desc(df, selected_part)
                    log(f"Filtered to part_desc: {selected_part} ({len(df)} events)", level="info")
                    break

            event_log = prepare_pm4py_log(df)
            dfg, start_activities, end_activities, performance_dfgs = discover_process_model(event_log)
            visualize_dfg(dfg, start_activities, end_activities, output_path="dfg_pm4py.png")
            export_dfg_data(dfg, start_activities, end_activities, output_path="dfg_relationships.csv")
            visualize_performance_dfg(performance_dfgs['mean'], start_activities, end_activities, output_path="performance_dfg_pm4py.png")

            log("Generating activity-based chunks...", level="debug")
            activity_chunks = generate_activity_based_chunks(
                dfg, start_activities, end_activities, build_activity_case_map(df)
            )
            log(f"   Generated {len(activity_chunks)} activity-based chunks", level="debug")

            log("Extracting process paths...", level="debug")
            frequent_paths, path_performance = extract_process_paths(dfg, performance_dfgs, min_frequency=1)

            log("Generating process-based chunks...", level="debug")
            process_chunks = generate_process_based_chunks(
                frequent_paths, path_performance
            )
            log(f"   Generated {len(process_chunks)} process-based chunks", level="debug")

            log("Extracting case variants...", level="debug")
            variant_stats = extract_case_variants(event_log, min_cases_per_variant=1)

            log("Generating variant-based chunks...", level="debug")
            variant_chunks = generate_variant_based_chunks(dfg, start_activities, end_activities, variant_stats)
            log(f"   Generated {len(variant_chunks)} variant-based chunks", level="debug")

            # --- STORE CHUNKS ---
            force_clean_neo4j_indexes(driver)
            local_embedder_gpu = get_local_embedder(device="cuda")
            store_chunks_in_neo4j(
                driver, dfg, start_activities, end_activities,
                activity_chunks, process_chunks, variant_chunks,
                frequent_paths, variant_stats, local_embedder_gpu
            )
            del local_embedder_gpu
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            local_embedder_cpu = get_local_embedder(device="cpu")
            retriever_activity = setup_enhanced_retriever(driver, "ActivityChunk", local_embedder_cpu, use_reranker=None)
            retriever_process = setup_enhanced_retriever(driver, "ProcessChunk", local_embedder_cpu, use_reranker=None)
            retriever_variant = setup_enhanced_retriever(driver, "VariantChunk", local_embedder_cpu, use_reranker=None)
            del local_embedder_cpu
            torch.cuda.empty_cache()

            if retriever_activity and retriever_process and retriever_variant:
                # --- QUERY INTERFACE ---
                go_back = graphrag_query_interface(retriever_activity, retriever_process, retriever_variant, selected_part=selected_part if selected_part else "All")
                if go_back:
                    log("Returning to part selection...", level="info")
                    continue  # Restart the loop for new part selection
                else:
                    break  # Exit the main loop
            else:
                log("Failed to setup GraphRAG retrievers. Check your local embedding model and Neo4j indexes.", level="error")
                break

    finally:
        driver.close()
        log("\nDisconnected from Neo4j. Goodbye!", level="info")

if __name__ == "__main__":
    main()