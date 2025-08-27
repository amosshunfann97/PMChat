"""
State management module for the Process Mining Chatbot Streamlit app.
Handles session state, data processing, and application state logic.
"""

import streamlit as st
from pmchatbot.src.data.data_loader import load_csv_data as original_load_csv_data, list_part_descs, filter_by_part_desc
from pmchatbot.src.data.data_processor import prepare_pm4py_log, discover_process_model, extract_case_variants, extract_process_paths
from pmchatbot.src.chunking.activity_chunker import generate_activity_based_chunks
from pmchatbot.src.chunking.process_chunker import generate_process_based_chunks
from pmchatbot.src.chunking.variant_chunker import generate_variant_based_chunks
from pmchatbot.src.embeddings.local_embedder import get_local_embedder
from pmchatbot.src.database.neo4j_manager import connect_neo4j
from pmchatbot.src.database.data_storage import store_chunks_in_neo4j
from pmchatbot.src.visualization.dfg_visualization import visualize_dfg
from pmchatbot.src.visualization.performance_dfg_visualization import visualize_performance_dfg
from pmchatbot.src.retrieval.enhanced_retriever import setup_enhanced_retriever
from pmchatbot.src.llm.llm_factory import get_llm, get_dynamic_llm


@st.cache_data
def load_csv_data():
    """Cache data loading function."""
    df = original_load_csv_data()
    return df


def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "processing_started" not in st.session_state:
        st.session_state.processing_started = False
    if "dfg_image_ready" not in st.session_state:
        st.session_state.dfg_image_ready = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "llm_config" not in st.session_state:
        # Default to ollama, configuration will be read from .env
        st.session_state.llm_config = {
            "type": "ollama",
            "valid": True
        }
    
    # Initialize LLM if not already present
    if "llm" not in st.session_state and st.session_state.llm_config.get("valid", False):
        try:
            st.session_state.llm = get_dynamic_llm(st.session_state.llm_config)
        except Exception as e:
            st.error(f"Failed to initialize LLM on startup: {str(e)}")
            st.session_state.llm = None


def clear_session_for_new_processing():
    """Clear session state for new processing."""
    st.session_state.processing_started = True
    st.session_state.dfg_image_ready = False
    st.session_state.chat_history = []
    
    # Clear any retriever references to ensure clean state
    retriever_keys = ["retriever_activity", "retriever_process", "retriever_variant", "llm"]
    for key in retriever_keys:
        if key in st.session_state:
            del st.session_state[key]


def get_application_state():
    """Get current application state flags."""
    processing_started = st.session_state.get("processing_started", False)
    dfg_ready = st.session_state.get("dfg_image_ready", False)
    return processing_started, dfg_ready


def process_data_and_setup_graphrag(selected_part):
    """
    Process the selected part data and setup GraphRAG components.
    This includes data processing, visualization generation, and retriever setup.
    """
    # Load and filter data
    df = load_csv_data()
    filtered_df = filter_by_part_desc(df, selected_part)
    
    # Process data for process mining
    event_log = prepare_pm4py_log(filtered_df)
    dfg, start_activities, end_activities, performance_dfgs, rework_cases = discover_process_model(event_log)

    # Generate chunks for different perspectives
    activity_case_map = filtered_df.groupby('activity')['case_id'].apply(list).to_dict()
    activity_chunks = generate_activity_based_chunks(dfg, start_activities, end_activities, activity_case_map, rework_cases)

    frequent_paths, path_performance = extract_process_paths(dfg, performance_dfgs, min_frequency=1)
    process_chunks = generate_process_based_chunks(frequent_paths, path_performance)

    variant_stats = extract_case_variants(event_log, min_cases_per_variant=1)
    variant_chunks = generate_variant_based_chunks(dfg, start_activities, end_activities, variant_stats)

    # Setup Neo4j and embeddings
    driver = connect_neo4j()
    local_embedder = get_local_embedder(device="cpu")
    
    if driver and local_embedder:
        # Store chunks in Neo4j
        store_chunks_in_neo4j(
            driver, dfg, start_activities, end_activities,
            activity_chunks, process_chunks, variant_chunks,
            frequent_paths, variant_stats, local_embedder
        )

        # Setup retrievers for GraphRAG
        st.session_state.retriever_activity = setup_enhanced_retriever(driver, "ActivityChunk", local_embedder)
        st.session_state.retriever_process = setup_enhanced_retriever(driver, "ProcessChunk", local_embedder)
        st.session_state.retriever_variant = setup_enhanced_retriever(driver, "VariantChunk", local_embedder)
        
        # Initialize LLM with current configuration
        try:
            st.session_state.llm = get_dynamic_llm(st.session_state.llm_config)
        except Exception as e:
            st.error(f"Failed to initialize LLM: {str(e)}")
            st.session_state.llm = None
        
        success = True
    else:
        st.error("Neo4j connection or local embedder failed.")
        success = False

    # Generate visualizations
    _generate_and_store_visualizations(dfg, start_activities, end_activities, performance_dfgs)
    
    return success


def _generate_and_store_visualizations(dfg, start_activities, end_activities, performance_dfgs):
    """Generate DFG visualizations and store them in session state."""
    # Visualize DFG and Performance DFG
    visualize_dfg(dfg, start_activities, end_activities, output_path="dfg_pm4py.png")
    visualize_performance_dfg(performance_dfgs['mean'], start_activities, end_activities, output_path="performance_dfg_pm4py.png")
    
    # Store images in session state
    with open("dfg_pm4py.png", "rb") as f:
        st.session_state.dfg_image = f.read()
    with open("performance_dfg_pm4py.png", "rb") as f:
        st.session_state.performance_dfg_image = f.read()
    
    st.session_state.dfg_image_ready = True


def has_retrievers_ready():
    """Check if all retrievers are ready in session state."""
    return all(key in st.session_state for key in ["retriever_activity", "retriever_process", "retriever_variant"])


def add_chat_message(role, content):
    """Add a message to chat history."""
    st.session_state.chat_history.append({"role": role, "content": content})


def get_chat_history():
    """Get the current chat history."""
    return st.session_state.chat_history


def update_llm_configuration(llm_config):
    """Update LLM configuration and reinitialize LLM if needed."""
    # Always update the configuration
    st.session_state.llm_config = llm_config
    
    # Check if LLM type changed by comparing with current LLM instance
    current_llm_type = None
    if hasattr(st.session_state, 'llm') and st.session_state.llm is not None:
        # Determine current LLM type from the instance
        from pmchatbot.src.llm.ollama_llm import OllamaLLM
        from neo4j_graphrag.llm import OpenAILLM
        if isinstance(st.session_state.llm, OpenAILLM):
            current_llm_type = "openai"
        elif isinstance(st.session_state.llm, OllamaLLM):
            current_llm_type = "ollama"
    
    # Reinitialize LLM if type changed and config is valid
    if llm_config.get("type") != current_llm_type and llm_config.get("valid", False):
        try:
            st.session_state.llm = get_dynamic_llm(llm_config)
            return True
        except Exception as e:
            st.error(f"Failed to update LLM configuration: {str(e)}")
            st.session_state.llm = None
            return False
    elif not llm_config.get("valid", False):
        # Clear LLM if configuration is invalid
        st.session_state.llm = None
    
    return llm_config.get("valid", False)


def get_llm_configuration():
    """Get current LLM configuration."""
    return st.session_state.get("llm_config", {})