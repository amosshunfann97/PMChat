"""
Main application orchestrator for the Process Mining Chatbot.
This is the refactored streamlit_app.py with improved structure and maintainability.
"""

import streamlit as st
import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our custom modules
from pmchatbot.src.streamlit_integration.core.state_manager import (
    load_csv_data, 
    initialize_session_state, 
    clear_session_for_new_processing,
    get_application_state,
    process_data_and_setup_graphrag,
    update_llm_configuration
)
from pmchatbot.src.streamlit_integration.components.ui_components import (
    render_sidebar_header,
    render_main_header,
    render_execute_instruction,
    render_loading_overlay,
    render_fallback_message,
    display_dfg_images,
    render_part_selection_sidebar,
    render_llm_configuration_sidebar,
    render_context_selection_sidebar,
    render_retrieval_settings_sidebar
)
from pmchatbot.src.streamlit_integration.handlers.chat_handler import render_chat_interface
from pmchatbot.src.data.data_loader import list_part_descs

# Configure page
st.set_page_config(page_title="Process Mining Chatbot using GraphRAG", page_icon="🤖", layout="wide")

# Initialize session state
initialize_session_state()

# Get current application state
processing_started, dfg_ready = get_application_state()

def main():
    """Main application logic."""
    # Load data once
    df = load_csv_data()
    part_options = list_part_descs(df)
    
    # Sidebar logic
    with st.sidebar:
        # Show header if processing started or DFG is ready
        if dfg_ready or processing_started:
            render_sidebar_header()
        
        # LLM Configuration
        llm_config = render_llm_configuration_sidebar()
        
        # Update LLM configuration immediately when toggle changes
        update_llm_configuration(llm_config)
        
        # Part selection
        selected_part, process_clicked = render_part_selection_sidebar(
            part_options, 
            st.session_state.get("selected_part", None)
        )
        
        # Store selected part in session state for consistency
        st.session_state.selected_part = selected_part
        
        # Context selection (only show if DFG is ready)
        if dfg_ready:
            context = render_context_selection_sidebar()
            # Retrieval settings sliders (only show if context is selected)
            if context:
                retriever_top_k, reranker_top_k = render_retrieval_settings_sidebar(context)
            else:
                retriever_top_k, reranker_top_k = None, None
        else:
            context = None
            retriever_top_k, reranker_top_k = None, None

    # Main content logic based on application state
    if not selected_part and not processing_started:
        # Initial state: Show centered header and welcome message
        render_main_header()
        
    elif selected_part and not process_clicked and not processing_started:
        # Part selected but process not executed yet
        render_execute_instruction()
        
    elif selected_part and process_clicked:
        # Process discovery clicked - start processing
        handle_process_discovery(selected_part)
        
    elif dfg_ready:
        # Results state: Show visualizations and chat
        handle_results_display(context, retriever_top_k, reranker_top_k)
        
    else:
        # Fallback for undefined states
        render_fallback_message()


def handle_process_discovery(selected_part):
    """Handle the process discovery execution."""
    # Clear session state and show loading immediately
    clear_session_for_new_processing()
    st.rerun()  # Force rerun so loading screen appears immediately


def handle_loading_and_processing(selected_part):
    """Handle the loading overlay and data processing."""
    # Show loading overlay
    render_loading_overlay()
    
    # Perform the actual processing
    success = process_data_and_setup_graphrag(selected_part)
    
    if success:
        # Processing completed successfully
        st.rerun()  # Refresh to show results
    else:
        # Reset state on failure
        st.session_state.processing_started = False
        st.rerun()


def handle_results_display(context, retriever_top_k=None, reranker_top_k=None):
    """Handle the display of results including visualizations and chat."""
    # Display DFG images
    display_dfg_images()
    
    # Show chat interface if context is selected
    if context:
        render_chat_interface(context, retriever_top_k, reranker_top_k)


# Main execution flow
if processing_started and not dfg_ready:
    # Show loading and perform processing
    selected_part = st.session_state.get("selected_part")
    if selected_part:
        handle_loading_and_processing(selected_part)
    else:
        # Reset if no selected part
        st.session_state.processing_started = False
        st.rerun()
else:
    # Normal application flow
    main()