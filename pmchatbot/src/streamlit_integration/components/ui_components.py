"""
UI Components module for the Process Mining Chatbot Streamlit app.
Contains all styling, display functions, and UI-related utilities.
"""

import streamlit as st
import base64
import importlib.resources
from ..assets import images


def get_base64_image(image_path):
    """Get base64 encoded image from file path."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def get_base64_image_from_resources(filename):
    """Get base64 encoded image from package resources."""
    try:
        with importlib.resources.open_binary(images, filename) as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None


def render_sidebar_header():
    """Render the sidebar header with background image and styling."""
    sidebar_bg_image = get_base64_image_from_resources("loading_background.jpg")
    
    st.markdown(
        f"""
        <div style="
            text-align: center; 
            margin-bottom: 1.5rem; 
            padding: 1rem; 
            background: linear-gradient(135deg, #7B61FF 0%, #9c88ff 100%);
            background-image: url('data:image/jpeg;base64,{sidebar_bg_image}');
            background-size: cover;
            background-position: center;
            border-radius: 10px; 
            color: white;
        ">
            <h4 style="margin: 0; font-size: 1.1rem; font-weight: 600;">
                Process Mining Chatbot
            </h4>
            <p style="margin: 0; font-size: 0.8rem; opacity: 0.9;">
                Powered by GraphRAG
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_main_header():
    """Render the main centered header for initial state."""
    st.markdown(
        """
        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 80vh;">
            <div style="text-align: center; margin-bottom: 3rem;">
                <h1 style="color: #7B61FF; margin-bottom: 0; font-size: 3rem;">Process Mining Chatbot</h1>
                <p style="color: #666666; font-size: 1.2rem; margin-top: 0.5rem;">
                    Powered by GraphRAG
                </p>
            </div>
            <div style="text-align: center;">
                <h3 style="color: #888888; font-weight: normal; margin: 0;">
                   Select a part from the sidebar to start process discovery
                </h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_execute_instruction():
    """Render instruction to execute process discovery."""
    st.markdown(
        """
        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 80vh;">
            <div style="text-align: center; margin-bottom: 3rem;">
                <h1 style="color: #7B61FF; margin-bottom: 0; font-size: 3rem;">Process Mining Chatbot</h1>
                <p style="color: #666666; font-size: 1.2rem; margin-top: 0.5rem;">
                    Powered by GraphRAG
                </p>
            </div>
            <div style="text-align: center;">
                <h3 style="color: #888888; font-weight: normal; margin: 0;">
                   Click "Execute Process Discovery" to analyze the selected part
                </h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_loading_overlay():
    """Render the loading overlay screen during processing."""
    bg_image = get_base64_image_from_resources("loading_background.jpg")
    
    st.markdown(
        f"""
        <div id="loading-overlay" style="
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.8);
            z-index: 9999;
            display: flex;
            justify-content: center;
            align-items: center;
            backdrop-filter: blur(3px);
        ">
            <div style="
                background-image: url('data:image/jpeg;base64,{bg_image}');
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-color: rgba(255,255,255,0.85);
                padding: 3rem;
                border-radius: 32px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
                max-width: 440px;
                width: 90%;
                min-height: 340px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
            ">
                <div style="
                    border: 4px solid #f3f3f3;
                    border-top: 4px solid #7B61FF;
                    border-radius: 50%;
                    width: 80px;
                    height: 80px;
                    animation: spin 1s linear infinite;
                    margin-bottom: 2rem;
                "></div>
                <div style="
                    width: 100%; 
                    display: flex; 
                    justify-content: center; 
                    align-items: center; 
                    margin-bottom: 1rem;
                ">
                    <span style="
                        color: #E6E6FA;
                        font-size: 2rem;
                        font-weight: 700;
                        user-select: none;
                        pointer-events: none;
                        text-align: center;
                        width: 100%;
                        ">
                        Processing Your Data
                    </span>
                </div>
                <div style="
                    width: 100%; 
                    display: flex; 
                    justify-content: center; 
                    align-items: center;
                ">
                    <span style="
                        color: #E6E6FA;
                        font-size: 1.15rem;
                        line-height: 1.5;
                        user-select: none;
                        pointer-events: none;
                        text-align: center;
                        width: 100%;
                        ">
                        Analyzing process data, generating visualizations, and setting up GraphRAG...
                    </span>
                </div>
            </div>
        </div>
        <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        #loading-overlay * {{
            user-select: none !important;
            pointer-events: none !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


def render_fallback_message():
    """Render fallback message for undefined states."""
    st.markdown(
        """
        <div style="display: flex; justify-content: center; align-items: center; height: 50vh;">
            <div style="text-align: center;">
                <h3 style="color: #666666; font-weight: normal;">
                   Select a part from the sidebar to start process discovery
                </h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_chat_getting_started():
    """Render the getting started message for chat functionality."""
    # Get current LLM configuration to show which LLM is active
    llm_config = st.session_state.get("llm_config", {})
    llm_type = llm_config.get("type", "ollama")
    
    # Get actual configuration from .env
    from pmchatbot.src.config.settings import Config
    config = Config()
    
    if llm_type == "openai":
        model_name = config.LLM_MODEL_NAME
        llm_display = f"OpenAI {model_name}"
    else:
        model_name = config.LLM_MODEL_NAME_OLLAMA
        llm_display = f"Ollama {model_name}"
    
    st.markdown(
        f"""
        <div style="background-color: #4c448e; padding: 1rem; margin: 1rem 0; border-radius: 8px;">
            <h4 style="margin: 0 0 0.5rem 0; color: #E6E6FA;">Getting Started</h4>
            <p style="margin: 0; color:  #E6E6FA;">
                Choose your <strong>Query Context</strong> from the sidebar, then ask your process mining question below!
                <br>• <strong>Activity-based:</strong> Focus on specific activities
                <br>• <strong>Process-based:</strong> Focus on process flows and paths  
                <br>• <strong>Variant-based:</strong> Focus on process variants
                <br>• <strong>All combined:</strong> Comprehensive analysis
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def display_dfg_images():
    """Display the DFG and Performance DFG images."""
    if "dfg_image" in st.session_state:
        image_container = st.container()
        with image_container:
            st.image(st.session_state.dfg_image, caption="Directly-Follows Graph (DFG)")
            st.image(st.session_state.performance_dfg_image, caption="Performance DFG (Mean Duration)")


def render_part_selection_sidebar(part_options, selected_part):
    """Render the part selection interface in sidebar."""
    st.header("Part Selection")
    
    # Calculate the correct index, handling cases where selected_part might not be in the list
    if selected_part is None or selected_part not in part_options:
        index = 0
    else:
        index = part_options.index(selected_part) + 1
    
    selected_part = st.selectbox(
        "Select a part",
        options=["Click to select..."] + part_options,
        index=index
    )
    if selected_part == "Click to select...":
        selected_part = None
    process_clicked = st.button("Execute Process Discovery")
    return selected_part, process_clicked


def _on_llm_toggle_change():
    """Callback function when LLM toggle changes."""
    # Immediately update LLM configuration when toggle changes
    from pmchatbot.src.streamlit_integration.core.state_manager import get_dynamic_llm
    
    llm_type = "openai" if st.session_state.llm_type_toggle else "ollama"
    
    # Get config and validate
    from pmchatbot.src.config.settings import Config
    config = Config()
    
    if llm_type == "openai":
        valid = bool(config.OPENAI_API_KEY and config.OPENAI_API_KEY.startswith("sk-"))
    else:
        valid = bool(config.LLM_MODEL_NAME_OLLAMA and config.OLLAMA_BASE_URL)
    
    # Update session state immediately
    st.session_state.llm_config = {
        "type": llm_type,
        "valid": valid
    }
    
    # Initialize LLM immediately if valid
    if valid:
        try:
            st.session_state.llm = get_dynamic_llm(st.session_state.llm_config)
        except Exception as e:
            st.session_state.llm = None


def render_llm_configuration_sidebar():
    """Render the LLM configuration interface in sidebar."""
    st.header("LLM Configuration")
    
    # Initialize toggle state if not present
    if "llm_type_toggle" not in st.session_state:
        st.session_state.llm_type_toggle = False  # Default to Ollama
    
    # Simple LLM Type toggle with callback
    llm_type = st.toggle("Use OpenAI API", 
                        help="Toggle off for Ollama (local), on for OpenAI API. Configure API keys and models in .env file",
                        key="llm_type_toggle",
                        on_change=_on_llm_toggle_change)
    
    # Get actual configuration from .env to show status
    from pmchatbot.src.config.settings import Config
    config = Config()
    
    if llm_type:  # OpenAI selected
        st.subheader("OpenAI Settings")
        
        # Check if OpenAI is properly configured in .env
        if config.OPENAI_API_KEY and config.OPENAI_API_KEY.startswith("sk-"):
            st.success(f"Model configured:\n{config.LLM_MODEL_NAME}")
            valid = True
        else:
            st.error("Configure OpenAI settings in .env file")
            valid = False
        
        llm_config = {
            "type": "openai",
            "valid": valid
        }
    else:  # Ollama selected
        st.subheader("Ollama Settings") 
        
        # Check if Ollama is properly configured in .env
        if config.LLM_MODEL_NAME_OLLAMA and config.OLLAMA_BASE_URL:
            st.success(f"Model configured:\n{config.LLM_MODEL_NAME_OLLAMA}")
            valid = True
        else:
            st.error("Configure Ollama settings in .env file")
            valid = False
        
        llm_config = {
            "type": "ollama", 
            "valid": valid
        }
    
    return llm_config


def render_context_selection_sidebar():
    """Render the query context selection in sidebar."""
    st.header("Query Context")
    context = st.radio(
        "Choose context for your question:",
        ["Activity-based", "Process-based", "Variant-based", "All combined"]
    )
    return context


def render_retrieval_settings_sidebar(context):
    """Render the retrieval settings sliders in sidebar."""
    st.header("Retrieval Settings")
    
    # Get maximum chunk counts based on selected context
    max_chunks = _get_max_chunks_for_context(context)
    
    # Handle edge case where max_chunks is 1 (Streamlit requires min < max)
    if max_chunks <= 1:
        st.info(f"📊 Only {max_chunks} chunk(s) available - using fixed values")
        # Use fixed values when only 1 chunk available
        retriever_top_k = 1
        reranker_top_k = 1
        st.write(f"**Retriever Top-K:** {retriever_top_k}")
        st.write(f"**Reranker Top-K:** {reranker_top_k}")
    else:
        # Retriever Top-K section
        st.markdown("**Retriever Top-K**")
        
        # Use columns to prevent value cutoff - give more space to slider
        col1, col2 = st.columns([9, 1])
        with col1:
            retriever_top_k = st.slider(
                "retriever_slider",
                min_value=1,
                max_value=max_chunks,
                value=min(st.session_state.get("retriever_top_k_slider", 15), max_chunks),
                help=f"Number of chunks to retrieve before reranking",
                label_visibility="collapsed"
            )
        with col2:
            st.write("")  # Keep column but leave it blank
        
        st.caption(f"Range: 1 to {max_chunks} chunks")
        
        # Add spacing between sliders
        st.markdown("<div style='margin: 15px 0;'></div>", unsafe_allow_html=True)
        
        # Reranker Top-K section
        max_reranker = min(retriever_top_k, max_chunks)
        
        # Check if reranker would also have min=max issue
        if max_reranker <= 1:
            st.markdown("**Reranker Top-K**")
            reranker_top_k = 1
            st.write(f"**Value:** {reranker_top_k} (fixed)")
        else:
            st.markdown("**Reranker Top-K**")
            
            # Use columns to prevent value cutoff
            col1, col2 = st.columns([9, 1])
            with col1:
                reranker_top_k = st.slider(
                    "reranker_slider",
                    min_value=1,
                    max_value=max_reranker,
                    value=min(st.session_state.get("reranker_top_k_slider", 5), max_reranker),
                    help=f"Number of final chunks after reranking",
                    label_visibility="collapsed"
                )
            with col2:
                st.write("")  # Keep column but leave it blank
            
            st.caption(f"Range: 1 to {max_reranker} chunks")
    
    # Store in session state (manual management for reactivity)
    st.session_state.retriever_top_k_slider = retriever_top_k
    st.session_state.reranker_top_k_slider = reranker_top_k
    
    return retriever_top_k, reranker_top_k


def _get_max_chunks_for_context(context):
    """Get maximum available chunks for the selected context."""
    try:
        if context == "Activity-based":
            if "retriever_activity" in st.session_state:
                return st.session_state.retriever_activity.get_total_chunk_count()
        elif context == "Process-based":
            if "retriever_process" in st.session_state:
                return st.session_state.retriever_process.get_total_chunk_count()
        elif context == "Variant-based":
            if "retriever_variant" in st.session_state:
                return st.session_state.retriever_variant.get_total_chunk_count()
        else:  # All combined
            # For combined context, use the maximum of all available chunks
            max_chunks = 0
            if "retriever_activity" in st.session_state:
                max_chunks = max(max_chunks, st.session_state.retriever_activity.get_total_chunk_count())
            if "retriever_process" in st.session_state:
                max_chunks = max(max_chunks, st.session_state.retriever_process.get_total_chunk_count())
            if "retriever_variant" in st.session_state:
                max_chunks = max(max_chunks, st.session_state.retriever_variant.get_total_chunk_count())
            return max_chunks if max_chunks > 0 else 50
    except Exception as e:
        # Fallback to a reasonable default
        pass
    
    return 50  # Default fallback