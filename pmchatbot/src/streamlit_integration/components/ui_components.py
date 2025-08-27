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
            <p style="margin: 0 0 0.5rem 0; color: #E6E6FA; font-size: 0.9rem;">
                <strong>Active LLM:</strong> {llm_display}
            </p>
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