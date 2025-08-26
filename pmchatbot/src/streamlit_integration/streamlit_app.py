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
from pmchatbot.src.llm.llm_factory import get_llm
from pmchatbot.src.config.settings import PROCESS_MINING_CONTEXT
import base64
import importlib.resources
from pmchatbot.assets import images

st.set_page_config(page_title="Process Mining Chatbot using GraphRAG", page_icon="🤖", layout="wide")

# Cache data loading function
@st.cache_data
def load_csv_data():
    df = original_load_csv_data()
    return df

# Check if processing is in progress or completed
processing_started = st.session_state.get("processing_started", False)
dfg_ready = st.session_state.get("dfg_image_ready", False)

# Function to get base64 encoded image
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def get_base64_image_from_resources(filename):
    """Get base64 encoded image from package resources"""
    try:
        with importlib.resources.open_binary(images, filename) as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None

# Sidebar: Part selection and process button
with st.sidebar:
    # Sidebar header always at the top
    if dfg_ready or processing_started:
        # Get base64 for sidebar background image
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
    st.header("Part Selection")
    df = load_csv_data()
    part_options = list_part_descs(df)
    selected_part = st.selectbox(
        "Select a part", 
        options=["Select a part..."] + part_options,
        index=0
    )
    if selected_part == "Select a part...":
        selected_part = None
    process_clicked = st.button("Execute Process Discovery")

# Main content logic
if not selected_part and not processing_started:
    # Initial state: Show centered header and welcome message
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

elif selected_part and not process_clicked and not processing_started:
    # Part selected but process not executed yet: Show header with instruction to execute
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

# Always show loading screen when process discovery is clicked
if selected_part and process_clicked:
    # Clear everything related to previous session immediately
    st.session_state.processing_started = True
    st.session_state.dfg_image_ready = False
    st.session_state.chat_history = []
    
    # Also clear any retriever references to ensure clean state
    if "retriever_activity" in st.session_state:
        del st.session_state.retriever_activity
    if "retriever_process" in st.session_state:
        del st.session_state.retriever_process
    if "retriever_variant" in st.session_state:
        del st.session_state.retriever_variant
    
    st.rerun()  # Force rerun so loading screen appears immediately

# Function to get base64 encoded image
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def get_base64_image_from_resources(filename):
    """Get base64 encoded image from package resources"""
    try:
        with importlib.resources.open_binary(images, filename) as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None

# Show overlay loading screen when processing
if processing_started and not dfg_ready:
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
    
    # Perform the actual processing (this happens while overlay is shown)
    filtered_df = filter_by_part_desc(df, selected_part)
    
    # Processing logic (this will run while showing the loading overlay)
    event_log = prepare_pm4py_log(filtered_df)
    dfg, start_activities, end_activities, performance_dfgs, rework_cases = discover_process_model(event_log)

    # Activity-based chunking
    activity_case_map = filtered_df.groupby('activity')['case_id'].apply(list).to_dict()
    activity_chunks = generate_activity_based_chunks(dfg, start_activities, end_activities, activity_case_map, rework_cases)

    # Process-based chunking
    frequent_paths, path_performance = extract_process_paths(dfg, performance_dfgs, min_frequency=1)
    process_chunks = generate_process_based_chunks(frequent_paths, path_performance)

    # Variant-based chunking
    variant_stats = extract_case_variants(event_log, min_cases_per_variant=1)
    variant_chunks = generate_variant_based_chunks(dfg, start_activities, end_activities, variant_stats)

    # Store chunks in Neo4j
    driver = connect_neo4j()
    local_embedder = get_local_embedder(device="cpu")
    if driver and local_embedder:
        store_chunks_in_neo4j(
            driver, dfg, start_activities, end_activities,
            activity_chunks, process_chunks, variant_chunks,
            frequent_paths, variant_stats, local_embedder
        )

        # Setup retrievers for GraphRAG
        st.session_state.retriever_activity = setup_enhanced_retriever(driver, "ActivityChunk", local_embedder)
        st.session_state.retriever_process = setup_enhanced_retriever(driver, "ProcessChunk", local_embedder)
        st.session_state.retriever_variant = setup_enhanced_retriever(driver, "VariantChunk", local_embedder)
        st.session_state.llm = get_llm()
    else:
        st.error("Neo4j connection or local embedder failed.")

    # Visualize DFG and Performance DFG (only generate once)
    visualize_dfg(dfg, start_activities, end_activities, output_path="dfg_pm4py.png")
    visualize_performance_dfg(performance_dfgs['mean'], start_activities, end_activities, output_path="performance_dfg_pm4py.png")
    
    # Store images in session state instead of just setting a flag
    with open("dfg_pm4py.png", "rb") as f:
        st.session_state.dfg_image = f.read()
    with open("performance_dfg_pm4py.png", "rb") as f:
        st.session_state.performance_dfg_image = f.read()
    
    st.session_state.dfg_image_ready = True
    st.rerun()  # Refresh to show the results

elif dfg_ready:
    # Results state: Show visualizations and chat

    # Display images
    if "dfg_image" in st.session_state:
        image_container = st.container()
        with image_container:
            st.image(st.session_state.dfg_image, caption="Directly-Follows Graph (DFG)")
            st.image(st.session_state.performance_dfg_image, caption="Performance DFG (Mean Duration)")

    # Chat functionality
    if "retriever_activity" in st.session_state:
        # Sidebar: Query context selection
        with st.sidebar:
            st.header("Query Context")
            context = st.radio(
                "Choose context for your question:",
                ["Activity-based", "Process-based", "Variant-based", "All combined"]
            )

        # Chat history for continueable conversation
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history first (before input)
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    with st.chat_message("user"):
                        st.write(msg["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(msg["content"])

        # Show context reminder if no chat history exists
        if not st.session_state.chat_history:
            st.markdown(
                """
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

        user_question = st.chat_input("Ask a process mining question:")

        if user_question:
            # Add user message immediately
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_question)

            # Retrieve relevant chunks based on context
            with st.spinner("Generating response..."):
                if context == "Activity-based":
                    result = st.session_state.retriever_activity.search(user_question)
                    retrieved_info = "\n".join(str(item.content) for item in result.items)
                    context_label = "ACTIVITY-BASED"
                elif context == "Process-based":
                    result = st.session_state.retriever_process.search(user_question)
                    retrieved_info = "\n".join(str(item.content) for item in result.items)
                    context_label = "PROCESS-BASED"
                elif context == "Variant-based":
                    result = st.session_state.retriever_variant.search(user_question)
                    retrieved_info = "\n".join(str(item.content) for item in result.items)
                    context_label = "VARIANT-BASED"
                else:  # All combined
                    result_a = st.session_state.retriever_activity.search(user_question)
                    result_p = st.session_state.retriever_process.search(user_question)
                    result_v = st.session_state.retriever_variant.search(user_question)
                    retrieved_info = (
                        "Activity Chunks:\n" + "\n".join(str(item.content) for item in result_a.items) +
                        "\nProcess Chunks:\n" + "\n".join(str(item.content) for item in result_p.items) +
                        "\nVariant Chunks:\n" + "\n".join(str(item.content) for item in result_v.items)
                    )
                    context_label = "ACTIVITY-BASED, PROCESS-BASED, and VARIANT-BASED"

                # Build prompt with system instructions
                prompt = f"""{PROCESS_MINING_CONTEXT}

Manufacturing Process Context: You are analyzing a manufacturing process with several activities. You have access to {context_label} chunks.

Retrieved Information: {retrieved_info}

User Question: {user_question}

Please provide a detailed process mining analysis based on the retrieved information."""
                answer = st.session_state.llm.invoke(prompt).content

            # Add bot response and display it
            st.session_state.chat_history.append({"role": "bot", "content": answer})
            with st.chat_message("assistant"):
                st.write(answer)

            # Force rerun to update the chat display
            st.rerun()

else:
    # Fallback: Show part selection message if somehow in an undefined state
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