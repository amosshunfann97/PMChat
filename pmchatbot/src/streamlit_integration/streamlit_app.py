import streamlit as st
from pmchatbot.src.data.data_loader import load_csv_data, list_part_descs, filter_by_part_desc
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

st.set_page_config(page_title="Process Mining GraphRAG Frontend", layout="wide")
st.title("Process Mining GraphRAG Frontend")

# Sidebar: Part selection and process button
with st.sidebar:
    st.header("Select Part")
    df = load_csv_data()
    part_options = list_part_descs(df)
    selected_part = st.selectbox("Select a part", part_options)
    process_clicked = st.button("Process Selected Part")

# Main area: DFG images and chat
if selected_part:
    filtered_df = filter_by_part_desc(df, selected_part)

    if process_clicked:
        with st.spinner("Processing and loading chunks..."):
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

            # Visualize DFG and Performance DFG
            visualize_dfg(dfg, start_activities, end_activities, output_path="dfg_pm4py.png")
            visualize_performance_dfg(performance_dfgs['mean'], start_activities, end_activities, output_path="performance_dfg_pm4py.png")
            st.session_state.dfg_image_ready = True

        st.success("Process mining, chunking, Neo4j storage, and retriever setup complete! Visualizing DFGs...")

# Always show images if ready
if st.session_state.get("dfg_image_ready"):
    st.image("dfg_pm4py.png", caption="Directly-Follows Graph (DFG)")
    st.image("performance_dfg_pm4py.png", caption="Performance DFG (Mean Duration)")

# --- Chat window below visualizations ---
if "retriever_activity" in st.session_state:
    st.header("Process Mining Chatbot")

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

    user_question = st.chat_input("Ask a process mining question:")

    if user_question:
        # Retrieve relevant chunks based on context
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

        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "bot", "content": answer})

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Bot:** {msg['content']}")