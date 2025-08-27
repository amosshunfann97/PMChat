"""
Chat functionality module for the Process Mining Chatbot Streamlit app.
Handles chat interface, message processing, and retrieval logic.
"""

import streamlit as st
from pmchatbot.src.config.settings import PROCESS_MINING_CONTEXT
from pmchatbot.src.streamlit_integration.core.state_manager import add_chat_message, get_chat_history, has_retrievers_ready, get_llm_configuration


def display_chat_history():
    """Display the chat history in the interface."""
    chat_history = get_chat_history()
    
    chat_container = st.container()
    with chat_container:
        for msg in chat_history:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(msg["content"])


def get_retrieved_information(context, user_question, retriever_top_k=None, reranker_top_k=None):
    """
    Retrieve relevant information based on the selected context.
    
    Args:
        context: The selected context type
        user_question: The user's question
        retriever_top_k: Number of chunks to retrieve (None for config default)
        reranker_top_k: Number of chunks after reranking (None for config default)
        
    Returns:
        tuple: (retrieved_info, context_label)
    """
    if context == "Activity-based":
        result = st.session_state.retriever_activity.search(user_question, top_k=retriever_top_k, reranker_top_k=reranker_top_k)
        retrieved_info = "\n".join(str(item.content) for item in result.items)
        context_label = "ACTIVITY-BASED"
    elif context == "Process-based":
        result = st.session_state.retriever_process.search(user_question, top_k=retriever_top_k, reranker_top_k=reranker_top_k)
        retrieved_info = "\n".join(str(item.content) for item in result.items)
        context_label = "PROCESS-BASED"
    elif context == "Variant-based":
        result = st.session_state.retriever_variant.search(user_question, top_k=retriever_top_k, reranker_top_k=reranker_top_k)
        retrieved_info = "\n".join(str(item.content) for item in result.items)
        context_label = "VARIANT-BASED"
    else:  # All combined
        result_a = st.session_state.retriever_activity.search(user_question, top_k=retriever_top_k, reranker_top_k=reranker_top_k)
        result_p = st.session_state.retriever_process.search(user_question, top_k=retriever_top_k, reranker_top_k=reranker_top_k)
        result_v = st.session_state.retriever_variant.search(user_question, top_k=retriever_top_k, reranker_top_k=reranker_top_k)
        retrieved_info = (
            "Activity Chunks:\n" + "\n".join(str(item.content) for item in result_a.items) +
            "\nProcess Chunks:\n" + "\n".join(str(item.content) for item in result_p.items) +
            "\nVariant Chunks:\n" + "\n".join(str(item.content) for item in result_v.items)
        )
        context_label = "ACTIVITY-BASED, PROCESS-BASED, and VARIANT-BASED"
    
    return retrieved_info, context_label


def generate_llm_response(user_question, retrieved_info, context_label):
    """
    Generate response using the LLM with the retrieved information.
    
    Args:
        user_question: The user's question
        retrieved_info: Retrieved information from the database
        context_label: Label describing the context type
        
    Returns:
        str: Generated response
    """
    # Build prompt with system instructions
    prompt = f"""{PROCESS_MINING_CONTEXT}

Manufacturing Process Context: You are analyzing a manufacturing process with several activities. You have access to {context_label} chunks.

Retrieved Information: {retrieved_info}

User Question: {user_question}

Please provide a detailed process mining analysis based on the retrieved information."""
    
    answer = st.session_state.llm.invoke(prompt).content
    return answer


def handle_chat_interaction(context, retriever_top_k=None, reranker_top_k=None):
    """
    Handle the complete chat interaction flow.
    
    Args:
        context: The selected context for the query
        retriever_top_k: Number of chunks to retrieve (None for config default)
        reranker_top_k: Number of chunks after reranking (None for config default)
        
    Returns:
        bool: True if a new message was processed, False otherwise
    """
    if not has_retrievers_ready():
        st.error("Retrievers are not ready. Please ensure the data processing is complete.")
        return False
    
    # Check LLM configuration validity
    llm_config = get_llm_configuration()
    if not llm_config.get("valid", False):
        if llm_config.get("type") == "openai":
            st.error("Please configure OpenAI API key in the .env file to use the chat functionality.")
        else:
            st.error("Please configure Ollama settings in the .env file to use the chat functionality.")
        return False
    
    # Check if LLM is available in session state
    if "llm" not in st.session_state or st.session_state.llm is None:
        st.error("LLM is not initialized. Please check your .env configuration.")
        return False
    
    user_question = st.chat_input("Ask a process mining question:")
    
    if user_question:
        # Add user message immediately
        add_chat_message("user", user_question)
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_question)

        # Retrieve relevant chunks and generate response
        with st.spinner("Generating response..."):
            retrieved_info, context_label = get_retrieved_information(context, user_question, retriever_top_k, reranker_top_k)
            answer = generate_llm_response(user_question, retrieved_info, context_label)

        # Add bot response and display it
        add_chat_message("bot", answer)
        with st.chat_message("assistant"):
            st.write(answer)

        return True
    
    return False


def render_chat_interface(context, retriever_top_k=None, reranker_top_k=None):
    """
    Render the complete chat interface including history and input.
    
    Args:
        context: The selected context for queries
        retriever_top_k: Number of chunks to retrieve (None for config default)
        reranker_top_k: Number of chunks after reranking (None for config default)
    """
    if not has_retrievers_ready():
        st.error("Chat functionality is not available. Please complete the data processing first.")
        return
    
    # Check LLM configuration
    llm_config = get_llm_configuration()
    if not llm_config.get("valid", False):
        if llm_config.get("type") == "openai":
            st.warning("⚠️ Please configure OpenAI API key in the .env file to enable chat functionality.")
        else:
            st.warning("⚠️ Please configure Ollama settings in the .env file to enable chat functionality.")
        return
    
    # Display chat history
    display_chat_history()
    
    # Show getting started message if no chat history
    chat_history = get_chat_history()
    if not chat_history:
        from pmchatbot.src.streamlit_integration.components.ui_components import render_chat_getting_started
        render_chat_getting_started()
    
    # Handle chat interaction
    new_message_processed = handle_chat_interaction(context, retriever_top_k, reranker_top_k)
    
    # Force rerun if new message was processed to update display
    if new_message_processed:
        st.rerun()