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
    # Format limited chat history to provide conversational memory
    def _format_chat_history(limit: int = 10) -> str:
        history = get_chat_history() or []
        if not history:
            return ""
        selected = history[-limit:]
        lines = []
        for msg in selected:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    history_block = _format_chat_history()
    history_section = f"\nConversation History (recent):\n{history_block}\n" if history_block else ""

    # Build prompt with system instructions and conversation memory
    prompt = f"""{PROCESS_MINING_CONTEXT}

Manufacturing Process Context: You are analyzing a manufacturing process with several activities. You have access to {context_label} chunks.
{history_section}
Conversation Use Policy:
- Use conversation history to resolve follow-ups, pronouns ("this", "that"), and references to earlier labels, assumptions, or decisions made in the chat.
- If the user asks what was said earlier, answer from conversation history even if it is not present in the retrieved chunks.
- When conversation history conflicts with retrieved data, state both succinctly and ask a brief clarification.

Retrieved Information: {retrieved_info}

User Question: {user_question}

Please answer concisely, using retrieved data for facts and conversation history for references and previously discussed items."""
    
    answer = st.session_state.llm.invoke(prompt).content

    # Normalize formatting: ensure the first non-empty line is not a bullet
    try:
        lines = answer.splitlines()
        # Find first non-empty line
        idx = next((i for i, l in enumerate(lines) if l.strip() != ""), None)
        if idx is not None:
            first = lines[idx]
            stripped = first.lstrip()
            # Patterns like '-', '*', '•', or numbered lists '1.'/'1)'
            bullet_prefixes = ("- ", "* ", "• ")
            number_list = False
            # Simple check for numbered list prefix
            if len(stripped) > 2 and stripped[0].isdigit():
                # e.g., '1. text' or '1) text'
                if stripped[1:3] in (". ", ") ") or (stripped[1] in ".)" and len(stripped) > 2 and stripped[2] == " "):
                    number_list = True
            if stripped.startswith(bullet_prefixes) or number_list:
                # Remove only the first bullet/number marker
                if stripped.startswith(bullet_prefixes):
                    for p in bullet_prefixes:
                        if stripped.startswith(p):
                           stripped = stripped[len(p):]
                           break
                elif number_list:
                    # Remove up to the first space after the marker
                    # Find first space after the marker characters
                    # e.g., '1. summary' -> 'summary'
                    after = stripped
                    # Remove leading digits
                    j = 0
                    while j < len(after) and after[j].isdigit():
                        j += 1
                    # Skip '.' or ')' and a following space if present
                    if j < len(after) and after[j] in ".)":
                        j += 1
                    if j < len(after) and after[j] == " ":
                        j += 1
                    stripped = after[j:]
                # Rebuild lines with normalized first line
                lines[idx] = ("" if first.startswith(" ") else "") + stripped
                answer = "\n".join(lines)
    except Exception:
        # If anything goes wrong, fall back to original answer
        pass

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
