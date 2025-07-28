import time
from config.settings import PROCESS_MINING_CONTEXT, EXAMPLE_QUESTIONS, Config
from llm.llm_factory import get_llm, get_current_model_info

config = Config()

def show_help():
    """Display example questions to help users"""
    print("\nEXAMPLE QUESTIONS:")
    print("=" * 50)
    for i, question in enumerate(EXAMPLE_QUESTIONS, 1):
        print(f"   {i}. '{question}'")
    print("\n" + "=" * 50)

def graphrag_query_interface(rag_activity, rag_process, rag_variant, selected_part=None):
    """Interactive query interface for GraphRAG"""
    model_info = get_current_model_info()
    
    print("\n" + "="*80)
    print(f"PROCESS MINING CHATBOT (Using {model_info['type'].upper()}: {model_info['model_name']}) (Part: {selected_part})")
    print("="*80)
    print("I'm your process mining chatbot. You can query:")
    print("1. Activity-based context (individual activities and their relationships)")
    print("2. Process-based context (activity sequences and transitions)")
    print("3. Variant-based context (process variants and execution patterns)")
    print("4. All combined (merged results)")
    print("\nType 'quit' to exit, 'help' for more examples, or 'back' to select a new part")
    print("-" * 80)
    
    while True:
        mode = input("\nChoose context: (1) Activity, (2) Process, (3) Variant, (4) All: ").strip()
        
        if mode.lower() in ['quit', 'exit', 'q']:
            print("\nThanks for using Process Mining Expert! Keep optimizing those processes!")
            return False  # Exit the main loop

        if mode.lower() in ['back', 'restart']:
            return True  # Signal to go back to part selection

        if mode.lower() in ['help', 'examples', '?']:
            show_help()
            continue

        if mode not in ['1', '2', '3', '4']:
            print("Please enter 1, 2, 3, or 4")
            continue

        question = input("\nProcess Mining Question: ").strip()
        if not question:
            continue

        try:
            if mode == "1":
                _handle_single_context_query(rag_activity, "ACTIVITY-BASED", question)
            elif mode == "2":
                _handle_single_context_query(rag_process, "PROCESS-BASED", question)
            elif mode == "3":
                _handle_single_context_query(rag_variant, "VARIANT-BASED", question)
            else:
                _handle_combined_context_query(rag_activity, rag_process, rag_variant, question)
        except Exception as e:
            print(f"\nAnalysis Error: {e}")
            print("Try rephrasing your question or type 'help' for examples")

def _handle_single_context_query(rag, context_label, question):
    """Handle query for single context (activity, process, or variant)"""
    print(f"\nRETRIEVED CHUNKS ({context_label}):")
    print("-" * 50)
    
    search_result = rag.search(question)
    for i, item in enumerate(search_result.items, 1):
        content = _extract_content(item.content)
        metadata = item.metadata or {}
        
        print(f"Chunk {i}: {content}")
        print(f"   Metadata: {metadata}")
        print() 
    
    # Get enhanced answer
    result = rag.search(question)
    retrieved_info = "\n".join(_extract_content(item.content) for item in result.items)
    enhanced_prompt = f"""{PROCESS_MINING_CONTEXT}

Manufacturing Process Context: You are analyzing a manufacturing process with several activities. You have access to {context_label} chunks.

Retrieved Information: {retrieved_info}

User Question: {question}

Please provide a detailed process mining analysis based on the retrieved information:"""

    llm = get_llm()
    answer_start_time = time.time()
    enhanced_answer = llm.invoke(enhanced_prompt)
    answer_end_time = time.time()
    elapsed = answer_end_time - answer_start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    
    print(f"\n💡 ANSWER:")
    print("-" * 50)
    print(f"{enhanced_answer.content}")
    print(f"\nTime to generate this answer: {minutes} min {seconds} sec")

def _handle_combined_context_query(rag_activity, rag_process, rag_variant, question):
    """Handle query combining all three contexts"""
    # Show retrieved chunks from all contexts
    print("\nRETRIEVED CHUNKS (Activity-Based):")
    print("-" * 50)
    search_result_a = rag_activity.search(question, top_k=config.RERANKER_TOP_K)
    for i, item in enumerate(search_result_a.items, 1):
        content = _extract_content(item.content)
        print(f"Activity Chunk {i}: {content}")
        print(f"   Metadata: {item.metadata}")
        print()
    
    print("\nRETRIEVED CHUNKS (Process-Based):")
    print("-" * 50)
    search_result_p = rag_process.search(question, top_k=config.RERANKER_TOP_K)
    for i, item in enumerate(search_result_p.items, 1):
        content = _extract_content(item.content)
        print(f"Process Chunk {i}: {content}")
        print(f"   Metadata: {item.metadata}")
        print()  # Add blank line between chunks
    
    print("\nRETRIEVED CHUNKS (Variant-Based):")
    print("-" * 50)
    search_result_v = rag_variant.search(question, top_k=config.RERANKER_TOP_K)
    for i, item in enumerate(search_result_v.items, 1):
        content = _extract_content(item.content)
        print(f"Variant Chunk {i}: {content}")
        print(f"   Metadata: {item.metadata}")
        print()  # Add blank line between chunks
    
    # Get answers from all contexts
    answer_a = "\n".join(_extract_content(item.content) for item in rag_activity.search(question).items)
    answer_p = "\n".join(_extract_content(item.content) for item in rag_process.search(question).items)
    answer_v = "\n".join(_extract_content(item.content) for item in rag_variant.search(question).items)
    
    # Create enhanced prompt with all three contexts
    enhanced_prompt = f"""{PROCESS_MINING_CONTEXT}

Manufacturing Process Context: You are analyzing a manufacturing process with several activities. You have access to ACTIVITY-BASED, PROCESS-BASED, and VARIANT-BASED chunks.

Activity-Based Retrieved Information: {answer_a}

Process-Based Retrieved Information: {answer_p}

Variant-Based Retrieved Information: {answer_v}

User Question: {question}

Please provide a detailed process mining analysis based on the retrieved information from all three perspectives:"""

    llm = get_llm()
    answer_start_time = time.time()
    enhanced_answer = llm.invoke(enhanced_prompt)
    answer_end_time = time.time()
    elapsed = answer_end_time - answer_start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    
    print(f"\n💡 ANSWER (ALL COMBINED):")
    print("-" * 50)
    print(f"{enhanced_answer.content}")
    print(f"\nTime to generate this answer: {minutes} min {seconds} sec")

def _extract_content(content):
    """Extract actual content from record format"""
    content_str = str(content)
    if content_str.startswith('<Record text="') and content_str.endswith('">'):
        return content_str[14:-2]
    else:
        return content_str