from config.settings import Config
from llm.ollama_llm import OllamaLLM
from neo4j_graphrag.llm import OpenAILLM

config = Config()

def get_llm():
    """Factory function to get the appropriate LLM based on configuration"""
    if config.LLM_TYPE.lower() == "ollama":
        print(f"Using Ollama LLM: {config.LLM_MODEL_NAME_OLLAMA}")
        return OllamaLLM(
            model_name=config.LLM_MODEL_NAME_OLLAMA,
            model_params=config.LLM_MODEL_PARAMS,
            base_url=config.OLLAMA_BASE_URL
        )
    elif config.LLM_TYPE.lower() == "openai":
        print(f"Using OpenAI LLM: {config.LLM_MODEL_NAME}")
        return OpenAILLM(
            model_name=config.LLM_MODEL_NAME,
            model_params=config.LLM_MODEL_PARAMS
        )
    else:
        raise ValueError(f"Unsupported LLM type: {config.LLM_TYPE}")

def get_current_model_info():
    """Get information about the currently configured model"""
    return {
        "type": config.LLM_TYPE,
        "model_name": config.CURRENT_MODEL_NAME,
        "temperature": config.LLM_MODEL_TEMPERATURE,
        "base_url": config.OLLAMA_BASE_URL if config.LLM_TYPE.lower() == "ollama" else "OpenAI API"
    }