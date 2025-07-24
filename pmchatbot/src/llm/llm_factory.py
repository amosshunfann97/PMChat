from config.settings import Config
from llm.ollama_llm import OllamaLLM
from neo4j_graphrag.llm import OpenAILLM
from utils.logging_utils import log

config = Config()

def get_llm():
    """Factory function to get the appropriate LLM based on configuration"""
    if config.LLM_TYPE.lower() == "ollama":
        log(f"Using Ollama LLM: {config.LLM_MODEL_NAME_OLLAMA}", level="info")
        return OllamaLLM(
            model_name=config.LLM_MODEL_NAME_OLLAMA,
            model_params=config.LLM_MODEL_PARAMS,
            base_url=config.OLLAMA_BASE_URL
        )
    elif config.LLM_TYPE.lower() == "openai":
        log(f"Using OpenAI LLM: {config.LLM_MODEL_NAME}", level="info")
        return OpenAILLM(
            model_name=config.LLM_MODEL_NAME,
            model_params=config.LLM_MODEL_PARAMS
        )
    else:
        log(f"Unsupported LLM type: {config.LLM_TYPE}", level="error")
        raise ValueError(f"Unsupported LLM type: {config.LLM_TYPE}")

def get_current_model_info():
    """Get information about the currently configured model"""
    return {
        "type": config.LLM_TYPE,
        "model_name": config.CURRENT_MODEL_NAME,
        "temperature": config.LLM_MODEL_TEMPERATURE,
        "base_url": config.OLLAMA_BASE_URL if config.LLM_TYPE.lower() == "ollama" else "OpenAI API"
    }

def get_embedding_model_name():
    """Return the embedding model name from config"""
    name = getattr(config, "EMBEDDING_MODEL_PATH", "Not specified")
    log(f"Embedding model: {name}", level="debug")
    return name

def get_reranker_model_name():
    """Return the reranker model name from config"""
    name = getattr(config, "RERANKER_MODEL_PATH", "Not specified")
    log(f"Reranker model: {name}", level="debug")
    return name