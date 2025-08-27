import os
from pmchatbot.src.config.settings import Config
from pmchatbot.src.llm.ollama_llm import OllamaLLM
from neo4j_graphrag.llm import OpenAILLM
from pmchatbot.src.utils.logging_utils import log

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

def get_dynamic_llm(llm_config):
    """Factory function to get LLM based on runtime configuration (for Streamlit)"""
    llm_type = llm_config.get("type", "ollama").lower()
    
    if llm_type == "ollama":
        log(f"Using Ollama LLM: {config.LLM_MODEL_NAME_OLLAMA}", level="info")
        return OllamaLLM(
            model_name=config.LLM_MODEL_NAME_OLLAMA,
            model_params=config.LLM_MODEL_PARAMS,
            base_url=config.OLLAMA_BASE_URL
        )
    elif llm_type == "openai":
        if not config.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found in .env file")
        
        if not config.OPENAI_API_KEY.startswith("sk-"):
            raise ValueError("Invalid OpenAI API key format in .env file. API key should start with 'sk-'")
        
        # Ensure the API key is set in the environment for the OpenAI client
        import os
        os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
        
        log(f"Using OpenAI LLM: {config.LLM_MODEL_NAME}", level="info")
        return OpenAILLM(
            model_name=config.LLM_MODEL_NAME,
            model_params=config.LLM_MODEL_PARAMS
        )
    else:
        log(f"Unsupported LLM type: {llm_type}", level="error")
        raise ValueError(f"Unsupported LLM type: {llm_type}")

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