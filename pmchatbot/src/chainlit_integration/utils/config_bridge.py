"""
Configuration bridge utilities for Chainlit integration.

This module provides utilities to bridge the existing configuration system
with the new Chainlit integration components.
"""

import sys
import os
from typing import Optional, Dict, Any

# Add the src directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import Config
from ..models import LLMConfiguration, LLMType


class ConfigBridge:
    """
    Bridge class to integrate existing configuration with Chainlit components.
    
    Provides methods to access existing configuration and create new
    configuration objects for the integration components.
    """
    
    def __init__(self):
        """Initialize the configuration bridge."""
        self._config = Config()
    
    def get_existing_config(self) -> Config:
        """
        Get the existing configuration instance.
        
        Returns:
            Existing Config instance
        """
        return self._config
    
    def create_llm_configuration(self, llm_type: LLMType, api_key: Optional[str] = None) -> LLMConfiguration:
        """
        Create LLM configuration based on type and existing settings.
        
        Args:
            llm_type: Type of LLM to configure
            api_key: API key for OpenAI (if applicable)
            
        Returns:
            LLM configuration object
        """
        if llm_type == LLMType.OPENAI:
            return LLMConfiguration(
                llm_type=llm_type,
                api_key=api_key or self._config.OPENAI_API_KEY,
                model_name=self._config.LLM_MODEL_NAME,
                temperature=self._config.LLM_MODEL_TEMPERATURE
            )
        elif llm_type == LLMType.OLLAMA:
            return LLMConfiguration(
                llm_type=llm_type,
                model_name=self._config.LLM_MODEL_NAME_OLLAMA,
                base_url=self._config.OLLAMA_BASE_URL,
                temperature=self._config.LLM_MODEL_TEMPERATURE
            )
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
    
    def get_neo4j_config(self) -> Dict[str, str]:
        """
        Get Neo4j configuration.
        
        Returns:
            Dictionary with Neo4j connection parameters
        """
        return {
            'uri': self._config.NEO4J_URI,
            'user': self._config.NEO4J_USER,
            'password': self._config.NEO4J_PASSWORD
        }
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """
        Get embedding model configuration.
        
        Returns:
            Dictionary with embedding configuration
        """
        return {
            'model_path': self._config.EMBEDDING_MODEL_PATH,
            'device': 'auto'  # Will be determined at runtime
        }
    
    def get_reranker_config(self) -> Dict[str, Any]:
        """
        Get reranker configuration.
        
        Returns:
            Dictionary with reranker configuration
        """
        return {
            'use_reranker': self._config.USE_RERANKER,
            'model_path': self._config.RERANKER_MODEL_PATH,
            'top_k': self._config.RERANKER_TOP_K,
            'device': self._config.RERANKER_DEVICE
        }
    
    def get_retrieval_config(self) -> Dict[str, Any]:
        """
        Get retrieval configuration.
        
        Returns:
            Dictionary with retrieval configuration
        """
        return {
            'top_k': self._config.RETRIEVER_TOP_K,
            'hybrid_ranker': self._config.HYBRID_RANKER,
            'hybrid_alpha': self._config.HYBRID_ALPHA
        }
    
    def get_csv_file_path(self) -> Optional[str]:
        """
        Get CSV file path from configuration.
        
        Returns:
            CSV file path or None if not configured
        """
        return self._config.CSV_FILE_PATH
    
    def get_process_mining_context(self) -> str:
        """
        Get process mining context prompt.
        
        Returns:
            Process mining context string
        """
        from config.settings import PROCESS_MINING_CONTEXT
        return PROCESS_MINING_CONTEXT
    
    def get_example_questions(self) -> list:
        """
        Get example questions for the interface.
        
        Returns:
            List of example questions
        """
        from config.settings import EXAMPLE_QUESTIONS
        return EXAMPLE_QUESTIONS
    
    def update_llm_config(self, llm_config: LLMConfiguration) -> None:
        """
        Update the existing configuration with new LLM settings.
        
        Args:
            llm_config: New LLM configuration
        """
        if llm_config.llm_type == LLMType.OPENAI:
            self._config.LLM_TYPE = "openai"
            if llm_config.api_key:
                self._config.OPENAI_API_KEY = llm_config.api_key
            if llm_config.model_name:
                self._config.LLM_MODEL_NAME = llm_config.model_name
        elif llm_config.llm_type == LLMType.OLLAMA:
            self._config.LLM_TYPE = "ollama"
            if llm_config.model_name:
                self._config.LLM_MODEL_NAME_OLLAMA = llm_config.model_name
            if llm_config.base_url:
                self._config.OLLAMA_BASE_URL = llm_config.base_url
        
        if llm_config.temperature is not None:
            self._config.LLM_MODEL_TEMPERATURE = llm_config.temperature
            self._config.LLM_MODEL_PARAMS = {"temperature": llm_config.temperature}
    
    def validate_configuration(self) -> Dict[str, bool]:
        """
        Validate the current configuration.
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            'neo4j_configured': bool(self._config.NEO4J_URI and self._config.NEO4J_USER and self._config.NEO4J_PASSWORD),
            'embedding_model_configured': bool(self._config.EMBEDDING_MODEL_PATH),
            'llm_configured': bool(self._config.LLM_TYPE),
            'openai_key_available': bool(self._config.OPENAI_API_KEY) if self._config.LLM_TYPE.lower() == "openai" else True,
            'ollama_configured': bool(self._config.OLLAMA_BASE_URL) if self._config.LLM_TYPE.lower() == "ollama" else True
        }
        
        validation['all_valid'] = all(validation.values())
        return validation
    
    def get_log_level(self) -> str:
        """
        Get configured log level.
        
        Returns:
            Log level string
        """
        return self._config.LOG_LEVEL


# Global instance for convenience
_config_bridge = ConfigBridge()


def get_config_bridge() -> ConfigBridge:
    """
    Get the global configuration bridge instance.
    
    Returns:
        ConfigBridge instance
    """
    return _config_bridge


def get_existing_config() -> Config:
    """
    Get the existing configuration instance.
    
    Returns:
        Existing Config instance
    """
    return _config_bridge.get_existing_config()


def validate_environment() -> Dict[str, bool]:
    """
    Validate the environment configuration.
    
    Returns:
        Dictionary with validation results
    """
    return _config_bridge.validate_configuration()