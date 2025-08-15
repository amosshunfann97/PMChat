"""
LLM Client Factory for creating and configuring LLM clients.

This module provides functionality to create and configure different types
of LLM clients (OpenAI, Ollama) based on user selection and configuration.
"""

import logging
from typing import Optional, Any, Dict
from abc import ABC, abstractmethod

from ..models import LLMConfiguration, LLMType


logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.
    
    Defines the interface that all LLM clients must implement.
    """
    
    def __init__(self, config: LLMConfiguration):
        """
        Initialize the LLM client.
        
        Args:
            config: LLM configuration
        """
        self.config = config
        self.client = None
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the LLM client.
        
        Returns:
            True if initialization was successful
        """
        pass
    
    @abstractmethod
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate response using the LLM.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated response
        """
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """
        Test connection to the LLM service.
        
        Returns:
            True if connection is successful
        """
        pass
    
    def cleanup(self) -> None:
        """Clean up client resources."""
        self.client = None


class OpenAIClient(BaseLLMClient):
    """
    OpenAI API client implementation.
    
    Handles OpenAI API configuration and response generation.
    """
    
    def __init__(self, config: LLMConfiguration):
        """
        Initialize OpenAI client.
        
        Args:
            config: OpenAI configuration with API key
        """
        super().__init__(config)
        self.logger = logger
    
    async def initialize(self) -> bool:
        """
        Initialize the OpenAI client.
        
        Returns:
            True if initialization was successful
        """
        try:
            # Try to import OpenAI
            try:
                import openai
            except ImportError:
                self.logger.error("OpenAI package not installed. Please install with: pip install openai")
                return False
            
            # Create OpenAI client
            self.client = openai.OpenAI(
                api_key=self.config.api_key
            )
            
            self.logger.info("OpenAI client initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            return False
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate response using OpenAI API.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated response
        """
        try:
            if not self.client:
                raise RuntimeError("OpenAI client not initialized")
            
            # Set default parameters
            params = {
                "model": self.config.model_name or "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_tokens": kwargs.get("max_tokens", 1000),
            }
            
            # Make API call
            response = self.client.chat.completions.create(**params)
            
            # Extract response content
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content.strip()
            else:
                raise RuntimeError("No response generated from OpenAI API")
                
        except Exception as e:
            self.logger.error(f"OpenAI response generation failed: {e}")
            raise
    
    async def test_connection(self) -> bool:
        """
        Test connection to OpenAI API.
        
        Returns:
            True if connection is successful
        """
        try:
            if not self.client:
                return False
            
            # Make a minimal test call
            response = self.client.models.list()
            return True
            
        except Exception as e:
            self.logger.error(f"OpenAI connection test failed: {e}")
            return False


class OllamaClient(BaseLLMClient):
    """
    Ollama local LLM client implementation.
    
    Handles Ollama local LLM configuration and response generation.
    """
    
    def __init__(self, config: LLMConfiguration):
        """
        Initialize Ollama client.
        
        Args:
            config: Ollama configuration with base URL
        """
        super().__init__(config)
        self.logger = logger
    
    async def initialize(self) -> bool:
        """
        Initialize the Ollama client.
        
        Returns:
            True if initialization was successful
        """
        try:
            # Try to import requests for HTTP calls
            try:
                import requests
                self.requests = requests
            except ImportError:
                self.logger.error("Requests package not installed. Please install with: pip install requests")
                return False
            
            # Set up base configuration
            self.base_url = self.config.base_url or "http://localhost:11434"
            self.model_name = self.config.model_name or "llama2"
            
            self.logger.info(f"Ollama client initialized for {self.base_url}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama client: {e}")
            return False
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate response using Ollama API.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated response
        """
        try:
            if not hasattr(self, 'requests'):
                raise RuntimeError("Ollama client not initialized")
            
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "num_predict": kwargs.get("max_tokens", 1000),
                }
            }
            
            # Make API call to Ollama
            response = self.requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                raise RuntimeError(f"Ollama API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            self.logger.error(f"Ollama response generation failed: {e}")
            raise
    
    async def test_connection(self) -> bool:
        """
        Test connection to Ollama service.
        
        Returns:
            True if connection is successful
        """
        try:
            if not hasattr(self, 'requests'):
                return False
            
            # Test connection with a simple API call
            response = self.requests.get(f"{self.base_url}/api/tags", timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Ollama connection test failed: {e}")
            return False


class LLMClientFactory:
    """
    Factory class for creating LLM clients.
    
    Provides a centralized way to create and manage different types of LLM clients
    based on configuration.
    """
    
    @staticmethod
    def create_client(config: LLMConfiguration) -> Optional[BaseLLMClient]:
        """
        Create an LLM client based on configuration.
        
        Args:
            config: LLM configuration
            
        Returns:
            Configured LLM client or None if creation failed
        """
        try:
            if config.llm_type == LLMType.OPENAI:
                return OpenAIClient(config)
            elif config.llm_type == LLMType.OLLAMA:
                return OllamaClient(config)
            else:
                logger.error(f"Unsupported LLM type: {config.llm_type}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create LLM client: {e}")
            return None
    
    @staticmethod
    async def create_and_initialize_client(config: LLMConfiguration) -> Optional[BaseLLMClient]:
        """
        Create and initialize an LLM client.
        
        Args:
            config: LLM configuration
            
        Returns:
            Initialized LLM client or None if creation/initialization failed
        """
        try:
            client = LLMClientFactory.create_client(config)
            if client and await client.initialize():
                return client
            else:
                logger.error("Failed to initialize LLM client")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create and initialize LLM client: {e}")
            return None
    
    @staticmethod
    def get_supported_llm_types() -> list:
        """
        Get list of supported LLM types.
        
        Returns:
            List of supported LLM type strings
        """
        return [llm_type.value for llm_type in LLMType]


class LLMClientManager:
    """
    Manager for LLM client lifecycle and operations.
    
    Handles client creation, switching, and cleanup.
    """
    
    def __init__(self):
        """Initialize LLM client manager."""
        self.current_client: Optional[BaseLLMClient] = None
        self.current_config: Optional[LLMConfiguration] = None
        self.logger = logger
    
    async def configure_client(self, config: LLMConfiguration) -> bool:
        """
        Configure LLM client with new configuration.
        
        Args:
            config: New LLM configuration
            
        Returns:
            True if configuration was successful
        """
        try:
            # Clean up existing client
            if self.current_client:
                self.current_client.cleanup()
                self.current_client = None
            
            # Create and initialize new client
            self.current_client = await LLMClientFactory.create_and_initialize_client(config)
            
            if self.current_client:
                self.current_config = config
                self.logger.info(f"LLM client configured successfully: {config.llm_type.value}")
                return True
            else:
                self.logger.error("Failed to configure LLM client")
                return False
                
        except Exception as e:
            self.logger.error(f"LLM client configuration failed: {e}")
            return False
    
    async def generate_response(self, prompt: str, **kwargs) -> Optional[str]:
        """
        Generate response using current LLM client.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated response or None if generation failed
        """
        try:
            if not self.current_client:
                self.logger.error("No LLM client configured")
                return None
            
            return await self.current_client.generate_response(prompt, **kwargs)
            
        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            return None
    
    async def test_current_client(self) -> bool:
        """
        Test current LLM client connection.
        
        Returns:
            True if client is working properly
        """
        try:
            if not self.current_client:
                return False
            
            return await self.current_client.test_connection()
            
        except Exception as e:
            self.logger.error(f"Client test failed: {e}")
            return False
    
    def get_current_config(self) -> Optional[LLMConfiguration]:
        """
        Get current LLM configuration.
        
        Returns:
            Current configuration or None if not configured
        """
        return self.current_config
    
    def is_configured(self) -> bool:
        """
        Check if LLM client is configured and ready.
        
        Returns:
            True if client is configured
        """
        return self.current_client is not None and self.current_config is not None
    
    def cleanup(self) -> None:
        """Clean up LLM client manager."""
        if self.current_client:
            self.current_client.cleanup()
            self.current_client = None
        self.current_config = None
        self.logger.info("LLM client manager cleaned up")