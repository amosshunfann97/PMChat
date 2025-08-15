"""
LLM Selection Manager for handling LLM type selection and configuration.

This module provides functionality for users to select between OpenAI API
and local LLM (Ollama), handle API key input, and configure LLM clients.
"""

import asyncio
import logging
from typing import Optional, Dict, Any

# Conditional import for Chainlit to avoid import errors in testing
try:
    import chainlit as cl
except ImportError:
    # Create a mock cl module for testing purposes
    class MockChainlit:
        class Action:
            def __init__(self, name: str, value: str, label: str):
                self.name = name
                self.value = value
                self.label = label
        
        class Message:
            def __init__(self, content: str):
                self.content = content
            
            async def send(self):
                pass
        
        class AskUserMessage:
            def __init__(self, content: str, actions: list = None, timeout: int = 60):
                self.content = content
                self.actions = actions or []
                self.timeout = timeout
            
            async def send(self):
                # Mock response for testing
                class MockResponse:
                    def __init__(self):
                        self.content = "mock-api-key"
                return MockResponse()
        
        class AskActionMessage:
            def __init__(self, content: str, actions: list, timeout: int = 60):
                self.content = content
                self.actions = actions
                self.timeout = timeout
            
            async def send(self):
                pass
        
        class TextInput:
            def __init__(self, id: str, label: str, placeholder: str = "", initial: str = ""):
                self.id = id
                self.label = label
                self.placeholder = placeholder
                self.initial = initial
    
    cl = MockChainlit()

from ..interfaces import LLMManagerInterface
from ..models import SessionState, LLMType, LLMConfiguration, ErrorContext
from ..utils.llm_client_factory import LLMClientManager


logger = logging.getLogger(__name__)


class LLMSelectionManager(LLMManagerInterface):
    """
    Manager for LLM selection and configuration.
    
    Handles the user interface for selecting between OpenAI API and local LLM,
    manages API key input and validation, and configures LLM clients based on
    user selection.
    """
    
    def __init__(self, session_state: SessionState):
        """
        Initialize LLM Selection Manager.
        
        Args:
            session_state: Current session state
        """
        super().__init__(session_state)
        self.logger = logger
        self.client_manager = LLMClientManager()
    
    async def initialize(self) -> bool:
        """
        Initialize the LLM Selection Manager.
        
        Returns:
            True if initialization was successful
        """
        try:
            self.logger.info("Initializing LLM Selection Manager")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM Selection Manager: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Clean up manager resources."""
        self.logger.info("Cleaning up LLM Selection Manager")
        self.client_manager.cleanup()
    
    async def show_llm_selector(self) -> None:
        """
        Display LLM selection interface to user.
        
        Auto-configures Ollama as the only available LLM option.
        """
        try:
            self.logger.info("Auto-configuring Ollama LLM")
            
            # Auto-select Ollama
            await self.handle_ollama_selection()
            
        except Exception as e:
            self.logger.error(f"Error in show_llm_selector: {e}")
            await cl.Message(
                content="❌ **Error**: Failed to configure Ollama. Please ensure Ollama is running locally."
            ).send()
    
    async def handle_llm_selection(self, llm_type: str) -> bool:
        """
        Handle LLM type selection.
        
        Args:
            llm_type: Selected LLM type (only "ollama" is supported)
            
        Returns:
            True if selection was successful
        """
        try:
            self.logger.info(f"Handling LLM selection: {llm_type}")
            
            if llm_type != "ollama":
                await cl.Message(
                    content="❌ Only Ollama (local LLM) is supported in this configuration."
                ).send()
                return False
            
            # Update session state with selected LLM type
            self.session_state.llm_type = LLMType(llm_type)
            
            # Configure Ollama with default settings
            config = LLMConfiguration(
                llm_type=LLMType.OLLAMA,
                base_url="http://localhost:11434",
                model_name="llama2"  # Default model
            )
            await self.configure_llm(config)
            
            return True
            
        except Exception as e:
            error_context = await self.handle_error(e, "handle_llm_selection")
            await cl.Message(content=error_context.format_user_message()).send()
            return False
    
    async def show_api_key_input(self) -> None:
        """
        Display API key input interface for OpenAI.
        
        Not supported in this configuration - only Ollama is available.
        """
        await cl.Message(
            content="❌ **API Key Input Not Supported**: This configuration only supports Ollama (local LLM). No API key required."
        ).send()
    
    async def validate_api_key(self, api_key: str) -> bool:
        """
        Validate OpenAI API key.
        
        Not supported in this configuration - only Ollama is available.
        
        Args:
            api_key: API key to validate
            
        Returns:
            False (not supported)
        """
        self.logger.info("API key validation not supported - only Ollama is available")
        return False
    
    async def configure_llm(self, config: LLMConfiguration) -> bool:
        """
        Configure LLM with provided settings.
        
        Args:
            config: LLM configuration
            
        Returns:
            True if configuration was successful
        """
        try:
            self.logger.info(f"Configuring LLM: {config.llm_type.value}")
            
            # Validate configuration
            if not config.is_valid():
                self.logger.error("Invalid LLM configuration")
                return False
            
            # Configure the LLM client using the client manager
            if await self.client_manager.configure_client(config):
                # Store configuration in session state
                self.session_state.llm_type = config.llm_type
                
                if config.llm_type == LLMType.OPENAI:
                    self.session_state.openai_api_key = config.api_key
                    self.logger.info("OpenAI client configured and stored in session")
                else:
                    self.logger.info("Ollama client configured successfully")
                
                return True
            else:
                self.logger.error("Failed to configure LLM client")
                return False
            
        except Exception as e:
            error_context = await self.handle_error(e, "configure_llm")
            self.logger.error(f"LLM configuration failed: {error_context.technical_details}")
            return False
    
    def get_current_llm_config(self) -> Optional[LLMConfiguration]:
        """
        Get current LLM configuration from session state.
        
        Returns:
            Current LLM configuration (always Ollama) or None if not configured
        """
        if not self.session_state.llm_type:
            return None
        
        # Only Ollama is supported
        return LLMConfiguration(
            llm_type=LLMType.OLLAMA,
            base_url="http://localhost:11434",
            model_name="llama2",
            temperature=0.1
        )
    
    def is_llm_configured(self) -> bool:
        """
        Check if LLM is properly configured.
        
        Returns:
            True if LLM is configured and ready to use
        """
        config = self.get_current_llm_config()
        return config is not None and config.is_valid()
    
    async def switch_llm(self) -> None:
        """
        Allow user to switch LLM during session.
        
        Not supported in this configuration - only Ollama is available.
        """
        try:
            self.logger.info("LLM switching not supported - only Ollama is available")
            
            await cl.Message(
                content="ℹ️ **LLM Switching Not Available**\n\nThis configuration only supports Ollama (local LLM). Ollama is already configured and ready to use."
            ).send()
            
        except Exception as e:
            error_context = await self.handle_error(e, "switch_llm")
            await cl.Message(content=error_context.format_user_message()).send()
    
    def get_llm_client_manager(self) -> LLMClientManager:
        """
        Get the LLM client manager.
        
        Returns:
            LLM client manager instance
        """
        return self.client_manager
    
    async def test_llm_connection(self) -> bool:
        """
        Test current LLM connection.
        
        Returns:
            True if LLM is connected and working
        """
        try:
            return await self.client_manager.test_current_client()
        except Exception as e:
            self.logger.error(f"LLM connection test failed: {e}")
            return False
    
    async def generate_test_response(self, prompt: str = "Hello, this is a test.") -> Optional[str]:
        """
        Generate a test response to verify LLM functionality.
        
        Args:
            prompt: Test prompt to use
            
        Returns:
            Generated response or None if failed
        """
        try:
            return await self.client_manager.generate_response(prompt)
        except Exception as e:
            self.logger.error(f"Test response generation failed: {e}")
            return None
    
    async def handle_openai_selection(self) -> bool:
        """Handle OpenAI selection - Not supported in this configuration."""
        try:
            self.logger.info("OpenAI selection attempted but not supported")
            await cl.Message(
                content="❌ **OpenAI Not Supported**: This configuration only supports Ollama (local LLM). Ollama is already configured for you."
            ).send()
            return False
        except Exception as e:
            self.logger.error(f"Error handling OpenAI selection: {e}")
            return False
    
    async def handle_ollama_selection(self) -> bool:
        """Handle Ollama selection."""
        try:
            self.logger.info("Ollama LLM selected")
            success = await self.handle_llm_selection("ollama")
            if success:
                await cl.Message(
                    content="✅ **Ollama Selected!** Using local LLM for processing."
                ).send()
            return success
        except Exception as e:
            self.logger.error(f"Error handling Ollama selection: {e}")
            await cl.Message(
                content=f"❌ **Error**: Failed to select Ollama. Please try again."
            ).send()
            return False