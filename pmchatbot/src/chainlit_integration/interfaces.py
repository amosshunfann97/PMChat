"""
Base interfaces and abstract classes for Chainlit integration components.

This module defines the contracts that all managers and handlers must implement
to ensure consistent behavior across the integration.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

# Conditional import for Chainlit to avoid import errors in testing
try:
    import chainlit as cl
except ImportError:
    # Create a mock cl module for testing purposes
    class MockChainlit:
        class Image:
            pass
        class File:
            pass
    cl = MockChainlit()

from .models import SessionState, ProcessingResult, LLMConfiguration, ErrorContext


class BaseManager(ABC):
    """
    Abstract base class for all manager components.
    
    Provides common functionality and defines the interface that all
    managers must implement.
    """
    
    def __init__(self, session_state: SessionState):
        """
        Initialize the base manager.
        
        Args:
            session_state: Current session state
        """
        self.session_state = session_state
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the manager.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up manager resources."""
        pass
    
    async def handle_error(self, error: Exception, context: str) -> ErrorContext:
        """
        Handle errors in a consistent way.
        
        Args:
            error: The exception that occurred
            context: Context information about where the error occurred
            
        Returns:
            Structured error context
        """
        return ErrorContext(
            error_type=type(error).__name__,
            component=self.__class__.__name__,
            operation=context,
            user_message=f"An error occurred in {context}",
            technical_details=str(error),
            recovery_suggestions=["Please try again", "Contact support if the issue persists"]
        )


class LLMManagerInterface(BaseManager):
    """Interface for LLM selection and configuration management."""
    
    @abstractmethod
    async def show_llm_selector(self) -> None:
        """Display LLM selection interface to user."""
        pass
    
    @abstractmethod
    async def handle_llm_selection(self, llm_type: str) -> bool:
        """
        Handle LLM type selection.
        
        Args:
            llm_type: Selected LLM type ("openai" or "ollama")
            
        Returns:
            True if selection was successful
        """
        pass
    
    @abstractmethod
    async def show_api_key_input(self) -> None:
        """Display API key input interface for OpenAI."""
        pass
    
    @abstractmethod
    async def validate_api_key(self, api_key: str) -> bool:
        """
        Validate OpenAI API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if API key is valid
        """
        pass
    
    @abstractmethod
    async def configure_llm(self, config: LLMConfiguration) -> bool:
        """
        Configure LLM with provided settings.
        
        Args:
            config: LLM configuration
            
        Returns:
            True if configuration was successful
        """
        pass


class PartSelectionManagerInterface(BaseManager):
    """Interface for part selection and data loading management."""
    
    @abstractmethod
    async def load_available_parts(self) -> List[str]:
        """
        Load available parts from CSV data.
        
        Returns:
            List of available part names
        """
        pass
    
    @abstractmethod
    async def show_searchable_dropdown(self) -> None:
        """Display searchable dropdown interface for part selection."""
        pass
    
    @abstractmethod
    async def filter_parts(self, search_term: str) -> List[str]:
        """
        Filter parts based on search term.
        
        Args:
            search_term: Search keyword
            
        Returns:
            Filtered list of parts
        """
        pass
    
    @abstractmethod
    async def handle_part_selection(self, selected_part: str) -> bool:
        """
        Handle part selection and validation.
        
        Args:
            selected_part: Selected part name
            
        Returns:
            True if selection was valid
        """
        pass


class ProcessMiningEngineInterface(BaseManager):
    """Interface for process mining operations and data processing."""
    
    @abstractmethod
    async def process_data(self, selected_part: str) -> ProcessingResult:
        """
        Process data for selected part using PM4py pipeline.
        
        Args:
            selected_part: Part to process
            
        Returns:
            Processing result with DFG data and metrics
        """
        pass
    
    @abstractmethod
    async def generate_automatic_visualizations(self, processing_result: ProcessingResult) -> Tuple[cl.Image, cl.Image]:
        """
        Generate automatic DFG visualizations.
        
        Args:
            processing_result: Result from data processing
            
        Returns:
            Tuple of (frequency_dfg_image, performance_dfg_image)
        """
        pass
    
    @abstractmethod
    async def setup_retrievers(self) -> Tuple[Any, Any, Any]:
        """
        Setup GraphRAG retrievers after data processing.
        
        Returns:
            Tuple of (activity_retriever, process_retriever, variant_retriever)
        """
        pass


class QueryContextManagerInterface(BaseManager):
    """Interface for query context selection and management."""
    
    @abstractmethod
    async def show_context_selector(self) -> None:
        """Display context selection interface."""
        pass
    
    @abstractmethod
    async def handle_context_selection(self, context: str) -> bool:
        """
        Handle context selection.
        
        Args:
            context: Selected context type
            
        Returns:
            True if selection was valid
        """
        pass
    
    @abstractmethod
    def reset_context_selection(self) -> None:
        """Reset context selection after each response."""
        pass
    
    @abstractmethod
    def get_available_contexts(self) -> List[str]:
        """
        Get list of available query contexts.
        
        Returns:
            List of available context types
        """
        pass


class ChatQueryHandlerInterface(BaseManager):
    """Interface for chat query processing and response generation."""
    
    @abstractmethod
    async def handle_query(self, query: str, context_mode: str) -> str:
        """
        Handle natural language query with specified context.
        
        Args:
            query: User's natural language query
            context_mode: Selected query context
            
        Returns:
            Generated response
        """
        pass
    
    @abstractmethod
    def check_termination_keywords(self, query: str) -> bool:
        """
        Check if query contains session termination keywords.
        
        Args:
            query: User query to check
            
        Returns:
            True if query contains termination keywords
        """
        pass
    
    @abstractmethod
    def format_clean_response(self, response: str) -> str:
        """
        Format response to hide technical details.
        
        Args:
            response: Raw response from LLM
            
        Returns:
            Cleaned response for user display
        """
        pass


class VisualizationManagerInterface(BaseManager):
    """Interface for visualization generation and display management."""
    
    @abstractmethod
    async def create_zoomable_image(self, image_path: str, title: str) -> cl.Image:
        """
        Create zoomable image component for Chainlit.
        
        Args:
            image_path: Path to image file
            title: Image title
            
        Returns:
            Chainlit image component
        """
        pass
    
    @abstractmethod
    async def export_data(self, export_type: str) -> cl.File:
        """
        Export process data to downloadable file.
        
        Args:
            export_type: Type of export (e.g., "csv")
            
        Returns:
            Chainlit file component for download
        """
        pass
    
    @abstractmethod
    def cleanup_temp_files(self) -> None:
        """Clean up temporary visualization files."""
        pass


class SessionManagerInterface(BaseManager):
    """Interface for session state management."""
    
    @abstractmethod
    async def initialize_session(self) -> SessionState:
        """
        Initialize a new session.
        
        Returns:
            Initialized session state
        """
        pass
    
    @abstractmethod
    async def save_session_state(self, session_state: SessionState) -> bool:
        """
        Save current session state.
        
        Args:
            session_state: Session state to save
            
        Returns:
            True if save was successful
        """
        pass
    
    @abstractmethod
    async def restore_session_state(self) -> Optional[SessionState]:
        """
        Restore session state from storage.
        
        Returns:
            Restored session state or None if not found
        """
        pass
    
    @abstractmethod
    async def clear_session(self) -> None:
        """Clear current session and reset to initial state."""
        pass


class ErrorHandlerInterface(ABC):
    """Interface for error handling and recovery."""
    
    @abstractmethod
    async def handle_error(self, error: Exception, context: str) -> str:
        """
        Handle error and return user-friendly message.
        
        Args:
            error: Exception that occurred
            context: Context where error occurred
            
        Returns:
            User-friendly error message
        """
        pass
    
    @abstractmethod
    def log_error(self, error: Exception, context: str) -> None:
        """
        Log error for debugging purposes.
        
        Args:
            error: Exception that occurred
            context: Context where error occurred
        """
        pass
    
    @abstractmethod
    async def suggest_recovery(self, error_type: str) -> List[str]:
        """
        Suggest recovery actions for specific error types.
        
        Args:
            error_type: Type of error that occurred
            
        Returns:
            List of recovery suggestions
        """
        pass