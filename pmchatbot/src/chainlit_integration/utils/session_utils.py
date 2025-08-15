"""
Session management utilities for Chainlit integration.

This module provides utilities for managing session state, including
initialization, persistence, and cleanup operations.
"""

from typing import Optional, Any, Dict
import json
import logging

# Conditional import for Chainlit
try:
    import chainlit as cl
except ImportError:
    # Create a mock cl module for testing purposes
    class MockChainlit:
        class user_session:
            _data = {}
            @classmethod
            def get(cls, key, default=None):
                return cls._data.get(key, default)
            @classmethod
            def set(cls, key, value):
                cls._data[key] = value
            @classmethod
            def keys(cls):
                return cls._data.keys()
    cl = MockChainlit()

from ..models import SessionState, LLMType, QueryContext


logger = logging.getLogger(__name__)


class SessionManager:
    """
    Utility class for managing Chainlit session state.
    
    Provides methods to initialize, save, restore, and manage session state
    across user interactions.
    """
    
    @staticmethod
    def get_session_state() -> SessionState:
        """
        Get current session state from Chainlit session.
        
        Returns:
            Current session state or a new one if not found
        """
        session_state = cl.user_session.get("session_state")
        if session_state is None:
            session_state = SessionState()
            cl.user_session.set("session_state", session_state)
        return session_state
    
    @staticmethod
    def save_session_state(session_state: SessionState) -> None:
        """
        Save session state to Chainlit session.
        
        Args:
            session_state: Session state to save
        """
        cl.user_session.set("session_state", session_state)
    
    @staticmethod
    def clear_session() -> None:
        """Clear all session data."""
        # Clear Chainlit session
        for key in list(cl.user_session.keys()):
            cl.user_session.set(key, None)
        
        # Initialize new session state
        new_session = SessionState()
        cl.user_session.set("session_state", new_session)
    
    @staticmethod
    def get_session_value(key: str, default: Any = None) -> Any:
        """
        Get a value from the session.
        
        Args:
            key: Session key
            default: Default value if key not found
            
        Returns:
            Session value or default
        """
        return cl.user_session.get(key, default)
    
    @staticmethod
    def set_session_value(key: str, value: Any) -> None:
        """
        Set a value in the session.
        
        Args:
            key: Session key
            value: Value to set
        """
        cl.user_session.set(key, value)
    
    @staticmethod
    def update_session_state(**kwargs) -> SessionState:
        """
        Update session state with provided values.
        
        Args:
            **kwargs: Fields to update in session state
            
        Returns:
            Updated session state
        """
        session_state = SessionManager.get_session_state()
        
        for key, value in kwargs.items():
            if hasattr(session_state, key):
                setattr(session_state, key, value)
            else:
                logger.warning(f"Unknown session state field: {key}")
        
        SessionManager.save_session_state(session_state)
        return session_state
    
    @staticmethod
    def is_session_ready_for_queries() -> bool:
        """
        Check if session is ready to handle queries.
        
        Returns:
            True if session is ready for queries
        """
        session_state = SessionManager.get_session_state()
        return session_state.is_ready_for_queries()
    
    @staticmethod
    def reset_for_new_part() -> None:
        """Reset session state for new part selection."""
        session_state = SessionManager.get_session_state()
        session_state.reset_for_new_part()
        SessionManager.save_session_state(session_state)
    
    @staticmethod
    def reset_query_context() -> None:
        """Reset query context after each response."""
        session_state = SessionManager.get_session_state()
        session_state.reset_query_context()
        SessionManager.save_session_state(session_state)


class SessionStateEncoder(json.JSONEncoder):
    """JSON encoder for session state objects."""
    
    def default(self, obj):
        if isinstance(obj, (LLMType, QueryContext)):
            return obj.value
        elif isinstance(obj, SessionState):
            return {
                'llm_type': obj.llm_type.value if obj.llm_type else None,
                'openai_api_key': obj.openai_api_key,
                'selected_part': obj.selected_part,
                'available_parts': obj.available_parts,
                'processing_complete': obj.processing_complete,
                'data_loaded': obj.data_loaded,
                'visualizations_displayed': obj.visualizations_displayed,
                'awaiting_context_selection': obj.awaiting_context_selection,
                'current_context_mode': obj.current_context_mode.value if obj.current_context_mode else None,
                'session_active': obj.session_active
            }
        return super().default(obj)


def serialize_session_state(session_state: SessionState) -> str:
    """
    Serialize session state to JSON string.
    
    Args:
        session_state: Session state to serialize
        
    Returns:
        JSON string representation
    """
    return json.dumps(session_state, cls=SessionStateEncoder)


def deserialize_session_state(json_str: str) -> SessionState:
    """
    Deserialize session state from JSON string.
    
    Args:
        json_str: JSON string to deserialize
        
    Returns:
        Deserialized session state
    """
    data = json.loads(json_str)
    
    session_state = SessionState()
    session_state.llm_type = LLMType(data['llm_type']) if data['llm_type'] else None
    session_state.openai_api_key = data['openai_api_key']
    session_state.selected_part = data['selected_part']
    session_state.available_parts = data['available_parts']
    session_state.processing_complete = data['processing_complete']
    session_state.data_loaded = data['data_loaded']
    session_state.visualizations_displayed = data['visualizations_displayed']
    session_state.awaiting_context_selection = data['awaiting_context_selection']
    session_state.current_context_mode = QueryContext(data['current_context_mode']) if data['current_context_mode'] else None
    session_state.session_active = data['session_active']
    
    return session_state


def log_session_state(session_state: Optional[SessionState] = None) -> None:
    """
    Log current session state for debugging.
    
    Args:
        session_state: Session state to log, or None to get current state
    """
    if session_state is None:
        session_state = SessionManager.get_session_state()
    
    logger.debug(f"Session State: {serialize_session_state(session_state)}")


def validate_session_state(session_state: SessionState) -> Dict[str, bool]:
    """
    Validate session state and return validation results.
    
    Args:
        session_state: Session state to validate
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        'has_llm_type': session_state.llm_type is not None,
        'has_api_key_if_openai': (
            session_state.llm_type != LLMType.OPENAI or 
            (session_state.openai_api_key is not None and len(session_state.openai_api_key.strip()) > 0)
        ),
        'has_selected_part': session_state.selected_part is not None,
        'processing_complete': session_state.processing_complete,
        'has_retrievers': session_state.retrievers is not None,
        'ready_for_queries': session_state.is_ready_for_queries()
    }
    
    return validation