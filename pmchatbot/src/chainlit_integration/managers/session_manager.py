"""
Session Manager for Chainlit Integration.

This module provides session state management functionality including
initialization, cleanup, persistence, and part switching capabilities.
"""

from typing import Optional, Dict, Any
import logging

# Conditional import for Chainlit to avoid import errors in testing
try:
    import chainlit as cl
except ImportError:
    # Create a mock cl module for testing purposes
    class MockChainlit:
        class user_session:
            @staticmethod
            def get(key):
                return None
            @staticmethod
            def set(key, value):
                pass
    cl = MockChainlit()

from ..models import SessionState, LLMType, QueryContext, ErrorContext
from ..interfaces import SessionManagerInterface


logger = logging.getLogger(__name__)


class SessionManager(SessionManagerInterface):
    """
    Manages session state and lifecycle for Chainlit integration.
    
    Handles session initialization, cleanup, state persistence across
    interactions, and part switching functionality.
    """
    
    def __init__(self, session_state: Optional[SessionState] = None):
        """Initialize the session manager."""
        if session_state is None:
            session_state = SessionState()
        super().__init__(session_state)
        self._session_key = "process_mining_session"
    
    async def initialize(self) -> bool:
        """
        Initialize the manager.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            await self._store_session_state(self.session_state)
            logger.info("Session manager initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize session manager: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Clean up manager resources."""
        await self.cleanup_session()
    
    async def initialize_session(self) -> SessionState:
        """
        Initialize a new session with default state.
        
        Returns:
            SessionState: Newly initialized session state
        """
        try:
            session_state = SessionState()
            await self._store_session_state(session_state)
            
            logger.info("Session initialized successfully")
            return session_state
            
        except Exception as e:
            logger.error(f"Failed to initialize session: {e}")
            raise
    
    async def get_session_state(self) -> Optional[SessionState]:
        """
        Retrieve current session state.
        
        Returns:
            Optional[SessionState]: Current session state or None if not found
        """
        try:
            return cl.user_session.get(self._session_key)
        except Exception as e:
            logger.error(f"Failed to retrieve session state: {e}")
            return None
    
    async def update_session_state(self, session_state: SessionState) -> bool:
        """
        Update the current session state.
        
        Args:
            session_state: Updated session state to store
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            await self._store_session_state(session_state)
            logger.debug("Session state updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update session state: {e}")
            return False
    
    async def cleanup_session(self) -> bool:
        """
        Clean up session resources and reset state.
        
        Returns:
            bool: True if cleanup successful, False otherwise
        """
        try:
            session_state = await self.get_session_state()
            if session_state:
                # Clean up any temporary files or resources
                if hasattr(session_state, 'dfg_images') and session_state.dfg_images:
                    # Clean up visualization resources if needed
                    pass
                
                # Reset session state
                session_state.session_active = False
                await self._store_session_state(session_state)
            
            # Clear session from Chainlit
            cl.user_session.set(self._session_key, None)
            
            logger.info("Session cleaned up successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup session: {e}")
            return False
    
    async def reset_for_new_part(self) -> bool:
        """
        Reset session state for new part selection.
        
        This method implements requirement 5.1, 5.2, 5.3 for part switching
        functionality while maintaining session continuity.
        
        Returns:
            bool: True if reset successful, False otherwise
        """
        try:
            session_state = await self.get_session_state()
            if not session_state:
                logger.error("No session state found for part reset")
                return False
            
            # Reset part-specific state while preserving LLM configuration
            session_state.reset_for_new_part()
            
            # Update stored session state
            await self._store_session_state(session_state)
            
            logger.info("Session reset for new part selection")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset session for new part: {e}")
            return False
    
    async def handle_session_termination(self, message: str) -> bool:
        """
        Handle session termination keywords.
        
        Implements requirement 5.5 for session termination handling.
        
        Args:
            message: User message to check for termination keywords
            
        Returns:
            bool: True if session should be terminated, False otherwise
        """
        termination_keywords = ['quit', 'exit', 'end', 'stop']
        message_lower = message.lower().strip()
        
        if message_lower in termination_keywords:
            await self.cleanup_session()
            logger.info(f"Session terminated by keyword: {message_lower}")
            return True
        
        return False
    
    async def is_session_ready_for_queries(self) -> bool:
        """
        Check if session is ready to handle queries.
        
        Returns:
            bool: True if session is ready for queries, False otherwise
        """
        session_state = await self.get_session_state()
        if not session_state:
            return False
        
        return session_state.is_ready_for_queries()
    
    async def update_llm_configuration(self, llm_type: LLMType, api_key: Optional[str] = None) -> bool:
        """
        Update LLM configuration in session state.
        
        Args:
            llm_type: Type of LLM to configure
            api_key: API key for OpenAI (if applicable)
            
        Returns:
            bool: True if configuration updated successfully, False otherwise
        """
        try:
            session_state = await self.get_session_state()
            if not session_state:
                logger.error("No session state found for LLM configuration update")
                return False
            
            session_state.llm_type = llm_type
            if llm_type == LLMType.OPENAI and api_key:
                session_state.openai_api_key = api_key
            
            await self._store_session_state(session_state)
            
            logger.info(f"LLM configuration updated to {llm_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update LLM configuration: {e}")
            return False
    
    async def update_part_selection(self, selected_part: str, available_parts: list) -> bool:
        """
        Update part selection in session state.
        
        Args:
            selected_part: The selected part name
            available_parts: List of all available parts
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            session_state = await self.get_session_state()
            if not session_state:
                logger.error("No session state found for part selection update")
                return False
            
            session_state.selected_part = selected_part
            session_state.available_parts = available_parts
            session_state.data_loaded = True
            
            await self._store_session_state(session_state)
            
            logger.info(f"Part selection updated to: {selected_part}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update part selection: {e}")
            return False
    
    async def mark_processing_complete(self, retrievers: tuple, dfg_images: tuple) -> bool:
        """
        Mark data processing as complete and store results.
        
        Args:
            retrievers: Tuple of GraphRAG retrievers
            dfg_images: Tuple of DFG visualization images
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            session_state = await self.get_session_state()
            if not session_state:
                logger.error("No session state found for processing completion")
                return False
            
            session_state.processing_complete = True
            session_state.retrievers = retrievers
            session_state.dfg_images = dfg_images
            session_state.visualizations_displayed = True
            
            await self._store_session_state(session_state)
            
            logger.info("Processing marked as complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to mark processing complete: {e}")
            return False
    
    async def reset_query_context(self) -> bool:
        """
        Reset query context after each response.
        
        Implements requirement 4.5 for context reset after each response.
        
        Returns:
            bool: True if reset successful, False otherwise
        """
        try:
            session_state = await self.get_session_state()
            if not session_state:
                logger.error("No session state found for query context reset")
                return False
            
            session_state.reset_query_context()
            await self._store_session_state(session_state)
            
            logger.debug("Query context reset")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset query context: {e}")
            return False
    
    async def set_query_context(self, context: QueryContext) -> bool:
        """
        Set the current query context.
        
        Args:
            context: Query context to set
            
        Returns:
            bool: True if context set successfully, False otherwise
        """
        try:
            session_state = await self.get_session_state()
            if not session_state:
                logger.error("No session state found for query context setting")
                return False
            
            session_state.current_context_mode = context
            session_state.awaiting_context_selection = False
            
            await self._store_session_state(session_state)
            
            logger.debug(f"Query context set to: {context.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set query context: {e}")
            return False
    
    async def get_session_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current session state for debugging/monitoring.
        
        Returns:
            Dict[str, Any]: Session state summary
        """
        session_state = await self.get_session_state()
        if not session_state:
            return {"status": "no_session"}
        
        return {
            "status": "active" if session_state.session_active else "inactive",
            "llm_configured": session_state.llm_type is not None,
            "part_selected": session_state.selected_part is not None,
            "processing_complete": session_state.processing_complete,
            "ready_for_queries": session_state.is_ready_for_queries(),
            "current_context": session_state.current_context_mode.value if session_state.current_context_mode else None,
            "awaiting_context": session_state.awaiting_context_selection
        }
    
    async def save_session_state(self, session_state: SessionState) -> bool:
        """
        Save current session state.
        
        Args:
            session_state: Session state to save
            
        Returns:
            True if save was successful
        """
        return await self.update_session_state(session_state)
    
    async def restore_session_state(self) -> Optional[SessionState]:
        """
        Restore session state from storage.
        
        Returns:
            Restored session state or None if not found
        """
        return await self.get_session_state()
    
    async def clear_session(self) -> None:
        """Clear current session and reset to initial state."""
        await self.cleanup_session()
    
    async def _store_session_state(self, session_state: SessionState) -> None:
        """
        Store session state in Chainlit user session.
        
        Args:
            session_state: Session state to store
        """
        cl.user_session.set(self._session_key, session_state)