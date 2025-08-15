"""
Part Switching Manager for Chainlit Integration.

This module provides functionality for switching between different parts
during analysis sessions while maintaining chat continuity and handling
data reprocessing.
"""

from typing import Optional, List, Dict, Any
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
        class Message:
            def __init__(self, content: str):
                self.content = content
    cl = MockChainlit()

from ..models import SessionState, ProcessingResult, ErrorContext
from .session_manager import SessionManager


logger = logging.getLogger(__name__)


class PartSwitchingManager:
    """
    Manages part switching functionality during analysis sessions.
    
    Handles returning to part selection, data reprocessing for new parts,
    and maintaining chat continuity during switches.
    """
    
    def __init__(self, session_manager: SessionManager):
        """
        Initialize the part switching manager.
        
        Args:
            session_manager: Session manager instance for state management
        """
        self.session_manager = session_manager
    
    async def can_switch_parts(self) -> bool:
        """
        Check if part switching is currently allowed.
        
        Returns:
            bool: True if part switching is allowed, False otherwise
        """
        try:
            session_state = await self.session_manager.get_session_state()
            if not session_state:
                return False
            
            # Can switch if we have a session and LLM is configured
            return (
                session_state.session_active and
                session_state.llm_type is not None
            )
            
        except Exception as e:
            logger.error(f"Error checking part switch capability: {e}")
            return False
    
    async def initiate_part_switch(self) -> bool:
        """
        Initiate the part switching process.
        
        This method implements requirement 5.1 - provide option to return
        to part selection during analysis.
        
        Returns:
            bool: True if switch initiation was successful, False otherwise
        """
        try:
            if not await self.can_switch_parts():
                await cl.Message(
                    content="❌ Part switching is not available at this time. Please ensure you have configured an LLM first."
                ).send()
                return False
            
            session_state = await self.session_manager.get_session_state()
            
            # Show confirmation message about switching
            if session_state.selected_part:
                await cl.Message(
                    content=f"🔄 **Switching from current part:** {session_state.selected_part}\n\n"
                           f"Your previous analysis will be cleared, but your LLM configuration will be preserved. "
                           f"You can now select a new part to analyze."
                ).send()
            else:
                await cl.Message(
                    content="🔄 **Returning to part selection**\n\n"
                           f"You can now select a part to analyze."
                ).send()
            
            # Reset session for new part selection
            success = await self.session_manager.reset_for_new_part()
            
            if success:
                logger.info("Part switch initiated successfully")
                return True
            else:
                await cl.Message(
                    content="❌ Failed to initiate part switch. Please try again."
                ).send()
                return False
                
        except Exception as e:
            logger.error(f"Error initiating part switch: {e}")
            await cl.Message(
                content="❌ An error occurred while switching parts. Please try again."
            ).send()
            return False
    
    async def handle_new_part_selection(self, new_part: str, available_parts: List[str]) -> bool:
        """
        Handle selection of a new part after switching.
        
        This method implements requirement 5.2 - reprocess data and update
        Neo4j storage for new part selections.
        
        Args:
            new_part: The newly selected part
            available_parts: List of all available parts
            
        Returns:
            bool: True if new part selection was handled successfully
        """
        try:
            session_state = await self.session_manager.get_session_state()
            if not session_state:
                logger.error("No session state found for new part selection")
                return False
            
            # Update session state with new part
            success = await self.session_manager.update_part_selection(new_part, available_parts)
            if not success:
                await cl.Message(
                    content=f"❌ Failed to select part: {new_part}"
                ).send()
                return False
            
            # Show confirmation of new part selection
            await cl.Message(
                content=f"✅ **Part selected:** {new_part}\n\n"
                       f"Data processing will begin shortly. This includes:\n"
                       f"• Processing CSV data for the selected part\n"
                       f"• Generating process visualizations\n"
                       f"• Setting up GraphRAG retrievers\n"
                       f"• Storing processed data in Neo4j\n\n"
                       f"Please wait while we prepare your analysis environment..."
            ).send()
            
            logger.info(f"New part selected after switch: {new_part}")
            return True
            
        except Exception as e:
            logger.error(f"Error handling new part selection: {e}")
            await cl.Message(
                content=f"❌ An error occurred while selecting the new part: {new_part}"
            ).send()
            return False
    
    async def complete_part_switch(self, processing_result: ProcessingResult, 
                                 retrievers: tuple, dfg_images: tuple) -> bool:
        """
        Complete the part switching process after data reprocessing.
        
        This method implements requirement 5.3 - maintain session continuity
        during part switches.
        
        Args:
            processing_result: Result from data processing
            retrievers: Tuple of GraphRAG retrievers
            dfg_images: Tuple of DFG visualization images
            
        Returns:
            bool: True if part switch completion was successful
        """
        try:
            session_state = await self.session_manager.get_session_state()
            if not session_state:
                logger.error("No session state found for part switch completion")
                return False
            
            # Mark processing as complete
            success = await self.session_manager.mark_processing_complete(retrievers, dfg_images)
            if not success:
                await cl.Message(
                    content="❌ Failed to complete part switch processing."
                ).send()
                return False
            
            # Show completion message with continuity information
            await cl.Message(
                content=f"🎉 **Part switch completed successfully!**\n\n"
                       f"**Current part:** {session_state.selected_part}\n"
                       f"**Processing status:** Complete\n"
                       f"**Visualizations:** Generated and ready\n"
                       f"**Query system:** Ready for questions\n\n"
                       f"You can now:\n"
                       f"• View the generated process visualizations\n"
                       f"• Ask questions about this part's process\n"
                       f"• Switch to another part anytime by typing 'switch part'\n\n"
                       f"Your chat history and LLM configuration have been preserved."
            ).send()
            
            logger.info(f"Part switch completed successfully for: {session_state.selected_part}")
            return True
            
        except Exception as e:
            logger.error(f"Error completing part switch: {e}")
            await cl.Message(
                content="❌ An error occurred while completing the part switch."
            ).send()
            return False
    
    async def get_switch_status(self) -> Dict[str, Any]:
        """
        Get current part switching status information.
        
        Returns:
            Dict[str, Any]: Status information about part switching capability
        """
        try:
            session_state = await self.session_manager.get_session_state()
            if not session_state:
                return {
                    "can_switch": False,
                    "reason": "No active session",
                    "current_part": None,
                    "llm_configured": False
                }
            
            can_switch = await self.can_switch_parts()
            
            return {
                "can_switch": can_switch,
                "reason": "Ready for part switching" if can_switch else "LLM not configured",
                "current_part": session_state.selected_part,
                "llm_configured": session_state.llm_type is not None,
                "processing_complete": session_state.processing_complete,
                "session_active": session_state.session_active
            }
            
        except Exception as e:
            logger.error(f"Error getting switch status: {e}")
            return {
                "can_switch": False,
                "reason": f"Error: {str(e)}",
                "current_part": None,
                "llm_configured": False
            }
    
    async def handle_switch_command(self, message: str) -> bool:
        """
        Handle user commands related to part switching.
        
        Args:
            message: User message to check for switch commands
            
        Returns:
            bool: True if a switch command was handled, False otherwise
        """
        switch_commands = [
            'switch part', 'change part', 'select part', 'new part',
            'switch to', 'change to', 'go back', 'part selection'
        ]
        
        message_lower = message.lower().strip()
        
        for command in switch_commands:
            if command in message_lower:
                await self.initiate_part_switch()
                return True
        
        return False
    
    async def show_switch_help(self) -> None:
        """Show help information about part switching functionality."""
        try:
            status = await self.get_switch_status()
            
            help_message = "🔄 **Part Switching Help**\n\n"
            
            if status["can_switch"]:
                help_message += (
                    f"**Current part:** {status['current_part'] or 'None selected'}\n"
                    f"**Status:** Ready for switching\n\n"
                    f"**Available commands:**\n"
                    f"• `switch part` - Return to part selection\n"
                    f"• `change part` - Switch to a different part\n"
                    f"• `select part` - Go back to part selection\n"
                    f"• `new part` - Choose a new part to analyze\n\n"
                    f"**What happens when you switch:**\n"
                    f"• Your current analysis data will be cleared\n"
                    f"• Your LLM configuration will be preserved\n"
                    f"• You'll be able to select a new part\n"
                    f"• New visualizations will be generated\n"
                    f"• Chat history will be maintained\n"
                )
            else:
                help_message += (
                    f"**Status:** {status['reason']}\n\n"
                    f"**To enable part switching:**\n"
                    f"• First configure an LLM (OpenAI or Ollama)\n"
                    f"• Then you'll be able to switch between parts\n"
                )
            
            await cl.Message(content=help_message).send()
            
        except Exception as e:
            logger.error(f"Error showing switch help: {e}")
            await cl.Message(
                content="❌ Unable to display part switching help at this time."
            ).send()
    
    async def cleanup_switch_resources(self) -> None:
        """Clean up resources related to part switching."""
        try:
            # Any cleanup specific to part switching can be added here
            logger.debug("Part switching resources cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up switch resources: {e}")