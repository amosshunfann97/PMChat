"""
Query Context Manager for handling query context selection and management.

This module provides functionality for users to select query contexts
(activity, process, variant, or combined) before asking questions, and
manages context state throughout the query process.
"""

import logging
from typing import List, Optional

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
        
        class AskActionMessage:
            def __init__(self, content: str, actions: list, timeout: int = 60):
                self.content = content
                self.actions = actions
                self.timeout = timeout
            
            async def send(self):
                # Mock response for testing
                class MockResponse:
                    def __init__(self, value: str = "activity"):
                        self.value = value
                return MockResponse()
    
    cl = MockChainlit()

from ..interfaces import QueryContextManagerInterface
from ..models import SessionState, QueryContext, ErrorContext


logger = logging.getLogger(__name__)


class QueryContextManager(QueryContextManagerInterface):
    """
    Manager for query context selection and management.
    
    Handles the user interface for selecting query contexts before each question,
    validates context selections, and manages context state throughout the
    query process.
    """
    
    def __init__(self, session_state: SessionState):
        """
        Initialize Query Context Manager.
        
        Args:
            session_state: Current session state
        """
        super().__init__(session_state)
        self.logger = logger
        self._available_contexts = [
            QueryContext.ACTIVITY,
            QueryContext.PROCESS,
            QueryContext.VARIANT,
            QueryContext.COMBINED
        ]
    
    async def initialize(self) -> bool:
        """
        Initialize the Query Context Manager.
        
        Returns:
            True if initialization was successful
        """
        try:
            self.logger.info("Initializing Query Context Manager")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Query Context Manager: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Clean up manager resources."""
        self.logger.info("Cleaning up Query Context Manager")
        self.reset_context_selection()
    
    async def show_context_selector(self) -> None:
        """
        Display context selection interface.
        
        Shows action buttons for users to select their query context
        (activity, process, variant, or combined) before asking questions.
        """
        try:
            self.logger.info("Displaying context selection interface")
            
            # Create action buttons for context selection
            actions = [
                cl.Action(
                    name="select_activity",
                    value="activity",
                    label="🎯 Activity Context"
                ),
                cl.Action(
                    name="select_process",
                    value="process", 
                    label="🔄 Process Context"
                ),
                cl.Action(
                    name="select_variant",
                    value="variant",
                    label="🌟 Variant Context"
                ),
                cl.Action(
                    name="select_combined",
                    value="combined",
                    label="🔗 Combined Context"
                )
            ]
            
            # Send message with context selection options
            await cl.Message(
                content=self._get_context_selection_message(),
                actions=actions
            ).send()
            
            # Mark that we're awaiting context selection
            self.session_state.awaiting_context_selection = True
            
        except Exception as e:
            error_context = await self.handle_error(e, "show_context_selector")
            await cl.Message(content=error_context.format_user_message()).send()
    
    def _get_context_selection_message(self) -> str:
        """
        Get the context selection message with descriptions.
        
        Returns:
            Formatted message explaining context options
        """
        return """**🎯 Query Context Selection**

Please select the context for your question:

• **Activity Context** - Focus on specific activities and their relationships
• **Process Context** - Analyze overall process flow and structure  
• **Variant Context** - Examine process variants and their differences
• **Combined Context** - Use all available information for comprehensive analysis

Choose the context that best matches what you want to know about the process."""
    
    async def handle_context_selection(self, context: str) -> bool:
        """
        Handle context selection and validation.
        
        Args:
            context: Selected context type
            
        Returns:
            True if selection was valid and processed successfully
        """
        try:
            self.logger.info(f"Handling context selection: {context}")
            
            # Validate context selection
            if not self._is_valid_context(context):
                await cl.Message(
                    content="❌ Invalid context selection. Please choose from the available options."
                ).send()
                return False
            
            # Convert string to QueryContext enum
            selected_context = QueryContext(context)
            
            # Update session state
            self.session_state.current_context_mode = selected_context
            self.session_state.awaiting_context_selection = False
            
            # Send confirmation message
            context_name = self._get_context_display_name(selected_context)
            await cl.Message(
                content=f"✅ **{context_name}** selected. You can now ask your question!"
            ).send()
            
            self.logger.info(f"Context selection successful: {context}")
            return True
            
        except Exception as e:
            error_context = await self.handle_error(e, "handle_context_selection")
            await cl.Message(content=error_context.format_user_message()).send()
            return False
    
    def _is_valid_context(self, context: str) -> bool:
        """
        Validate if the provided context is valid.
        
        Args:
            context: Context string to validate
            
        Returns:
            True if context is valid
        """
        try:
            QueryContext(context)
            return True
        except ValueError:
            return False
    
    def _get_context_display_name(self, context: QueryContext) -> str:
        """
        Get display name for context.
        
        Args:
            context: QueryContext enum value
            
        Returns:
            Human-readable context name
        """
        context_names = {
            QueryContext.ACTIVITY: "Activity Context",
            QueryContext.PROCESS: "Process Context", 
            QueryContext.VARIANT: "Variant Context",
            QueryContext.COMBINED: "Combined Context"
        }
        return context_names.get(context, "Unknown Context")
    
    def reset_context_selection(self) -> None:
        """
        Reset context selection after each response.
        
        This method is called after each query response to ensure
        users must select a context for their next question.
        """
        self.logger.info("Resetting context selection")
        self.session_state.awaiting_context_selection = False
        self.session_state.current_context_mode = None
    
    def get_available_contexts(self) -> List[str]:
        """
        Get list of available query contexts.
        
        Returns:
            List of available context types as strings
        """
        return [context.value for context in self._available_contexts]
    
    def is_context_selected(self) -> bool:
        """
        Check if a context is currently selected.
        
        Returns:
            True if a context is selected and ready for queries
        """
        return (
            self.session_state.current_context_mode is not None and
            not self.session_state.awaiting_context_selection
        )
    
    def get_current_context(self) -> Optional[QueryContext]:
        """
        Get the currently selected context.
        
        Returns:
            Current QueryContext or None if no context selected
        """
        return self.session_state.current_context_mode
    
    def get_current_context_string(self) -> Optional[str]:
        """
        Get the currently selected context as a string.
        
        Returns:
            Current context as string or None if no context selected
        """
        if self.session_state.current_context_mode:
            return self.session_state.current_context_mode.value
        return None
    
    async def prompt_for_context_if_needed(self) -> bool:
        """
        Prompt for context selection if none is currently selected.
        
        Returns:
            True if context is available (either already selected or just selected)
        """
        if not self.is_context_selected():
            await cl.Message(
                content="🎯 **Context Required** - Please select a query context before asking your question."
            ).send()
            await self.show_context_selector()
            return False
        return True
    
    async def handle_context_reset_request(self) -> None:
        """
        Handle user request to reset/change context.
        
        Allows users to change their context selection during a session.
        """
        try:
            self.logger.info("Handling context reset request")
            
            self.reset_context_selection()
            
            await cl.Message(
                content="🔄 **Context Reset** - Please select a new query context."
            ).send()
            
            await self.show_context_selector()
            
        except Exception as e:
            error_context = await self.handle_error(e, "handle_context_reset_request")
            await cl.Message(content=error_context.format_user_message()).send()
    
    def validate_context_for_query(self, query: str) -> bool:
        """
        Validate that a context is selected before processing a query.
        
        Args:
            query: User query to validate context for
            
        Returns:
            True if context is properly selected for the query
        """
        if not self.is_context_selected():
            self.logger.warning(f"Query attempted without context selection: {query[:50]}...")
            return False
        
        self.logger.info(f"Context validation passed for query with context: {self.get_current_context_string()}")
        return True
    
    async def route_query_to_retriever(self, query: str, retrievers: tuple) -> dict:
        """
        Route query to appropriate retriever based on selected context.
        
        Args:
            query: User's natural language query
            retrievers: Tuple of (activity_retriever, process_retriever, variant_retriever)
            
        Returns:
            Dictionary containing retrieval results and metadata
        """
        try:
            if not self.is_context_selected():
                raise ValueError("No context selected for query routing")
            
            activity_retriever, process_retriever, variant_retriever = retrievers
            context = self.get_current_context()
            
            # Validate context is a proper QueryContext enum
            if not isinstance(context, QueryContext):
                raise ValueError(f"Invalid context type: {type(context)}")
            
            self.logger.info(f"Routing query to {context.value} context: {query[:50]}...")
            
            if context == QueryContext.ACTIVITY:
                return await self._query_single_context(
                    activity_retriever, query, "Activity-Based"
                )
            elif context == QueryContext.PROCESS:
                return await self._query_single_context(
                    process_retriever, query, "Process-Based"
                )
            elif context == QueryContext.VARIANT:
                return await self._query_single_context(
                    variant_retriever, query, "Variant-Based"
                )
            elif context == QueryContext.COMBINED:
                return await self._query_combined_context(
                    activity_retriever, process_retriever, variant_retriever, query
                )
            else:
                raise ValueError(f"Unknown context type: {context}")
                
        except Exception as e:
            error_context = await self.handle_error(e, "route_query_to_retriever")
            self.logger.error(f"Query routing failed: {error_context.technical_details}")
            raise
    
    async def _query_single_context(self, retriever, query: str, context_label: str) -> dict:
        """
        Query a single retriever context.
        
        Args:
            retriever: The retriever to query
            query: User's query
            context_label: Label for the context type
            
        Returns:
            Dictionary with retrieval results
        """
        try:
            self.logger.info(f"Querying {context_label} context")
            
            # Perform the search
            search_result = retriever.search(query)
            
            # Extract content and metadata
            retrieved_items = []
            for item in search_result.items:
                content = self._extract_content(item.content)
                metadata = item.metadata or {}
                retrieved_items.append({
                    "content": content,
                    "metadata": metadata
                })
            
            # Combine all retrieved content
            retrieved_info = "\n".join(item["content"] for item in retrieved_items)
            
            return {
                "context_type": context_label,
                "retrieved_items": retrieved_items,
                "retrieved_info": retrieved_info,
                "query": query
            }
            
        except Exception as e:
            self.logger.error(f"Single context query failed for {context_label}: {e}")
            raise
    
    async def _query_combined_context(self, activity_retriever, process_retriever, 
                                    variant_retriever, query: str) -> dict:
        """
        Query all three retriever contexts and combine results.
        
        Args:
            activity_retriever: Activity context retriever
            process_retriever: Process context retriever  
            variant_retriever: Variant context retriever
            query: User's query
            
        Returns:
            Dictionary with combined retrieval results
        """
        try:
            self.logger.info("Querying combined context")
            
            # Query all three contexts
            activity_result = await self._query_single_context(
                activity_retriever, query, "Activity-Based"
            )
            process_result = await self._query_single_context(
                process_retriever, query, "Process-Based"
            )
            variant_result = await self._query_single_context(
                variant_retriever, query, "Variant-Based"
            )
            
            return {
                "context_type": "Combined",
                "activity_results": activity_result,
                "process_results": process_result,
                "variant_results": variant_result,
                "query": query
            }
            
        except Exception as e:
            self.logger.error(f"Combined context query failed: {e}")
            raise
    
    def _extract_content(self, content) -> str:
        """
        Extract actual content from record format.
        
        Args:
            content: Raw content from retriever
            
        Returns:
            Cleaned content string
        """
        content_str = str(content)
        if content_str.startswith('<Record text="') and content_str.endswith('">'):
            return content_str[14:-2]
        else:
            return content_str
    
    def validate_retrievers(self, retrievers: tuple) -> bool:
        """
        Validate that retrievers tuple is properly formatted.
        
        Args:
            retrievers: Tuple of retrievers to validate
            
        Returns:
            True if retrievers are valid
        """
        try:
            if not isinstance(retrievers, tuple) or len(retrievers) != 3:
                self.logger.error("Retrievers must be a tuple of 3 elements")
                return False
            
            activity_retriever, process_retriever, variant_retriever = retrievers
            
            # Check that all retrievers have a search method
            for i, retriever in enumerate(retrievers):
                if not hasattr(retriever, 'search'):
                    self.logger.error(f"Retriever {i} does not have search method")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Retriever validation failed: {e}")
            return False
    
    async def handle_query_routing_error(self, error: Exception, query: str) -> str:
        """
        Handle errors that occur during query routing.
        
        Args:
            error: The exception that occurred
            query: The query that caused the error
            
        Returns:
            User-friendly error message
        """
        try:
            error_type = type(error).__name__
            
            if "context" in str(error).lower():
                message = "❌ **Context Error** - Please select a query context before asking questions."
                await cl.Message(content=message).send()
                await self.show_context_selector()
                return message
            
            elif "retriever" in str(error).lower():
                message = "❌ **Retrieval Error** - There was an issue accessing the process data. Please try again."
                await cl.Message(content=message).send()
                return message
            
            else:
                message = f"❌ **Query Error** - An error occurred while processing your question: {str(error)}"
                await cl.Message(content=message).send()
                return message
                
        except Exception as e:
            self.logger.error(f"Error handling query routing error: {e}")
            return "❌ An unexpected error occurred. Please try again."
    
    def get_context_routing_info(self) -> dict:
        """
        Get information about current context routing configuration.
        
        Returns:
            Dictionary with routing information
        """
        return {
            "current_context": self.get_current_context_string(),
            "is_context_selected": self.is_context_selected(),
            "awaiting_selection": self.session_state.awaiting_context_selection,
            "available_contexts": self.get_available_contexts()
        }
    
    async def show_context_help(self) -> None:
        """
        Show detailed help about query contexts.
        
        Provides users with detailed information about what each
        context type is used for and when to select each one.
        """
        try:
            help_message = """**📚 Query Context Help**

**🎯 Activity Context**
- Best for questions about specific activities in your process
- Examples: "What activities come after X?", "How long does activity Y take?"
- Focuses on individual process steps and their relationships

**🔄 Process Context** 
- Best for questions about overall process flow and structure
- Examples: "What is the main process flow?", "How does the process start/end?"
- Provides high-level process insights

**🌟 Variant Context**
- Best for questions about different process variants and paths
- Examples: "What are the different ways this process can flow?", "Which variant is most common?"
- Analyzes process variations and alternatives

**🔗 Combined Context**
- Best for comprehensive questions requiring all available information
- Examples: "Give me a complete analysis", "What insights can you provide?"
- Uses all data sources for thorough analysis

Choose the context that best matches your question type for optimal results."""
            
            await cl.Message(content=help_message).send()
            
        except Exception as e:
            error_context = await self.handle_error(e, "show_context_help")
            await cl.Message(content=error_context.format_user_message()).send()