"""
Chat Query Handler for processing natural language queries using GraphRAG.

This module handles query processing, response generation, and session termination
for the Chainlit integration. It integrates existing GraphRAG retrievers with
Chainlit, routes queries to appropriate retrievers based on context, and hides
technical retrieval details from the user interface.
"""

import logging
import time
from typing import Optional, Tuple, Any, Dict, List

# Conditional import for Chainlit to avoid import errors in testing
try:
    import chainlit as cl
except ImportError:
    # Create a mock cl module for testing purposes
    class MockChainlit:
        class Message:
            def __init__(self, content: str):
                self.content = content
            
            async def send(self):
                pass
    
    cl = MockChainlit()

# Import models with fallback for testing
try:
    from ..models import SessionState, QueryContext, ErrorContext
except ImportError:
    # Mock models for testing
    from enum import Enum
    
    class QueryContext(Enum):
        ACTIVITY = "activity"
        PROCESS = "process"
        VARIANT = "variant"
        COMBINED = "combined"
    
    class SessionState:
        def __init__(self):
            self.retrievers = None
            self.selected_part = None
            self.session_active = True
        
        def reset_query_context(self):
            pass
    
    class ErrorContext:
        def __init__(self, **kwargs):
            self.technical_details = kwargs.get('technical_details', '')
        
        def format_user_message(self):
            return "An error occurred"

# Conditional imports to handle missing dependencies in testing
try:
    from config.settings import PROCESS_MINING_CONTEXT
except ImportError:
    PROCESS_MINING_CONTEXT = "You are a process mining expert."

try:
    from llm.llm_factory import get_llm
except ImportError:
    # Mock get_llm for testing
    def get_llm():
        class MockLLM:
            def invoke(self, prompt):
                class MockResponse:
                    content = "Mock response for testing"
                return MockResponse()
        return MockLLM()

logger = logging.getLogger(__name__)


class ChatQueryHandler:
    """
    Handler for processing natural language queries using GraphRAG.
    
    This class integrates existing GraphRAG retrievers with Chainlit,
    routes queries to appropriate retrievers based on context, and
    generates clean responses without technical details.
    """
    
    def __init__(self, session_state: SessionState):
        """
        Initialize Chat Query Handler.
        
        Args:
            session_state: Current session state
        """
        self.session_state = session_state
        self.logger = logger
        self._termination_keywords = {'quit', 'exit', 'end', 'stop'}
    
    async def initialize(self) -> bool:
        """
        Initialize the Chat Query Handler.
        
        Returns:
            True if initialization was successful
        """
        try:
            self.logger.info("Initializing Chat Query Handler")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Chat Query Handler: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Clean up handler resources."""
        self.logger.info("Cleaning up Chat Query Handler")
    
    async def handle_query(self, query: str, context_mode: str) -> str:
        """
        Handle natural language query with specified context.
        
        Args:
            query: User's natural language query
            context_mode: Selected query context
            
        Returns:
            Generated response
        """
        try:
            self.logger.info(f"Handling query with context {context_mode}: {query[:50]}...")
            
            # Check for termination keywords first
            if self.check_termination_keywords(query):
                return await self._handle_session_termination()
            
            # Validate that we have retrievers
            if not self.session_state.retrievers:
                raise ValueError("No retrievers available - data processing may not be complete")
            
            # Route query to appropriate retriever based on context
            retrieval_result = await self._route_query_to_retriever(query, context_mode)
            
            # Generate response using selected LLM
            response = await self._generate_response(query, retrieval_result, context_mode)
            
            # Format clean response without technical details
            clean_response = self.format_clean_response(response)
            
            self.logger.info(f"Query processed successfully for context: {context_mode}")
            return clean_response
            
        except Exception as e:
            error_context = await self.handle_error(e, "handle_query")
            self.logger.error(f"Query handling failed: {error_context.technical_details}")
            return error_context.format_user_message()
    
    async def _route_query_to_retriever(self, query: str, context_mode: str) -> dict:
        """
        Route query to appropriate retriever based on context.
        
        Args:
            query: User's natural language query
            context_mode: Selected query context
            
        Returns:
            Dictionary containing retrieval results
        """
        try:
            activity_retriever, process_retriever, variant_retriever = self.session_state.retrievers
            
            # Convert context mode to QueryContext enum if it's a string
            if isinstance(context_mode, str):
                context = QueryContext(context_mode)
            else:
                context = context_mode
            
            self.logger.info(f"Routing query to {context.value} context")
            
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
            self.logger.error(f"Query routing failed: {e}")
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
            
            # Extract content and metadata (hide from user)
            retrieved_items = []
            for item in search_result.items:
                content = self._extract_content(item.content)
                metadata = item.metadata or {}
                retrieved_items.append({
                    "content": content,
                    "metadata": metadata
                })
            
            # Combine all retrieved content for LLM processing
            retrieved_info = "\n".join(item["content"] for item in retrieved_items)
            
            return {
                "context_type": context_label,
                "retrieved_info": retrieved_info,
                "query": query,
                "item_count": len(retrieved_items)
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
            
            # Combine retrieved information
            combined_info = f"""Activity-Based Information:
{activity_result['retrieved_info']}

Process-Based Information:
{process_result['retrieved_info']}

Variant-Based Information:
{variant_result['retrieved_info']}"""
            
            return {
                "context_type": "Combined",
                "retrieved_info": combined_info,
                "query": query,
                "activity_count": activity_result["item_count"],
                "process_count": process_result["item_count"],
                "variant_count": variant_result["item_count"]
            }
            
        except Exception as e:
            self.logger.error(f"Combined context query failed: {e}")
            raise
    
    async def _generate_response(self, query: str, retrieval_result: dict, context_mode: str) -> str:
        """
        Generate response using selected LLM.
        
        Args:
            query: Original user query
            retrieval_result: Results from retriever
            context_mode: Query context mode
            
        Returns:
            Generated response from LLM
        """
        try:
            self.logger.info("Generating LLM response")
            
            # Get the selected part for context
            selected_part = self.session_state.selected_part or "All"
            
            # Create enhanced prompt based on context type
            if retrieval_result["context_type"] == "Combined":
                enhanced_prompt = f"""{PROCESS_MINING_CONTEXT}

Manufacturing Process Context: You are analyzing a manufacturing process for part: {selected_part}. You have access to ACTIVITY-BASED, PROCESS-BASED, and VARIANT-BASED information.

{retrieval_result['retrieved_info']}

User Question: {query}

Please provide a detailed process mining analysis based on the retrieved information from all three perspectives. Focus on actionable insights and avoid mentioning technical retrieval details."""
            else:
                enhanced_prompt = f"""{PROCESS_MINING_CONTEXT}

Manufacturing Process Context: You are analyzing a manufacturing process for part: {selected_part}. You have access to {retrieval_result['context_type']} information.

Retrieved Information: {retrieval_result['retrieved_info']}

User Question: {query}

Please provide a detailed process mining analysis based on the retrieved information. Focus on actionable insights and avoid mentioning technical retrieval details."""
            
            # Generate response using LLM
            llm = get_llm()
            answer_start_time = time.time()
            enhanced_answer = llm.invoke(enhanced_prompt)
            answer_end_time = time.time()
            
            # Log response time for debugging
            elapsed = answer_end_time - answer_start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.logger.info(f"LLM response generated in {minutes}m {seconds}s")
            
            return enhanced_answer.content
            
        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
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
    
    def check_termination_keywords(self, query: str) -> bool:
        """
        Check if query contains session termination keywords.
        
        Args:
            query: User query to check
            
        Returns:
            True if query contains termination keywords
        """
        query_lower = query.lower().strip()
        
        # Check for exact matches or queries that are just termination keywords
        if query_lower in self._termination_keywords:
            return True
        
        # Check for queries that start with termination keywords
        for keyword in self._termination_keywords:
            if query_lower.startswith(keyword + ' ') or query_lower.startswith(keyword):
                return True
        
        return False
    
    async def _handle_session_termination(self) -> str:
        """
        Handle session termination request.
        
        Returns:
            Termination message
        """
        try:
            self.logger.info("Handling session termination request")
            
            # Reset session state
            self.session_state.session_active = False
            self.session_state.reset_query_context()
            
            # Send termination message
            termination_message = """🔚 **Session Ended**

Thank you for using the Process Mining Chatbot! Your session has been terminated.

To start a new analysis:
1. Refresh the page to start over
2. Select a new part to analyze  
3. Begin asking questions about your process data

Have a great day! 👋"""
            
            await cl.Message(content=termination_message).send()
            
            return termination_message
            
        except Exception as e:
            self.logger.error(f"Session termination handling failed: {e}")
            return "Session terminated. Please refresh to start a new session."
    
    def format_clean_response(self, response: str) -> str:
        """
        Format response to hide technical details.
        
        Args:
            response: Raw response from LLM
            
        Returns:
            Cleaned response for user display
        """
        try:
            # Remove any technical retrieval details that might have leaked through
            clean_response = response
            
            # Remove common technical terms that shouldn't appear in user responses
            technical_terms_to_remove = [
                "chunk", "chunks", "retrieval", "retrieved", "embedding", "embeddings",
                "vector", "vectors", "similarity", "score", "metadata", "neo4j",
                "graphrag", "rag", "reranker", "top_k"
            ]
            
            # Don't remove these terms if they're part of meaningful content
            # Just clean up obvious technical references
            lines = clean_response.split('\n')
            cleaned_lines = []
            
            for line in lines:
                line_lower = line.lower()
                
                # Skip lines that are clearly technical debug info
                if any(term in line_lower for term in ["chunk ", "retrieved ", "metadata:", "score:"]):
                    continue
                
                # Skip lines that look like technical output
                if line.strip().startswith(('Chunk ', 'Retrieved ', 'Score:', 'Metadata:')):
                    continue
                
                cleaned_lines.append(line)
            
            clean_response = '\n'.join(cleaned_lines)
            
            # Ensure response is conversational and user-friendly
            if not clean_response.strip():
                clean_response = "I apologize, but I couldn't generate a proper response to your question. Please try rephrasing your question or select a different context."
            
            return clean_response.strip()
            
        except Exception as e:
            self.logger.error(f"Response formatting failed: {e}")
            return response  # Return original response if formatting fails
    
    async def process_with_context(self, query: str, context: str) -> str:
        """
        Process query with specific context (convenience method).
        
        Args:
            query: User's natural language query
            context: Query context type
            
        Returns:
            Processed response
        """
        return await self.handle_query(query, context)
    
    def validate_query_readiness(self) -> bool:
        """
        Validate that the handler is ready to process queries.
        
        Returns:
            True if ready to process queries
        """
        return (
            self.session_state.retrievers is not None and
            self.session_state.processing_complete and
            self.session_state.session_active
        )
    
    async def handle_query_error(self, error: Exception, query: str, context: str) -> str:
        """
        Handle errors that occur during query processing.
        
        Args:
            error: The exception that occurred
            query: The query that caused the error
            context: The context in which the error occurred
            
        Returns:
            User-friendly error message
        """
        try:
            error_type = type(error).__name__
            
            if "retriever" in str(error).lower() or "search" in str(error).lower():
                message = "❌ **Retrieval Error** - There was an issue accessing the process data. Please try rephrasing your question or select a different context."
            elif "llm" in str(error).lower() or "model" in str(error).lower():
                message = "❌ **LLM Error** - There was an issue generating a response. Please check your LLM configuration and try again."
            elif "context" in str(error).lower():
                message = "❌ **Context Error** - Please ensure you have selected a valid query context before asking questions."
            else:
                message = f"❌ **Query Error** - An error occurred while processing your question. Please try again or rephrase your question."
            
            await cl.Message(content=message).send()
            return message
                
        except Exception as e:
            self.logger.error(f"Error handling query error: {e}")
            return "❌ An unexpected error occurred. Please try again."
    
    async def handle_query_error(self, error: Exception, query: str, context: str) -> str:
        """
        Handle errors that occur during query processing.
        
        Args:
            error: The exception that occurred
            query: The query that caused the error
            context: The context in which the error occurred
            
        Returns:
            User-friendly error message
        """
        try:
            error_type = type(error).__name__
            
            if "retriever" in str(error).lower() or "search" in str(error).lower():
                message = "❌ **Retrieval Error** - There was an issue accessing the process data. Please try rephrasing your question or select a different context."
            elif "llm" in str(error).lower() or "model" in str(error).lower():
                message = "❌ **LLM Error** - There was an issue generating a response. Please check your LLM configuration and try again."
            elif "context" in str(error).lower():
                message = "❌ **Context Error** - Please ensure you have selected a valid query context before asking questions."
            else:
                message = f"❌ **Query Error** - An error occurred while processing your question. Please try again or rephrase your question."
            
            await cl.Message(content=message).send()
            return message
                
        except Exception as e:
            self.logger.error(f"Error handling query error: {e}")
            return "❌ An unexpected error occurred. Please try again."
    
    def get_query_statistics(self) -> dict:
        """
        Get statistics about query processing.
        
        Returns:
            Dictionary with query processing statistics
        """
        return {
            "session_active": self.session_state.session_active,
            "retrievers_available": self.session_state.retrievers is not None,
            "processing_complete": self.session_state.processing_complete,
            "ready_for_queries": self.validate_query_readiness()
        }