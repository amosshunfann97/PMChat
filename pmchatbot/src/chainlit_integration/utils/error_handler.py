"""
Error handling utilities for Chainlit integration.

This module provides comprehensive error handling and recovery mechanisms
for the Chainlit integration components.
"""

import logging
import traceback
from typing import Dict, List, Optional, Any
from enum import Enum

from ..models import ErrorContext
from ..interfaces import ErrorHandlerInterface


class ErrorType(Enum):
    """Enumeration of error types for categorization."""
    CONFIGURATION_ERROR = "Configuration Error"
    CONNECTION_ERROR = "Connection Error"
    DATA_PROCESSING_ERROR = "Data Processing Error"
    VALIDATION_ERROR = "Validation Error"
    RESOURCE_ERROR = "Resource Error"
    LLM_ERROR = "LLM Error"
    VISUALIZATION_ERROR = "Visualization Error"
    SESSION_ERROR = "Session Error"
    UNKNOWN_ERROR = "Unknown Error"


class ErrorSeverity(Enum):
    """Enumeration of error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorHandler(ErrorHandlerInterface):
    """
    Comprehensive error handler for Chainlit integration.
    
    Provides error categorization, user-friendly messages, recovery suggestions,
    and fallback options for failed operations.
    """
    
    def __init__(self):
        """Initialize the error handler."""
        self.logger = logging.getLogger(__name__)
        self._error_mappings = self._initialize_error_mappings()
        self._recovery_suggestions = self._initialize_recovery_suggestions()
        self._fallback_options = self._initialize_fallback_options()
    
    def _initialize_error_mappings(self) -> Dict[str, ErrorType]:
        """Initialize error type mappings."""
        return {
            'ConnectionError': ErrorType.CONNECTION_ERROR,
            'Neo4jError': ErrorType.CONNECTION_ERROR,
            'OpenAIError': ErrorType.LLM_ERROR,
            'ValidationError': ErrorType.VALIDATION_ERROR,
            'ValueError': ErrorType.VALIDATION_ERROR,
            'FileNotFoundError': ErrorType.CONFIGURATION_ERROR,
            'PermissionError': ErrorType.RESOURCE_ERROR,
            'MemoryError': ErrorType.RESOURCE_ERROR,
            'TimeoutError': ErrorType.CONNECTION_ERROR,
            'KeyError': ErrorType.CONFIGURATION_ERROR,
            'ImportError': ErrorType.CONFIGURATION_ERROR,
            'ModuleNotFoundError': ErrorType.CONFIGURATION_ERROR
        }
    
    def _initialize_recovery_suggestions(self) -> Dict[ErrorType, List[str]]:
        """Initialize recovery suggestions for each error type."""
        return {
            ErrorType.CONFIGURATION_ERROR: [
                "Check your configuration settings",
                "Verify all required environment variables are set",
                "Ensure all required files exist and are accessible",
                "Review the setup documentation",
                "Try using default configuration values"
            ],
            ErrorType.CONNECTION_ERROR: [
                "Check your network connection",
                "Verify Neo4j is running and accessible",
                "Check firewall settings",
                "Retry the operation after a few moments",
                "Try connecting with reduced timeout settings"
            ],
            ErrorType.DATA_PROCESSING_ERROR: [
                "Verify your CSV data format is correct",
                "Check if the selected part exists in the data",
                "Ensure the data contains required columns",
                "Try with a different part selection",
                "Use a smaller subset of data for testing"
            ],
            ErrorType.VALIDATION_ERROR: [
                "Check your input format",
                "Verify all required fields are provided",
                "Ensure values are within expected ranges",
                "Review the input requirements",
                "Try with simplified input parameters"
            ],
            ErrorType.RESOURCE_ERROR: [
                "Check available system memory",
                "Verify disk space is sufficient",
                "Close other applications to free resources",
                "Try processing smaller data sets",
                "Restart the application to free memory"
            ],
            ErrorType.LLM_ERROR: [
                "Verify your API key is correct and active",
                "Check your internet connection",
                "Ensure you have sufficient API credits",
                "Try switching to a different LLM model",
                "Use local LLM as fallback option"
            ],
            ErrorType.VISUALIZATION_ERROR: [
                "Check if the visualization data is valid",
                "Verify temporary directory permissions",
                "Ensure required visualization libraries are installed",
                "Try regenerating the visualization",
                "Use simplified visualization options"
            ],
            ErrorType.SESSION_ERROR: [
                "Try refreshing the page",
                "Clear your browser cache",
                "Start a new session",
                "Check if cookies are enabled",
                "Try using a different browser"
            ],
            ErrorType.UNKNOWN_ERROR: [
                "Try the operation again",
                "Restart the application",
                "Check the logs for more details",
                "Contact support if the issue persists",
                "Try with minimal configuration"
            ]
        }
    
    def _initialize_fallback_options(self) -> Dict[ErrorType, Dict[str, Any]]:
        """Initialize fallback options for different error types."""
        return {
            ErrorType.CONFIGURATION_ERROR: {
                "use_defaults": True,
                "skip_optional_features": True,
                "minimal_config": {
                    "neo4j_timeout": 30,
                    "chunk_size": 1000,
                    "max_retries": 3
                }
            },
            ErrorType.CONNECTION_ERROR: {
                "retry_count": 3,
                "retry_delay": 5,
                "use_offline_mode": True,
                "fallback_storage": "local_cache"
            },
            ErrorType.DATA_PROCESSING_ERROR: {
                "use_sample_data": True,
                "skip_validation": False,
                "simplified_processing": True,
                "max_rows": 10000
            },
            ErrorType.LLM_ERROR: {
                "switch_to_local": True,
                "use_cached_responses": True,
                "simplified_prompts": True,
                "fallback_model": "ollama"
            },
            ErrorType.VISUALIZATION_ERROR: {
                "use_basic_plots": True,
                "skip_complex_features": True,
                "text_only_mode": True,
                "export_raw_data": True
            },
            ErrorType.RESOURCE_ERROR: {
                "reduce_memory_usage": True,
                "process_in_chunks": True,
                "cleanup_temp_files": True,
                "use_streaming": True
            },
            ErrorType.SESSION_ERROR: {
                "reset_session": True,
                "use_stateless_mode": True,
                "clear_cache": True
            },
            ErrorType.VALIDATION_ERROR: {
                "use_lenient_validation": True,
                "skip_optional_fields": True,
                "auto_correct_format": True
            },
            ErrorType.UNKNOWN_ERROR: {
                "safe_mode": True,
                "minimal_features": True,
                "debug_mode": True
            }
        }
    
    def _categorize_error(self, error: Exception) -> ErrorType:
        """
        Categorize an error based on its type.
        
        Args:
            error: Exception to categorize
            
        Returns:
            Categorized error type
        """
        error_name = type(error).__name__
        return self._error_mappings.get(error_name, ErrorType.UNKNOWN_ERROR)
    
    def _determine_severity(self, error: Exception, context: str) -> ErrorSeverity:
        """
        Determine error severity based on error type and context.
        
        Args:
            error: Exception that occurred
            context: Context where error occurred
            
        Returns:
            Error severity level
        """
        error_type = self._categorize_error(error)
        
        # Critical errors that prevent core functionality
        if error_type in [ErrorType.CONFIGURATION_ERROR, ErrorType.CONNECTION_ERROR]:
            return ErrorSeverity.CRITICAL
        
        # High severity errors that affect major features
        if error_type in [ErrorType.DATA_PROCESSING_ERROR, ErrorType.LLM_ERROR]:
            return ErrorSeverity.HIGH
        
        # Medium severity errors that affect specific operations
        if error_type in [ErrorType.VALIDATION_ERROR, ErrorType.VISUALIZATION_ERROR]:
            return ErrorSeverity.MEDIUM
        
        # Low severity errors that are recoverable
        return ErrorSeverity.LOW
    
    async def handle_error(self, error: Exception, context: str) -> str:
        """
        Handle error and return user-friendly message.
        
        Args:
            error: Exception that occurred
            context: Context where error occurred
            
        Returns:
            User-friendly error message
        """
        error_type = self._categorize_error(error)
        severity = self._determine_severity(error, context)
        
        # Log the error
        self.log_error(error, context)
        
        # Create error context
        error_context = ErrorContext(
            error_type=error_type.value,
            component=context,
            operation=context,
            user_message=self._create_user_message(error, error_type, severity),
            technical_details=str(error),
            recovery_suggestions=self._recovery_suggestions.get(error_type, [])
        )
        
        return error_context.format_user_message()
    
    def _create_user_message(self, error: Exception, error_type: ErrorType, severity: ErrorSeverity) -> str:
        """
        Create user-friendly error message.
        
        Args:
            error: Exception that occurred
            error_type: Categorized error type
            severity: Error severity
            
        Returns:
            User-friendly error message
        """
        severity_emoji = {
            ErrorSeverity.LOW: "⚠️",
            ErrorSeverity.MEDIUM: "🔶",
            ErrorSeverity.HIGH: "🔴",
            ErrorSeverity.CRITICAL: "🚨"
        }
        
        base_messages = {
            ErrorType.CONFIGURATION_ERROR: "There's an issue with the system configuration",
            ErrorType.CONNECTION_ERROR: "Unable to connect to required services",
            ErrorType.DATA_PROCESSING_ERROR: "Error processing your data",
            ErrorType.VALIDATION_ERROR: "Invalid input provided",
            ErrorType.RESOURCE_ERROR: "System resources are insufficient",
            ErrorType.LLM_ERROR: "Error communicating with the language model",
            ErrorType.VISUALIZATION_ERROR: "Error generating visualizations",
            ErrorType.SESSION_ERROR: "Session management error occurred",
            ErrorType.UNKNOWN_ERROR: "An unexpected error occurred"
        }
        
        emoji = severity_emoji.get(severity, "❓")
        base_message = base_messages.get(error_type, "An error occurred")
        
        return f"{emoji} {base_message}"
    
    def log_error(self, error: Exception, context: str) -> None:
        """
        Log error for debugging purposes.
        
        Args:
            error: Exception that occurred
            context: Context where error occurred
        """
        error_type = self._categorize_error(error)
        severity = self._determine_severity(error, context)
        
        log_message = f"Error in {context}: {error_type.value} ({severity.value})"
        
        if severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            self.logger.error(log_message, exc_info=True)
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    async def suggest_recovery(self, error_type: str) -> List[str]:
        """
        Suggest recovery actions for specific error types.
        
        Args:
            error_type: Type of error that occurred
            
        Returns:
            List of recovery suggestions
        """
        # Convert string to ErrorType enum
        try:
            error_enum = ErrorType(error_type)
        except ValueError:
            error_enum = ErrorType.UNKNOWN_ERROR
        
        return self._recovery_suggestions.get(error_enum, [])
    
    def create_error_context(self, error: Exception, component: str, operation: str) -> ErrorContext:
        """
        Create structured error context.
        
        Args:
            error: Exception that occurred
            component: Component where error occurred
            operation: Operation being performed
            
        Returns:
            Structured error context
        """
        error_type = self._categorize_error(error)
        severity = self._determine_severity(error, operation)
        
        return ErrorContext(
            error_type=error_type.value,
            component=component,
            operation=operation,
            user_message=self._create_user_message(error, error_type, severity),
            technical_details=str(error),
            recovery_suggestions=self._recovery_suggestions.get(error_type, [])
        )
    
    def get_fallback_options(self, error_type: ErrorType) -> Dict[str, Any]:
        """
        Get fallback options for specific error type.
        
        Args:
            error_type: Type of error that occurred
            
        Returns:
            Dictionary of fallback options
        """
        return self._fallback_options.get(error_type, {})
    
    async def apply_fallback_strategy(self, error: Exception, context: str, operation_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Apply fallback strategy based on error type and context.
        
        Args:
            error: Exception that occurred
            context: Context where error occurred
            operation_data: Optional data about the failed operation
            
        Returns:
            Dictionary containing fallback configuration and actions
        """
        error_type = self._categorize_error(error)
        fallback_options = self.get_fallback_options(error_type)
        
        fallback_strategy = {
            "error_type": error_type.value,
            "context": context,
            "fallback_applied": True,
            "options": fallback_options,
            "actions": []
        }
        
        # Apply specific fallback actions based on error type
        if error_type == ErrorType.LLM_ERROR:
            fallback_strategy["actions"].extend([
                "switch_to_local_llm",
                "use_simplified_prompts",
                "enable_response_caching"
            ])
        
        elif error_type == ErrorType.CONNECTION_ERROR:
            fallback_strategy["actions"].extend([
                "enable_offline_mode",
                "use_local_cache",
                "reduce_timeout_settings"
            ])
        
        elif error_type == ErrorType.DATA_PROCESSING_ERROR:
            fallback_strategy["actions"].extend([
                "use_sample_data",
                "enable_simplified_processing",
                "reduce_data_size"
            ])
        
        elif error_type == ErrorType.VISUALIZATION_ERROR:
            fallback_strategy["actions"].extend([
                "use_basic_plots",
                "enable_text_mode",
                "export_raw_data"
            ])
        
        elif error_type == ErrorType.RESOURCE_ERROR:
            fallback_strategy["actions"].extend([
                "enable_memory_optimization",
                "process_in_chunks",
                "cleanup_resources"
            ])
        
        elif error_type == ErrorType.CONFIGURATION_ERROR:
            fallback_strategy["actions"].extend([
                "use_default_config",
                "skip_optional_features",
                "enable_minimal_mode"
            ])
        
        self.logger.info(f"Applied fallback strategy for {error_type.value} in {context}")
        return fallback_strategy
    
    async def execute_recovery_action(self, action: str, context: str, **kwargs) -> bool:
        """
        Execute specific recovery action.
        
        Args:
            action: Recovery action to execute
            context: Context where recovery is needed
            **kwargs: Additional parameters for the recovery action
            
        Returns:
            True if recovery action was successful
        """
        try:
            if action == "switch_to_local_llm":
                return await self._switch_to_local_llm(**kwargs)
            elif action == "use_sample_data":
                return await self._use_sample_data(**kwargs)
            elif action == "enable_offline_mode":
                return await self._enable_offline_mode(**kwargs)
            elif action == "cleanup_resources":
                return await self._cleanup_resources(**kwargs)
            elif action == "reset_session":
                return await self._reset_session(**kwargs)
            elif action == "use_basic_plots":
                return await self._use_basic_plots(**kwargs)
            else:
                self.logger.warning(f"Unknown recovery action: {action}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to execute recovery action {action}: {str(e)}")
            return False
    
    async def _switch_to_local_llm(self, **kwargs) -> bool:
        """Switch to local LLM as fallback."""
        try:
            # This would be implemented by the LLM manager
            self.logger.info("Switching to local LLM as fallback")
            return True
        except Exception:
            return False
    
    async def _use_sample_data(self, **kwargs) -> bool:
        """Use sample data as fallback."""
        try:
            # This would be implemented by the data processor
            self.logger.info("Using sample data as fallback")
            return True
        except Exception:
            return False
    
    async def _enable_offline_mode(self, **kwargs) -> bool:
        """Enable offline mode as fallback."""
        try:
            # This would be implemented by the connection manager
            self.logger.info("Enabling offline mode as fallback")
            return True
        except Exception:
            return False
    
    async def _cleanup_resources(self, **kwargs) -> bool:
        """Cleanup resources as recovery action."""
        try:
            import gc
            import os
            
            # Force garbage collection
            gc.collect()
            
            # Clean up temporary files if temp_dir provided
            temp_dir = kwargs.get('temp_dir')
            if temp_dir and os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    try:
                        os.remove(os.path.join(temp_dir, file))
                    except Exception:
                        pass
            
            self.logger.info("Resources cleaned up successfully")
            return True
        except Exception:
            return False
    
    async def _reset_session(self, **kwargs) -> bool:
        """Reset session as recovery action."""
        try:
            # This would be implemented by the session manager
            self.logger.info("Session reset as recovery action")
            return True
        except Exception:
            return False
    
    async def _use_basic_plots(self, **kwargs) -> bool:
        """Use basic plots as fallback for visualization errors."""
        try:
            # This would be implemented by the visualization manager
            self.logger.info("Using basic plots as fallback")
            return True
        except Exception:
            return False
    
    def is_recoverable_error(self, error: Exception) -> bool:
        """
        Determine if an error is recoverable with fallback options.
        
        Args:
            error: Exception to check
            
        Returns:
            True if error is recoverable
        """
        error_type = self._categorize_error(error)
        severity = self._determine_severity(error, "general")
        
        # Critical configuration errors might not be recoverable
        if error_type == ErrorType.CONFIGURATION_ERROR and severity == ErrorSeverity.CRITICAL:
            return False
        
        # Most other errors have some form of fallback
        return error_type in self._fallback_options
    
    async def get_recovery_plan(self, error: Exception, context: str) -> Dict[str, Any]:
        """
        Generate comprehensive recovery plan for an error.
        
        Args:
            error: Exception that occurred
            context: Context where error occurred
            
        Returns:
            Recovery plan with steps and fallback options
        """
        error_type = self._categorize_error(error)
        severity = self._determine_severity(error, context)
        
        recovery_plan = {
            "error_type": error_type.value,
            "severity": severity.value,
            "recoverable": self.is_recoverable_error(error),
            "immediate_actions": [],
            "fallback_options": self.get_fallback_options(error_type),
            "recovery_suggestions": self._recovery_suggestions.get(error_type, []),
            "estimated_recovery_time": self._estimate_recovery_time(error_type, severity)
        }
        
        # Add immediate actions based on error type
        if error_type == ErrorType.CONNECTION_ERROR:
            recovery_plan["immediate_actions"] = [
                "retry_connection",
                "check_network_status",
                "enable_offline_mode"
            ]
        elif error_type == ErrorType.LLM_ERROR:
            recovery_plan["immediate_actions"] = [
                "validate_api_key",
                "switch_to_local_llm",
                "use_cached_responses"
            ]
        elif error_type == ErrorType.RESOURCE_ERROR:
            recovery_plan["immediate_actions"] = [
                "cleanup_memory",
                "reduce_processing_load",
                "enable_streaming_mode"
            ]
        
        return recovery_plan
    
    def _estimate_recovery_time(self, error_type: ErrorType, severity: ErrorSeverity) -> str:
        """
        Estimate recovery time based on error type and severity.
        
        Args:
            error_type: Type of error
            severity: Error severity
            
        Returns:
            Estimated recovery time as string
        """
        if severity == ErrorSeverity.CRITICAL:
            return "5-10 minutes"
        elif severity == ErrorSeverity.HIGH:
            return "2-5 minutes"
        elif severity == ErrorSeverity.MEDIUM:
            return "30 seconds - 2 minutes"
        else:
            return "Immediate"


# Global error handler instance
_error_handler = ErrorHandler()


def get_error_handler() -> ErrorHandler:
    """
    Get the global error handler instance.
    
    Returns:
        ErrorHandler instance
    """
    return _error_handler


async def handle_error(error: Exception, context: str) -> str:
    """
    Convenience function to handle errors.
    
    Args:
        error: Exception that occurred
        context: Context where error occurred
        
    Returns:
        User-friendly error message
    """
    return await _error_handler.handle_error(error, context)


def log_error(error: Exception, context: str) -> None:
    """
    Convenience function to log errors.
    
    Args:
        error: Exception that occurred
        context: Context where error occurred
    """
    _error_handler.log_error(error, context)