"""
Chainlit Integration Package

This package provides the integration layer between the existing process mining
chatbot functionality and the Chainlit web interface. It includes managers for
LLM selection, part selection, process mining operations, query handling, and
visualization management.

The package follows a modular architecture with clear interfaces and async/sync
integration patterns to bridge the existing synchronous codebase with Chainlit's
asynchronous requirements.

Main Components:
- models: Data models and session state management
- interfaces: Abstract base classes and contracts
- managers: Component managers for different functionality areas
- handlers: Event and query handlers
- utils: Utility functions and helper classes

Usage:
    from chainlit_integration import SessionManager, get_config_bridge
    from chainlit_integration.models import SessionState, LLMType
    from chainlit_integration.utils import get_error_handler
"""

from .models import (
    SessionState,
    ProcessingResult,
    LLMConfiguration,
    VisualizationData,
    ErrorContext,
    LLMType,
    QueryContext
)

from .interfaces import (
    BaseManager,
    LLMManagerInterface,
    PartSelectionManagerInterface,
    ProcessMiningEngineInterface,
    QueryContextManagerInterface,
    ChatQueryHandlerInterface,
    VisualizationManagerInterface,
    SessionManagerInterface,
    ErrorHandlerInterface
)

from .utils.session_utils import SessionManager
from .utils.config_bridge import get_config_bridge, get_existing_config
from .utils.error_handler import get_error_handler, handle_error, log_error
from .utils.async_helpers import (
    AsyncSyncBridge,
    run_sync_operation,
    get_resource_manager,
    ProgressTracker
)

__version__ = "1.0.0"
__author__ = "Process Mining Chatbot Team"

__all__ = [
    # Models
    "SessionState",
    "ProcessingResult", 
    "LLMConfiguration",
    "VisualizationData",
    "ErrorContext",
    "LLMType",
    "QueryContext",
    
    # Interfaces
    "BaseManager",
    "LLMManagerInterface",
    "PartSelectionManagerInterface", 
    "ProcessMiningEngineInterface",
    "QueryContextManagerInterface",
    "ChatQueryHandlerInterface",
    "VisualizationManagerInterface",
    "SessionManagerInterface",
    "ErrorHandlerInterface",
    
    # Utilities
    "SessionManager",
    "get_config_bridge",
    "get_existing_config",
    "get_error_handler",
    "handle_error",
    "log_error",
    "AsyncSyncBridge",
    "run_sync_operation",
    "get_resource_manager",
    "ProgressTracker"
]