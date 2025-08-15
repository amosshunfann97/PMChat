"""
Main Chainlit Application for Process Mining Integration.

This module implements the core Chainlit event handlers and orchestrates
all manager components to provide a unified web-based process mining interface.

Requirements implemented:
- 1.1: Part selection with searchable dropdown
- 2.1: Automatic DFG visualization display
- 3.1: Ollama LLM (auto-configured)
- 4.1: Context-based query system
- 6.4: Error handling and component coordination
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
        class Message:
            def __init__(self, content: str, actions=None):
                self.content = content
                self.actions = actions or []
            
            async def send(self):
                pass
        
        class Action:
            def __init__(self, name: str, value: str, label: str, payload=None):
                self.name = name
                self.value = value
                self.label = label
                self.payload = payload or {}
        
        class user_session:
            @staticmethod
            def get(key):
                return None
            @staticmethod
            def set(key, value):
                pass
        
        def on_chat_start(func):
            return func
        
        def on_message(func):
            return func
        
        def action_callback(name):
            def decorator(func):
                return func
            return decorator
    
    cl = MockChainlit()

# Import manager components
from chainlit_integration.managers.session_manager import SessionManager
from chainlit_integration.managers.llm_selection_manager import LLMSelectionManager
from chainlit_integration.managers.part_selection_manager import PartSelectionManager
from chainlit_integration.managers.process_mining_engine import ProcessMiningEngine
from chainlit_integration.managers.query_context_manager import QueryContextManager
from chainlit_integration.managers.chat_query_handler import ChatQueryHandler
from chainlit_integration.managers.visualization_manager import VisualizationManager
from chainlit_integration.utils.error_handler import ErrorHandler
from chainlit_integration.utils.chat_history import chat_history
from chainlit_integration.models import SessionState, LLMType, QueryContext

# Import performance and UI enhancements (commented out for now due to missing psutil)
# from chainlit_integration.utils.performance_optimizer import (
#     performance_monitor, resource_manager, global_cache, monitor_performance,
#     optimize_async_operations, get_performance_summary
# )
# from chainlit_integration.utils.progress_indicators import (
#     progress_manager, show_data_processing_progress, show_visualization_progress,
#     show_query_processing_progress, show_retriever_setup_progress
# )
# from chainlit_integration.utils.ui_improvements import (
#     message_formatter, interactive_elements, user_guidance
# )


# Configure logging with debug capabilities
import os
DEBUG_MODE = os.getenv('CHAINLIT_DEBUG', 'false').lower() == 'true'
LOG_LEVEL = logging.DEBUG if DEBUG_MODE else logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('chainlit_app.log') if DEBUG_MODE else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

if DEBUG_MODE:
    logger.info("Debug mode enabled - detailed logging active")


class ChainlitProcessMiningApp:
    """
    Main application class that orchestrates all manager components.
    
    This class implements the core Chainlit integration by coordinating
    between different managers and handling the overall application flow.
    
    Component Coordination:
    - Manages lifecycle of all manager components
    - Handles error propagation between components
    - Coordinates workflow transitions
    - Provides centralized logging and debugging
    """
    
    def __init__(self):
        """Initialize the application with all manager components."""
        self.session_manager: Optional[SessionManager] = None
        self.llm_manager: Optional[LLMSelectionManager] = None
        self.part_manager: Optional[PartSelectionManager] = None
        self.engine: Optional[ProcessMiningEngine] = None
        self.context_manager: Optional[QueryContextManager] = None
        self.query_handler: Optional[ChatQueryHandler] = None
        self.viz_manager: Optional[VisualizationManager] = None
        self.error_handler = ErrorHandler()
        
        # Component coordination state
        self._initialization_complete = False
        self._component_health = {}
        self._debug_mode = DEBUG_MODE
        
        # Performance optimization (commented out for now)
        # optimize_async_operations()
        # resource_manager.start_monitoring()
        
        logger.info(f"ChainlitProcessMiningApp initialized (debug_mode={self._debug_mode})")
    
    async def initialize_managers(self, session_state: SessionState) -> bool:
        """
        Initialize all manager components with session state.
        
        Implements proper component coordination and error propagation.
        
        Args:
            session_state: Current session state
            
        Returns:
            True if all managers initialized successfully
        """
        try:
            logger.info("Starting manager initialization sequence")
            
            # Initialize managers in dependency order with health checks
            import os
            # Check if CSV file exists in current directory, otherwise look in pmchatbot directory
            if os.path.exists("Production_Event_Log.csv"):
                csv_file_path = "Production_Event_Log.csv"
            elif os.path.exists("pmchatbot/Production_Event_Log.csv"):
                csv_file_path = "pmchatbot/Production_Event_Log.csv"
            else:
                csv_file_path = "Production_Event_Log.csv"  # Default fallback
            
            # Initialize each manager with proper parameters
            try:
                logger.debug("Initializing session_manager")
                self.session_manager = SessionManager(session_state)
                self._component_health["session_manager"] = True
                
                logger.debug("Initializing llm_manager")
                self.llm_manager = LLMSelectionManager(session_state)
                self._component_health["llm_manager"] = True
                
                logger.debug("Initializing part_manager")
                self.part_manager = PartSelectionManager(session_state, csv_file_path)
                self._component_health["part_manager"] = True
                
                logger.debug("Initializing engine")
                self.engine = ProcessMiningEngine(session_state)
                self._component_health["engine"] = True
                
                logger.debug("Initializing context_manager")
                self.context_manager = QueryContextManager(session_state)
                self._component_health["context_manager"] = True
                
                logger.debug("Initializing query_handler")
                self.query_handler = ChatQueryHandler(session_state)
                self._component_health["query_handler"] = True
                
                logger.debug("Initializing viz_manager")
                self.viz_manager = VisualizationManager(session_state)
                self._component_health["viz_manager"] = True
                
                # Initialize all managers
                managers_to_init = [
                    ("session_manager", self.session_manager),
                    ("llm_manager", self.llm_manager),
                    ("part_manager", self.part_manager),
                    ("engine", self.engine),
                    ("context_manager", self.context_manager),
                    ("query_handler", self.query_handler),
                    ("viz_manager", self.viz_manager)
                ]
                
                for manager_name, manager_instance in managers_to_init:
                    try:
                        # Initialize if method exists
                        if hasattr(manager_instance, 'initialize'):
                            await manager_instance.initialize()
                        
                        logger.debug(f"{manager_name} initialized successfully")
                        
                    except Exception as e:
                        logger.error(f"Failed to initialize {manager_name}: {str(e)}")
                        self._component_health[manager_name] = False
                        # Propagate critical initialization errors
                        raise Exception(f"Critical component {manager_name} failed to initialize: {str(e)}")
                        
            except Exception as e:
                logger.error(f"Manager initialization failed: {str(e)}")
                raise
            
            # Verify all components are healthy
            unhealthy_components = [name for name, healthy in self._component_health.items() if not healthy]
            if unhealthy_components:
                raise Exception(f"Unhealthy components detected: {unhealthy_components}")
            
            self._initialization_complete = True
            logger.info("All managers initialized successfully")
            return True
            
        except Exception as e:
            error_msg = await self.error_handler.handle_error(e, "manager_initialization")
            logger.error(f"Failed to initialize managers: {error_msg}")
            self._initialization_complete = False
            return False
    
    def get_component_health(self) -> Dict[str, bool]:
        """
        Get health status of all components.
        
        Returns:
            Dictionary mapping component names to health status
        """
        return self._component_health.copy()
    
    def is_ready(self) -> bool:
        """
        Check if the application is ready to handle requests.
        
        Returns:
            True if all components are initialized and healthy
        """
        return (
            self._initialization_complete and
            all(self._component_health.values()) and
            all([
                self.session_manager is not None,
                self.llm_manager is not None,
                self.part_manager is not None,
                self.engine is not None,
                self.context_manager is not None,
                self.query_handler is not None,
                self.viz_manager is not None
            ])
        )
    
    async def handle_component_error(self, component_name: str, error: Exception, context: str) -> str:
        """
        Handle errors from specific components with proper propagation.
        
        Args:
            component_name: Name of the component that failed
            error: Exception that occurred
            context: Context where the error occurred
            
        Returns:
            User-friendly error message
        """
        logger.error(f"Component {component_name} error in {context}: {str(error)}")
        
        # Mark component as unhealthy
        self._component_health[component_name] = False
        
        # Handle specific component failures
        if component_name in ["session_manager", "error_handler"]:
            # Critical components - application cannot continue
            return await self.error_handler.handle_error(
                Exception(f"Critical component {component_name} failed: {str(error)}"),
                f"{component_name}_{context}"
            )
        else:
            # Non-critical components - try to continue with degraded functionality
            return await self.error_handler.handle_error(error, f"{component_name}_{context}")
    
    async def cleanup_all_components(self):
        """
        Cleanup all components in reverse initialization order.
        
        Ensures proper resource cleanup and error handling.
        """
        logger.info("Starting component cleanup sequence")
        
        # Cleanup in reverse order
        cleanup_order = [
            ("viz_manager", "cleanup_temp_files"),
            ("engine", "cleanup_resources"),
            ("query_handler", "cleanup"),
            ("context_manager", "cleanup"),
            ("part_manager", "cleanup"),
            ("llm_manager", "cleanup"),
            ("session_manager", "cleanup")
        ]
        
        for component_name, cleanup_method in cleanup_order:
            try:
                component = getattr(self, component_name, None)
                if component and hasattr(component, cleanup_method):
                    if asyncio.iscoroutinefunction(getattr(component, cleanup_method)):
                        await getattr(component, cleanup_method)()
                    else:
                        getattr(component, cleanup_method)()
                    logger.debug(f"Cleaned up {component_name}")
            except Exception as e:
                logger.warning(f"Error cleaning up {component_name}: {str(e)}")
        
        # Reset component health
        self._component_health.clear()
        self._initialization_complete = False
        logger.info("Component cleanup completed")
    
    def get_debug_info(self) -> Dict[str, Any]:
        """
        Get debug information about the application state.
        
        Returns:
            Dictionary containing debug information
        """
        return {
            "initialization_complete": self._initialization_complete,
            "component_health": self._component_health.copy(),
            "debug_mode": self._debug_mode,
            "is_ready": self.is_ready(),
            "components": {
                name: getattr(self, name, None) is not None
                for name in ["session_manager", "llm_manager", "part_manager", 
                           "engine", "context_manager", "query_handler", "viz_manager"]
            }
        }


# Global app instance
app = ChainlitProcessMiningApp()


@cl.on_settings_update
async def setup_agent(settings):
    """Handle settings updates from the sidebar."""
    session_state: SessionState = cl.user_session.get("session_state")
    app_instance: ChainlitProcessMiningApp = cl.user_session.get("app")
    
    if not session_state or not app_instance:
        return
    
    # Handle chat history actions
    history_action = settings.get("chat_history_action", "none")
    
    if history_action == "view_current":
        current_session = chat_history.get_session_history()
        if current_session:
            message_count = len(current_session.get("messages", []))
            start_time = current_session.get("start_time", "Unknown")
            response = f"""## 📚 Current Session History

**Session Started**: {start_time}
**Total Messages**: {message_count}
**LLM Type**: Ollama (Local)
**Selected Part**: {current_session.get('metadata', {}).get('selected_part', 'None')}

This session is being automatically saved. You can export it using the sidebar options."""
            await cl.Message(content=response).send()
    
    elif history_action == "list_sessions":
        sessions = chat_history.list_sessions()
        if sessions:
            response = "## 📚 Previous Chat Sessions\n\n"
            for i, session in enumerate(sessions[:10]):  # Show last 10 sessions
                response += f"**{i+1}.** {session['start_time'][:19]} - {session['message_count']} messages"
                if session['selected_part']:
                    response += f" - {session['selected_part']}"
                response += "\n"
            
            if len(sessions) > 10:
                response += f"\n... and {len(sessions) - 10} more sessions"
        else:
            response = "## 📚 No Previous Sessions\n\nThis is your first chat session!"
        
        await cl.Message(content=response).send()
    
    elif history_action == "export_current":
        exported = chat_history.export_session(format="txt")
        if exported:
            # Create a file for download
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                f.write(exported)
                temp_path = f.name
            
            await cl.Message(
                content="## 📥 Session Exported\n\nYour current chat session has been exported.",
                elements=[cl.File(name="chat_history.txt", path=temp_path, display="inline")]
            ).send()
        else:
            await cl.Message(content="❌ **Export Failed** - No session data available to export.").send()


@cl.on_chat_start
async def start_session():
    """
    Handle chat session start event.
    
    Initializes session state, managers, and auto-configures Ollama LLM.
    
    Requirements: 1.1, 3.1
    """
    try:
        logger.info("Starting new chat session")
        
        # Initialize session state
        session_state = SessionState()
        cl.user_session.set("session_state", session_state)
        
        # Start chat history tracking
        session_id = cl.user_session.get("id", "unknown")
        chat_history.start_new_session(session_id)
        chat_history.add_message("system", "Session started", {"event": "session_start"})
        
        # Initialize managers with health checking
        success = await app.initialize_managers(session_state)
        if not success:
            health_status = app.get_component_health()
            failed_components = [name for name, healthy in health_status.items() if not healthy]
            
            error_details = f"Failed components: {', '.join(failed_components)}" if failed_components else "Unknown initialization error"
            
            await cl.Message(
                content=f"❌ **System Error**: Failed to initialize the application.\n\n**Details**: {error_details}\n\nPlease refresh and try again."
            ).send()
            return
        
        # Store app instance in session
        cl.user_session.set("app", app)
        
        # Auto-configure Ollama as the default LLM
        ollama_success = await app.llm_manager.handle_ollama_selection()
        
        # Show part selector after successful Ollama configuration
        if ollama_success:
            await app.part_manager.show_part_selector()
        
        # Display welcome message
        welcome_message = """# 🔍 Process Mining Analysis Tool

Welcome! This tool helps you analyze process mining data through an interactive chat interface.

## 🚀 Getting Started

🏠 **Using Ollama (Local AI)** - Your AI model is ready and configured!

You can now start analyzing your process data. The system will automatically load available parts for analysis.

Ready to begin your process mining analysis!"""
        
        await cl.Message(content=welcome_message).send()
        
        logger.info("Chat session started successfully")
        
    except Exception as e:
        error_msg = await app.error_handler.handle_error(e, "session_start")
        await cl.Message(content=error_msg).send()
        logger.error(f"Error starting session: {str(e)}")


@cl.on_message
async def handle_message(message: cl.Message):
    """
    Handle incoming chat messages.
    
    Routes messages based on session state and handles different phases
    of the process mining workflow.
    
    Requirements: 4.1, 4.2, 5.5
    """
    try:
        # Get session data
        session_state: SessionState = cl.user_session.get("session_state")
        app_instance: ChainlitProcessMiningApp = cl.user_session.get("app")
        
        if not session_state or not app_instance:
            await cl.Message(
                content="❌ **Session Error**: Please refresh the page to start a new session."
            ).send()
            return
        
        user_input = message.content.strip()
        logger.info(f"Received message: {user_input[:50]}...")
        
        # Track user message in chat history
        chat_history.add_message("user", user_input, {
            "llm_type": session_state.llm_type.value if session_state.llm_type else None,
            "selected_part": session_state.selected_part
        })
        
        # Handle debug commands if debug mode is enabled
        if app_instance._debug_mode and user_input.startswith('/debug'):
            await handle_debug_command(user_input, session_state, app_instance)
            return
        
        # Check for session termination keywords
        if app_instance.query_handler.check_termination_keywords(user_input):
            await handle_session_termination(session_state, app_instance)
            return
        
        # Route message based on session state
        if not session_state.selected_part:
            # Handle text-based part selection
            if session_state.available_parts:
                # Check if user input matches any available part
                matching_parts = [part for part in session_state.available_parts 
                                if user_input.lower() in part.lower() or part.lower() in user_input.lower()]
                
                if len(matching_parts) == 1:
                    # Exact match found, select this part
                    selected_part = matching_parts[0]
                    success = await app_instance.part_manager.handle_part_selection(selected_part)
                    if success:
                        await cl.Message(
                            content=f"✅ Selected part: **{selected_part}**\n\nProcessing data..."
                        ).send()
                        
                        # Process the selected part
                        processing_result = await app_instance.engine.process_data(selected_part)
                        
                        if processing_result.success:
                            await cl.Message(
                                content="✅ **Processing Complete!** You can now ask questions about your process data."
                            ).send()
                        else:
                            await cl.Message(
                                content=f"❌ **Processing Failed**: {processing_result.error_message}"
                            ).send()
                    return
                elif len(matching_parts) > 1:
                    # Multiple matches, show options
                    parts_list = "\n".join([f"• {part}" for part in matching_parts[:10]])
                    if len(matching_parts) > 10:
                        parts_list += f"\n• ... and {len(matching_parts) - 10} more"
                    
                    await cl.Message(
                        content=f"Found {len(matching_parts)} matching parts:\n\n{parts_list}\n\nPlease type the exact part name you want to select."
                    ).send()
                    return
                else:
                    # No matches, show available parts
                    parts_list = "\n".join([f"• {part}" for part in session_state.available_parts[:10]])
                    if len(session_state.available_parts) > 10:
                        parts_list += f"\n• ... and {len(session_state.available_parts) - 10} more"
                    
                    await cl.Message(
                        content=f"No parts found matching '{user_input}'. Available parts:\n\n{parts_list}\n\nPlease type the exact part name you want to select."
                    ).send()
                    return
            else:
                await cl.Message(
                    content="Please wait while parts are being loaded, or refresh the page if this persists."
                ).send()
            
        elif not session_state.processing_complete:
            await cl.Message(
                content="⏳ Data is still being processed. Please wait for processing to complete."
            ).send()
            
        elif not session_state.visualizations_displayed:
            await cl.Message(
                content="⏳ Visualizations are being generated. Please wait for them to be displayed."
            ).send()
            
        elif session_state.awaiting_context_selection:
            await cl.Message(
                content="Please select a query context first using the buttons above."
            ).send()
            
        else:
            # Handle query - first show context selection
            await app_instance.context_manager.show_context_selector()
            # Store the query for processing after context selection
            cl.user_session.set("pending_query", user_input)
        
    except Exception as e:
        error_msg = await app.error_handler.handle_error(e, "message_handling")
        await cl.Message(content=error_msg).send()
        logger.error(f"Error handling message: {str(e)}")


# Note: show_llm_toggle function removed - only Ollama is supported


# Note: update_llm_toggle_display and track_assistant_message functions removed - only Ollama is supported


async def handle_session_termination(session_state: SessionState, app_instance: ChainlitProcessMiningApp):
    """
    Handle session termination and cleanup.
    
    Uses coordinated component cleanup for proper resource management.
    
    Requirements: 5.5
    """
    try:
        logger.info("Handling session termination")
        
        # Perform coordinated cleanup of all components
        await app_instance.cleanup_all_components()
        
        # Reset session state
        session_state.session_active = False
        session_state.reset_for_new_part()
        
        # Display termination message with component status
        health_status = app_instance.get_component_health()
        cleanup_status = "All components cleaned up successfully" if not health_status else "Some components may still be active"
        
        await cl.Message(
            content=f"""## 👋 Session Ended

Thank you for using the Process Mining Analysis Tool!

**Cleanup Status**: {cleanup_status}

To start a new analysis session, please refresh the page.

**What you can do:**
- Refresh the page to start over
- Close this tab if you're done
- Contact support if you encountered any issues"""
        ).send()
        
        logger.info("Session terminated successfully")
        
    except Exception as e:
        error_msg = await app_instance.handle_component_error("session_manager", e, "termination")
        logger.error(f"Error during session termination: {error_msg}")
        
        # Still try to display termination message even if cleanup failed
        await cl.Message(
            content=f"""## ⚠️ Session Ended with Warnings

The session has been terminated, but some cleanup operations failed.

**Error**: {error_msg}

Please refresh the page to ensure a clean restart."""
        ).send()


async def handle_debug_command(command: str, session_state: SessionState, app_instance: ChainlitProcessMiningApp):
    """
    Handle debug commands for troubleshooting and monitoring.
    
    Available commands:
    - /debug status: Show application status
    - /debug health: Show component health
    - /debug session: Show session state
    - /debug help: Show available debug commands
    """
    try:
        parts = command.split()
        if len(parts) < 2:
            subcommand = "help"
        else:
            subcommand = parts[1].lower()
        
        if subcommand == "status":
            debug_info = app_instance.get_debug_info()
            status_msg = f"""## 🔧 Debug Status

**Application Ready**: {debug_info['is_ready']}
**Initialization Complete**: {debug_info['initialization_complete']}
**Debug Mode**: {debug_info['debug_mode']}

**Components**:
{chr(10).join([f"• {name}: {'✅' if exists else '❌'}" for name, exists in debug_info['components'].items()])}

**Component Health**:
{chr(10).join([f"• {name}: {'✅ Healthy' if healthy else '❌ Unhealthy'}" for name, healthy in debug_info['component_health'].items()])}"""
            
            await cl.Message(content=status_msg).send()
        
        elif subcommand == "health":
            health = app_instance.get_component_health()
            health_msg = f"""## 🏥 Component Health Report

**Overall Status**: {'✅ All Healthy' if all(health.values()) else '⚠️ Issues Detected'}

**Detailed Health**:
{chr(10).join([f"• **{name}**: {'✅ Healthy' if healthy else '❌ Unhealthy'}" for name, healthy in health.items()])}

**Recommendations**:
{chr(10).join([f"• Restart {name}" for name, healthy in health.items() if not healthy]) if not all(health.values()) else '• All components are functioning normally'}"""
            
            await cl.Message(content=health_msg).send()
        
        elif subcommand == "session":
            session_info = f"""## 📊 Session State

**LLM Type**: {session_state.llm_type.value if session_state.llm_type else 'Not selected'}
**Selected Part**: {session_state.selected_part or 'None'}
**Processing Complete**: {'✅' if session_state.processing_complete else '❌'}
**Visualizations Displayed**: {'✅' if session_state.visualizations_displayed else '❌'}
**Ready for Queries**: {'✅' if session_state.is_ready_for_queries() else '❌'}
**Session Active**: {'✅' if session_state.session_active else '❌'}
**Current Context**: {session_state.current_context_mode.value if session_state.current_context_mode else 'None'}
**Awaiting Context Selection**: {'✅' if session_state.awaiting_context_selection else '❌'}

**Available Parts**: {len(session_state.available_parts)} parts loaded
**Retrievers**: {'✅ Configured' if session_state.retrievers else '❌ Not configured'}"""
            
            await cl.Message(content=session_info).send()
        
        elif subcommand == "performance":
            perf_summary = get_performance_summary()
            perf_msg = f"""## ⚡ Performance Dashboard

**Resource Usage**:
• Memory: {perf_summary['resource_usage']['memory_mb']:.1f} MB
• CPU: {perf_summary['resource_usage']['cpu_percent']:.1f}%
• Threads: {perf_summary['resource_usage']['num_threads']}

**Performance Metrics**:
• Total Operations: {perf_summary['performance_metrics']['total_operations']}
• Successful: {perf_summary['performance_metrics']['successful_operations']}
• Failed: {perf_summary['performance_metrics']['failed_operations']}
• Average Duration: {perf_summary['performance_metrics']['average_duration']:.3f}s

**Cache Statistics**:
• Cache Size: {perf_summary['cache_stats']['size']}/{perf_summary['cache_stats']['max_size']}
• Hit Ratio: {perf_summary['cache_stats']['hit_ratio']:.1%}

**Timestamp**: {perf_summary['timestamp']}"""
            
            await cl.Message(content=perf_msg).send()
        
        elif subcommand == "help":
            help_msg = """## 🔧 Debug Commands

Available debug commands:

• `/debug status` - Show overall application status
• `/debug health` - Show component health report  
• `/debug session` - Show current session state
• `/debug performance` - Show performance dashboard
• `/debug help` - Show this help message

**Note**: Debug commands are only available when debug mode is enabled."""
            
            await cl.Message(content=help_msg).send()
        
        else:
            await cl.Message(
                content=f"Unknown debug command: `{subcommand}`. Use `/debug help` for available commands."
            ).send()
    
    except Exception as e:
        logger.error(f"Error handling debug command: {str(e)}")
        await cl.Message(
            content=f"❌ **Debug Command Error**: {str(e)}"
        ).send()


# Note: LLM selection callbacks removed - only Ollama is supported


@cl.action_callback("view_history")
async def on_view_history(action):
    """Handle chat history viewing."""
    try:
        sessions = chat_history.list_sessions()
        current_session = chat_history.get_session_history()
        
        if current_session:
            message_count = len(current_session.get("messages", []))
            start_time = current_session.get("start_time", "Unknown")
            
            response = f"""## 📚 Chat History

### Current Session
**Started**: {start_time}  
**Messages**: {message_count}  
**LLM**: {current_session.get('metadata', {}).get('llm_type', 'Not selected')}  
**Part**: {current_session.get('metadata', {}).get('selected_part', 'None')}

### Previous Sessions
"""
            
            if sessions and len(sessions) > 1:
                for i, session in enumerate(sessions[1:6]):  # Show 5 previous sessions
                    response += f"**{i+1}.** {session['start_time'][:19]} - {session['message_count']} messages\n"
            else:
                response += "No previous sessions found.\n"
            
            response += "\n💡 **Tip**: Your chat history is automatically saved and can be exported."
            
        else:
            response = "## 📚 Chat History\n\nNo chat history available."
        
        await cl.Message(content=response).send()
        
    except Exception as e:
        logger.error(f"Error viewing chat history: {e}")
        await cl.Message(content="❌ **Error**: Unable to load chat history.").send()


# Note: API key submission callback removed - only Ollama is supported


# Part Selection Action Callbacks
@cl.action_callback("select_part")
async def on_select_part(action):
    """Handle direct part selection from dropdown."""
    try:
        session_state: SessionState = cl.user_session.get("session_state")
        app_instance: ChainlitProcessMiningApp = cl.user_session.get("app")
        
        if not session_state or not app_instance:
            return
        
        selected_part = action.value
        logger.info(f"Part selected via dropdown: {selected_part}")
        
        # Handle part selection
        success = await app_instance.part_manager.handle_part_selection(selected_part)
        
        if success:
            # Show processing message
            await cl.Message(
                content=f"⏳ **Processing data for part: {selected_part}**\n\nThis may take a few moments..."
            ).send()
            
            # Process data
            processing_result = await app_instance.engine.process_data(selected_part)
            
            if processing_result.success:
                # Generate and display visualizations
                await app_instance.viz_manager.generate_and_display_automatic_visualizations(
                    processing_result.dfg_data,
                    processing_result.performance_data
                )
                
                # Setup retrievers for queries
                await app_instance.engine.setup_retrievers()
                
                # Show ready message
                await cl.Message(
                    content="""## ✅ **Processing Complete!**

Your data has been processed and visualizations have been generated above.

**You can now ask questions about your process data!**

Simply type your question and I'll help you analyze the process."""
                ).send()
                
            else:
                await cl.Message(
                    content=f"❌ **Processing Failed**: {processing_result.error_message}"
                ).send()
        
    except Exception as e:
        error_msg = await app.error_handler.handle_error(e, "part_selection_dropdown")
        await cl.Message(content=error_msg).send()


@cl.action_callback("browse_letter")
async def on_browse_letter(action):
    """Handle browsing parts by letter."""
    try:
        session_state: SessionState = cl.user_session.get("session_state")
        app_instance: ChainlitProcessMiningApp = cl.user_session.get("app")
        
        if not session_state or not app_instance:
            return
        
        letter = action.value
        await app_instance.part_manager.show_parts_by_letter(letter)
        
    except Exception as e:
        error_msg = await app.error_handler.handle_error(e, "browse_letter")
        await cl.Message(content=error_msg).send()


@cl.action_callback("browse_alphabetically")
async def on_browse_alphabetically(action):
    """Handle alphabetical browsing."""
    try:
        session_state: SessionState = cl.user_session.get("session_state")
        app_instance: ChainlitProcessMiningApp = cl.user_session.get("app")
        
        if not session_state or not app_instance:
            return
        
        await app_instance.part_manager.show_alphabetical_browser()
        
    except Exception as e:
        error_msg = await app.error_handler.handle_error(e, "browse_alphabetically")
        await cl.Message(content=error_msg).send()


@cl.action_callback("search_parts_realtime")
async def on_search_parts_realtime(action):
    """Handle real-time part search."""
    try:
        session_state: SessionState = cl.user_session.get("session_state")
        app_instance: ChainlitProcessMiningApp = cl.user_session.get("app")
        
        if not session_state or not app_instance:
            return
        
        await app_instance.part_manager.show_realtime_search_interface()
        
    except Exception as e:
        error_msg = await app.error_handler.handle_error(e, "search_parts_realtime")
        await cl.Message(content=error_msg).send()


@cl.action_callback("show_all_parts")
async def on_show_all_parts(action):
    """Handle showing all parts."""
    try:
        session_state: SessionState = cl.user_session.get("session_state")
        app_instance: ChainlitProcessMiningApp = cl.user_session.get("app")
        
        if not session_state or not app_instance:
            return
        
        await app_instance.part_manager.show_all_parts()
        
    except Exception as e:
        error_msg = await app.error_handler.handle_error(e, "show_all_parts")
        await cl.Message(content=error_msg).send()


# Part Selection Action Callbacks
@cl.action_callback("part_selected")
# @monitor_performance("part_selection_workflow")  # Commented out for now
async def on_part_selected(action):
    """
    Handle part selection and trigger data processing.
    
    Requirements: 1.1, 1.3, 1.4, 2.1
    """
    try:
        session_state: SessionState = cl.user_session.get("session_state")
        app_instance: ChainlitProcessMiningApp = cl.user_session.get("app")
        
        if not session_state or not app_instance:
            return
        
        selected_part = action.value
        logger.info(f"Part selected: {selected_part}")
        
        # Handle part selection
        success = await app_instance.part_manager.handle_part_selection(selected_part)
        
        if success:
            # Show processing message
            await cl.Message(
                content=f"⏳ **Processing data for part: {selected_part}**\n\nThis may take a few moments..."
            ).send()
            
            # Process data
            processing_result = await app_instance.engine.process_data(selected_part)
            
            if processing_result.success:
                # Generate and display visualizations
                await app_instance.viz_manager.generate_and_display_automatic_visualizations(
                    processing_result.dfg_data,
                    processing_result.performance_data
                )
                
                # Setup retrievers for queries
                await app_instance.engine.setup_retrievers()
                
                # Show ready message
                await cl.Message(
                    content="""## ✅ **Processing Complete!**

Your data has been processed and visualizations have been generated above.

**You can now ask questions about your process data!**

Simply type your question and I'll help you analyze the process."""
                ).send()
                
            else:
                await cl.Message(
                    content=f"❌ **Processing Failed**: {processing_result.error_message}"
                ).send()
        
    except Exception as e:
        error_msg = await app.error_handler.handle_error(e, "part_selection")
        await cl.Message(content=error_msg).send()


@cl.action_callback("switch_part")
async def on_switch_part(action):
    """
    Handle switching to a different part.
    
    Requirements: 5.1, 5.2, 5.3
    """
    try:
        session_state: SessionState = cl.user_session.get("session_state")
        app_instance: ChainlitProcessMiningApp = cl.user_session.get("app")
        
        if not session_state or not app_instance:
            return
        
        logger.info("Part switching requested")
        
        # Reset session for new part
        session_state.reset_for_new_part()
        
        # Show part selection again
        await app_instance.part_manager.show_part_selector()
        
        await cl.Message(
            content="🔄 **Part Selection Reset**\n\nPlease select a new part for analysis."
        ).send()
        
    except Exception as e:
        error_msg = await app.error_handler.handle_error(e, "part_switching")
        await cl.Message(content=error_msg).send()


# Query Context Action Callbacks
@cl.action_callback("context_activity")
async def on_context_activity(action):
    """Handle activity context selection."""
    await handle_context_selection("activity")


@cl.action_callback("context_process")
async def on_context_process(action):
    """Handle process context selection."""
    await handle_context_selection("process")


@cl.action_callback("context_variant")
async def on_context_variant(action):
    """Handle variant context selection."""
    await handle_context_selection("variant")


@cl.action_callback("context_combined")
async def on_context_combined(action):
    """Handle combined context selection."""
    await handle_context_selection("combined")


# @monitor_performance("query_processing")  # Commented out for now
async def handle_context_selection(context_type: str):
    """
    Handle query context selection and process pending query.
    
    Requirements: 4.1, 4.2, 4.3
    """
    try:
        session_state: SessionState = cl.user_session.get("session_state")
        app_instance: ChainlitProcessMiningApp = cl.user_session.get("app")
        pending_query = cl.user_session.get("pending_query")
        
        if not session_state or not app_instance or not pending_query:
            return
        
        logger.info(f"Context selected: {context_type}")
        
        # Set context
        await app_instance.context_manager.handle_context_selection(context_type)
        
        # Process the pending query
        response = await app_instance.query_handler.handle_query(pending_query, context_type)
        
        if response:
            # Display response
            await cl.Message(content=f"""## 🎯 Analysis Result ({context_type.title()} Context)

**Your Question**: {pending_query}

**Answer**: {response}

---

**Ready for another question?** Simply type your next question, and I'll ask you to select a context again.""").send()
        else:
            await cl.Message(content=f"❌ **Query Processing Failed**: Unable to process your query. Please try rephrasing your question.").send()
        
        # Clear pending query and reset context
        cl.user_session.set("pending_query", None)
        session_state.reset_query_context()
        
    except Exception as e:
        error_msg = await app.error_handler.handle_error(e, "query_processing")
        await cl.Message(content=error_msg).send()
        
        # Reset context for next query
        session_state.reset_query_context()
        cl.user_session.set("pending_query", None)
        
        # Show options for next action
        await show_next_action_options()
        
    except Exception as e:
        error_msg = await app.error_handler.handle_error(e, "context_selection")
        await cl.Message(content=error_msg).send()


async def show_next_action_options():
    """Show options for next user action."""
    actions = [
        cl.Action(
            name="switch_part",
            value="switch",
            label="🔄 Switch Part",
            payload={"action": "switch_part"}
        )
    ]
    
    await cl.Message(
        content="**What would you like to do next?**\n\n• Ask another question (just type it)\n• Switch to a different part (use button below)",
        actions=actions
    ).send()


# Export Action Callback
@cl.action_callback("export_data")
async def on_export_data(action):
    """
    Handle data export request.
    
    Requirements: 6.5
    """
    try:
        session_state: SessionState = cl.user_session.get("session_state")
        app_instance: ChainlitProcessMiningApp = cl.user_session.get("app")
        
        if not session_state or not app_instance:
            return
        
        export_type = action.value
        logger.info(f"Export requested: {export_type}")
        
        # Show progress for longer exports
        if export_type in ["json", "images", "summary"]:
            progress_msg = await cl.Message(
                content=f"⏳ **Preparing {export_type.upper()} Export**\n\nPlease wait while we generate your export file..."
            ).send()
        
        # Generate export file
        export_file = await app_instance.viz_manager.export_data(export_type)
        
        if export_file:
            # Update progress message if it exists
            if export_type in ["json", "images", "summary"]:
                await progress_msg.update(
                    content="📁 **Export Complete!**\n\nYour data has been exported successfully."
                )
            
            # Send export file with specific message
            export_messages = {
                "csv": "📊 **CSV Export Complete!**\n\nProcess relationships data exported as CSV file.",
                "json": "📋 **JSON Export Complete!**\n\nProcess data exported as structured JSON file.",
                "summary": "📄 **Summary Export Complete!**\n\nProcess analysis summary exported as text file.",
                "images": "🖼️ **Images Export Complete!**\n\nProcess visualizations exported as ZIP file."
            }
            
            await cl.Message(
                content=export_messages.get(export_type, "📁 **Export Complete!**\n\nYour data has been exported successfully."),
                elements=[export_file]
            ).send()
        else:
            await cl.Message(
                content=f"❌ **Export Failed**: Unable to generate {export_type.upper()} export file. Please ensure process analysis is complete."
            ).send()
        
    except Exception as e:
        error_msg = await app.error_handler.handle_error(e, "data_export")
        await cl.Message(content=error_msg).send()


# Show Export Options Action Callback
@cl.action_callback("show_export_options")
async def on_show_export_options(action):
    """
    Show available export options to the user.
    
    Requirements: 6.5
    """
    try:
        session_state: SessionState = cl.user_session.get("session_state")
        app_instance: ChainlitProcessMiningApp = cl.user_session.get("app")
        
        if not session_state or not app_instance:
            return
        
        logger.info("Export options requested")
        
        # Show export options
        await app_instance.viz_manager.show_export_options()
        
    except Exception as e:
        error_msg = await app.error_handler.handle_error(e, "data_export")
        await cl.Message(content=error_msg).send()


if __name__ == "__main__":
    logger.info("Chainlit Process Mining App starting...")
    # The app will be started by Chainlit's CLI command
         