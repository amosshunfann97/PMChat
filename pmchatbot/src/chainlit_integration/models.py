"""
Data models and session state classes for Chainlit integration.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum


class LLMType(Enum):
    """Enumeration for LLM types."""
    OPENAI = "openai"
    OLLAMA = "ollama"


class QueryContext(Enum):
    """Enumeration for query context types."""
    ACTIVITY = "activity"
    PROCESS = "process"
    VARIANT = "variant"
    COMBINED = "combined"


@dataclass
class SessionState:
    """
    Session state model to track user session data across interactions.
    
    This class maintains all the necessary state information for a user's
    process mining analysis session, including LLM configuration, selected
    parts, processing status, and query context.
    """
    # LLM Configuration
    llm_type: Optional[LLMType] = None
    openai_api_key: Optional[str] = None
    
    # Part Selection
    selected_part: Optional[str] = None
    available_parts: List[str] = field(default_factory=list)
    
    # Processing Status
    processing_complete: bool = False
    data_loaded: bool = False
    
    # GraphRAG Components
    retrievers: Optional[Tuple[Any, Any, Any]] = None  # (activity, process, variant)
    
    # Visualization Status
    visualizations_displayed: bool = False
    dfg_images: Optional[Tuple[Any, Any]] = None  # (frequency_dfg, performance_dfg)
    
    # Query Management
    awaiting_context_selection: bool = False
    current_context_mode: Optional[QueryContext] = None
    
    # Session Control
    session_active: bool = True
    
    def reset_for_new_part(self) -> None:
        """Reset state when switching to a new part selection."""
        self.selected_part = None
        self.processing_complete = False
        self.retrievers = None
        self.visualizations_displayed = False
        self.dfg_images = None
        self.awaiting_context_selection = False
        self.current_context_mode = None
    
    def reset_query_context(self) -> None:
        """Reset query context after each response."""
        self.awaiting_context_selection = False
        self.current_context_mode = None
    
    def is_ready_for_queries(self) -> bool:
        """Check if session is ready to handle queries."""
        return (
            self.llm_type is not None and
            self.selected_part is not None and
            self.processing_complete and
            self.retrievers is not None
        )


@dataclass
class ProcessingResult:
    """
    Result model for process mining data processing operations.
    
    Contains the results of PM4py pipeline execution including DFG data,
    performance metrics, and chunk counts for storage verification.
    """
    success: bool
    dfg_data: Optional[Dict] = None
    performance_data: Optional[Dict] = None
    start_activities: Optional[List[str]] = None
    end_activities: Optional[List[str]] = None
    chunk_counts: Dict[str, int] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    def get_total_chunks(self) -> int:
        """Get total number of chunks processed."""
        return sum(self.chunk_counts.values())


@dataclass
class LLMConfiguration:
    """
    Configuration model for LLM settings.
    
    Encapsulates all LLM-related configuration including API keys,
    model names, and connection parameters.
    """
    llm_type: LLMType
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    
    def is_valid(self) -> bool:
        """Validate LLM configuration."""
        if self.llm_type == LLMType.OPENAI:
            return self.api_key is not None and len(self.api_key.strip()) > 0
        elif self.llm_type == LLMType.OLLAMA:
            return self.base_url is not None
        return False


@dataclass
class VisualizationData:
    """
    Data model for visualization information.
    
    Contains paths and metadata for generated visualizations.
    """
    frequency_dfg_path: Optional[str] = None
    performance_dfg_path: Optional[str] = None
    export_csv_path: Optional[str] = None
    temp_files: List[str] = field(default_factory=list)
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary visualization files."""
        import os
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass  # Ignore cleanup errors
        self.temp_files.clear()


@dataclass
class ErrorContext:
    """
    Context information for error handling.
    
    Provides structured error information with context for better
    error messages and recovery suggestions.
    """
    error_type: str
    component: str
    operation: str
    user_message: str
    technical_details: Optional[str] = None
    recovery_suggestions: List[str] = field(default_factory=list)
    
    def format_user_message(self) -> str:
        """Format error message for user display."""
        message = f"❌ **{self.error_type}**: {self.user_message}"
        
        if self.recovery_suggestions:
            message += "\n\n**Suggestions:**"
            for suggestion in self.recovery_suggestions:
                message += f"\n• {suggestion}"
        
        return message