"""
Progress indicators and loading states for Chainlit integration.

This module provides user-friendly progress indicators for long-running operations
including data processing, visualization generation, and query processing.

Features:
- Real-time progress updates
- Estimated completion times
- Visual progress bars
- Status messages
- Error handling with progress context
"""

import asyncio
import time
import logging
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

# Conditional import for Chainlit
try:
    import chainlit as cl
except ImportError:
    # Mock for testing
    class MockChainlit:
        class Message:
            def __init__(self, content: str):
                self.content = content
            async def send(self): pass
            async def update(self, content: str): pass
    cl = MockChainlit()


logger = logging.getLogger(__name__)


class ProgressState(Enum):
    """Progress indicator states."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressStep:
    """Individual progress step."""
    name: str
    description: str
    weight: float = 1.0  # Relative weight for progress calculation
    completed: bool = False
    error: Optional[str] = None


class ChainlitProgressIndicator:
    """Progress indicator specifically designed for Chainlit interface."""
    
    def __init__(self, title: str, steps: List[ProgressStep] = None):
        self.title = title
        self.steps = steps or []
        self.current_step_index = 0
        self.state = ProgressState.INITIALIZING
        self.start_time = time.time()
        self.message: Optional[cl.Message] = None
        self.total_weight = sum(step.weight for step in self.steps)
        self._update_callbacks: List[Callable] = []
    
    def add_step(self, name: str, description: str, weight: float = 1.0):
        """Add a progress step."""
        step = ProgressStep(name=name, description=description, weight=weight)
        self.steps.append(step)
        self.total_weight += weight
    
    def add_update_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for progress updates."""
        self._update_callbacks.append(callback)
    
    async def start(self):
        """Start the progress indicator."""
        self.state = ProgressState.RUNNING
        self.start_time = time.time()
        
        content = self._generate_progress_content()
        self.message = cl.Message(content=content)
        await self.message.send()
        
        logger.info(f"Progress indicator started: {self.title}")
    
    async def update_step(self, step_name: str, completed: bool = True, error: str = None):
        """Update a specific step."""
        for i, step in enumerate(self.steps):
            if step.name == step_name:
                step.completed = completed
                step.error = error
                
                if completed and not error:
                    self.current_step_index = max(self.current_step_index, i + 1)
                
                await self._update_display()
                break
        else:
            logger.warning(f"Step '{step_name}' not found in progress steps")
    
    async def next_step(self, description: str = None):
        """Move to the next step."""
        if self.current_step_index < len(self.steps):
            current_step = self.steps[self.current_step_index]
            current_step.completed = True
            
            if description:
                current_step.description = description
            
            self.current_step_index += 1
            await self._update_display()
    
    async def complete(self, success_message: str = None):
        """Complete the progress indicator."""
        self.state = ProgressState.COMPLETED
        
        # Mark all steps as completed
        for step in self.steps:
            if not step.error:
                step.completed = True
        
        if success_message:
            await self._update_display(success_message)
        else:
            await self._update_display()
        
        logger.info(f"Progress indicator completed: {self.title}")
    
    async def fail(self, error_message: str):
        """Mark progress as failed."""
        self.state = ProgressState.FAILED
        
        # Mark current step as failed
        if self.current_step_index < len(self.steps):
            self.steps[self.current_step_index].error = error_message
        
        await self._update_display()
        logger.error(f"Progress indicator failed: {self.title} - {error_message}")
    
    async def _update_display(self, custom_message: str = None):
        """Update the progress display."""
        content = custom_message or self._generate_progress_content()
        
        if self.message:
            await self.message.update(content)
        
        # Call update callbacks
        progress_data = self._get_progress_data()
        for callback in self._update_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(progress_data)
                else:
                    callback(progress_data)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")
    
    def _generate_progress_content(self) -> str:
        """Generate progress content for display."""
        if self.state == ProgressState.COMPLETED:
            return self._generate_completed_content()
        elif self.state == ProgressState.FAILED:
            return self._generate_failed_content()
        else:
            return self._generate_running_content()
    
    def _generate_running_content(self) -> str:
        """Generate content for running state."""
        progress_percent = self._calculate_progress_percent()
        elapsed_time = time.time() - self.start_time
        estimated_total = self._estimate_total_time()
        
        # Progress bar
        bar_length = 20
        filled_length = int(bar_length * progress_percent / 100)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        content = f"""## ⏳ {self.title}

**Progress**: {progress_percent:.1f}% `{bar}` 

**Elapsed Time**: {self._format_time(elapsed_time)}
**Estimated Total**: {self._format_time(estimated_total)}

**Steps**:"""
        
        # Add step details
        for i, step in enumerate(self.steps):
            if step.completed:
                status = "✅"
            elif step.error:
                status = "❌"
            elif i == self.current_step_index:
                status = "⏳"
            else:
                status = "⏸️"
            
            content += f"\n{status} **{step.name}**: {step.description}"
            
            if step.error:
                content += f"\n   ❌ Error: {step.error}"
        
        return content
    
    def _generate_completed_content(self) -> str:
        """Generate content for completed state."""
        elapsed_time = time.time() - self.start_time
        
        content = f"""## ✅ {self.title} - Complete!

**Total Time**: {self._format_time(elapsed_time)}

**Completed Steps**:"""
        
        for step in self.steps:
            if step.completed and not step.error:
                content += f"\n✅ **{step.name}**: {step.description}"
            elif step.error:
                content += f"\n⚠️ **{step.name}**: {step.description} (with warnings)"
        
        return content
    
    def _generate_failed_content(self) -> str:
        """Generate content for failed state."""
        elapsed_time = time.time() - self.start_time
        
        content = f"""## ❌ {self.title} - Failed

**Time Before Failure**: {self._format_time(elapsed_time)}

**Step Status**:"""
        
        for step in self.steps:
            if step.completed:
                status = "✅"
            elif step.error:
                status = "❌"
            else:
                status = "⏸️"
            
            content += f"\n{status} **{step.name}**: {step.description}"
            
            if step.error:
                content += f"\n   ❌ Error: {step.error}"
        
        return content
    
    def _calculate_progress_percent(self) -> float:
        """Calculate current progress percentage."""
        if not self.steps:
            return 0.0
        
        completed_weight = sum(step.weight for step in self.steps if step.completed)
        return (completed_weight / self.total_weight) * 100 if self.total_weight > 0 else 0.0
    
    def _estimate_total_time(self) -> float:
        """Estimate total completion time."""
        if self.current_step_index == 0:
            return 0.0
        
        elapsed_time = time.time() - self.start_time
        progress_percent = self._calculate_progress_percent()
        
        if progress_percent > 0:
            return elapsed_time * (100 / progress_percent)
        else:
            return 0.0
    
    def _format_time(self, seconds: float) -> str:
        """Format time duration."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def _get_progress_data(self) -> Dict[str, Any]:
        """Get progress data for callbacks."""
        return {
            'title': self.title,
            'state': self.state.value,
            'progress_percent': self._calculate_progress_percent(),
            'current_step_index': self.current_step_index,
            'total_steps': len(self.steps),
            'elapsed_time': time.time() - self.start_time,
            'estimated_total_time': self._estimate_total_time(),
            'steps': [
                {
                    'name': step.name,
                    'description': step.description,
                    'completed': step.completed,
                    'error': step.error
                }
                for step in self.steps
            ]
        }


class LoadingSpinner:
    """Simple loading spinner for quick operations."""
    
    def __init__(self, message: str = "Loading..."):
        self.message = message
        self.spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.current_char = 0
        self.is_spinning = False
        self.spinner_task = None
        self.display_message: Optional[cl.Message] = None
    
    async def start(self):
        """Start the loading spinner."""
        self.is_spinning = True
        self.display_message = cl.Message(content=f"{self.spinner_chars[0]} {self.message}")
        await self.display_message.send()
        
        self.spinner_task = asyncio.create_task(self._spin())
    
    async def stop(self, final_message: str = None):
        """Stop the loading spinner."""
        self.is_spinning = False
        
        if self.spinner_task:
            self.spinner_task.cancel()
        
        if self.display_message and final_message:
            await self.display_message.update(final_message)
    
    async def _spin(self):
        """Animate the spinner."""
        while self.is_spinning:
            try:
                self.current_char = (self.current_char + 1) % len(self.spinner_chars)
                spinner_content = f"{self.spinner_chars[self.current_char]} {self.message}"
                
                if self.display_message:
                    await self.display_message.update(spinner_content)
                
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in spinner animation: {e}")
                break


class ProgressManager:
    """Manage multiple progress indicators."""
    
    def __init__(self):
        self.active_indicators: Dict[str, ChainlitProgressIndicator] = {}
        self.completed_indicators: List[ChainlitProgressIndicator] = []
    
    def create_indicator(self, identifier: str, title: str, steps: List[ProgressStep] = None) -> ChainlitProgressIndicator:
        """Create a new progress indicator."""
        indicator = ChainlitProgressIndicator(title, steps)
        self.active_indicators[identifier] = indicator
        return indicator
    
    async def start_indicator(self, identifier: str):
        """Start a progress indicator."""
        if identifier in self.active_indicators:
            await self.active_indicators[identifier].start()
    
    async def complete_indicator(self, identifier: str, success_message: str = None):
        """Complete a progress indicator."""
        if identifier in self.active_indicators:
            indicator = self.active_indicators.pop(identifier)
            await indicator.complete(success_message)
            self.completed_indicators.append(indicator)
    
    async def fail_indicator(self, identifier: str, error_message: str):
        """Fail a progress indicator."""
        if identifier in self.active_indicators:
            indicator = self.active_indicators.pop(identifier)
            await indicator.fail(error_message)
            self.completed_indicators.append(indicator)
    
    def get_indicator(self, identifier: str) -> Optional[ChainlitProgressIndicator]:
        """Get a progress indicator by identifier."""
        return self.active_indicators.get(identifier)
    
    def get_active_count(self) -> int:
        """Get count of active indicators."""
        return len(self.active_indicators)
    
    async def cleanup_completed(self, max_completed: int = 10):
        """Cleanup old completed indicators."""
        if len(self.completed_indicators) > max_completed:
            self.completed_indicators = self.completed_indicators[-max_completed:]


# Global progress manager
progress_manager = ProgressManager()


# Convenience functions for common progress patterns

async def show_data_processing_progress(part_name: str) -> ChainlitProgressIndicator:
    """Show progress for data processing operations."""
    steps = [
        ProgressStep("load_data", f"Loading data for {part_name}", 1.0),
        ProgressStep("filter_data", "Filtering and preparing data", 1.0),
        ProgressStep("discover_dfg", "Discovering process model", 2.0),
        ProgressStep("discover_performance", "Analyzing performance metrics", 2.0),
        ProgressStep("store_data", "Storing processed data", 1.0)
    ]
    
    indicator = progress_manager.create_indicator(
        f"data_processing_{part_name}",
        f"Processing Data for {part_name}",
        steps
    )
    
    await progress_manager.start_indicator(f"data_processing_{part_name}")
    return indicator


async def show_visualization_progress() -> ChainlitProgressIndicator:
    """Show progress for visualization generation."""
    steps = [
        ProgressStep("generate_frequency_dfg", "Generating frequency DFG", 2.0),
        ProgressStep("generate_performance_dfg", "Generating performance DFG", 2.0),
        ProgressStep("optimize_layout", "Optimizing visualization layout", 1.0),
        ProgressStep("save_images", "Saving visualization images", 1.0),
        ProgressStep("display", "Displaying visualizations", 1.0)
    ]
    
    indicator = progress_manager.create_indicator(
        "visualization_generation",
        "Generating Visualizations",
        steps
    )
    
    await progress_manager.start_indicator("visualization_generation")
    return indicator


async def show_query_processing_progress(query: str) -> LoadingSpinner:
    """Show progress for query processing."""
    spinner = LoadingSpinner(f"Processing query: {query[:50]}...")
    await spinner.start()
    return spinner


async def show_retriever_setup_progress() -> ChainlitProgressIndicator:
    """Show progress for retriever setup."""
    steps = [
        ProgressStep("chunk_data", "Chunking processed data", 1.0),
        ProgressStep("create_embeddings", "Creating embeddings", 2.0),
        ProgressStep("setup_activity_retriever", "Setting up activity retriever", 1.0),
        ProgressStep("setup_process_retriever", "Setting up process retriever", 1.0),
        ProgressStep("setup_variant_retriever", "Setting up variant retriever", 1.0)
    ]
    
    indicator = progress_manager.create_indicator(
        "retriever_setup",
        "Setting up Query Retrievers",
        steps
    )
    
    await progress_manager.start_indicator("retriever_setup")
    return indicator


# Context manager for automatic progress handling
class ProgressContext:
    """Context manager for automatic progress indicator handling."""
    
    def __init__(self, identifier: str, title: str, steps: List[ProgressStep] = None):
        self.identifier = identifier
        self.title = title
        self.steps = steps
        self.indicator = None
    
    async def __aenter__(self) -> ChainlitProgressIndicator:
        self.indicator = progress_manager.create_indicator(self.identifier, self.title, self.steps)
        await progress_manager.start_indicator(self.identifier)
        return self.indicator
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            await progress_manager.complete_indicator(self.identifier)
        else:
            error_message = str(exc_val) if exc_val else "Unknown error occurred"
            await progress_manager.fail_indicator(self.identifier, error_message)


# Usage example:
# async with ProgressContext("my_operation", "My Operation", steps) as progress:
#     await progress.update_step("step1", completed=True)
#     # ... do work ...
#     await progress.next_step("Updated description")