"""
Async/sync integration utilities for Chainlit integration.

This module provides utilities to bridge synchronous operations from the existing
codebase with the asynchronous requirements of Chainlit.
"""

import asyncio
import functools
from typing import Any, Callable, Coroutine, TypeVar, Union
from concurrent.futures import ThreadPoolExecutor
import threading

T = TypeVar('T')


class AsyncSyncBridge:
    """
    Bridge class to handle async/sync integration patterns.
    
    Provides utilities to run synchronous operations in async context
    and manage thread pools for CPU-intensive operations.
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize the async/sync bridge.
        
        Args:
            max_workers: Maximum number of worker threads for sync operations
        """
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
    
    async def run_sync(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Run a synchronous function in an async context.
        
        Args:
            func: Synchronous function to run
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the synchronous function
        """
        loop = asyncio.get_event_loop()
        partial_func = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(self._executor, partial_func)
    
    async def run_sync_with_progress(
        self, 
        func: Callable[..., T], 
        progress_callback: Callable[[str], Coroutine[Any, Any, None]] = None,
        *args, 
        **kwargs
    ) -> T:
        """
        Run a synchronous function with progress updates.
        
        Args:
            func: Synchronous function to run
            progress_callback: Async callback for progress updates
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the synchronous function
        """
        if progress_callback:
            await progress_callback("Starting operation...")
        
        result = await self.run_sync(func, *args, **kwargs)
        
        if progress_callback:
            await progress_callback("Operation completed")
        
        return result
    
    def cleanup(self) -> None:
        """Clean up the thread pool executor."""
        self._executor.shutdown(wait=True)


def async_wrapper(func: Callable[..., T]) -> Callable[..., Coroutine[Any, Any, T]]:
    """
    Decorator to wrap synchronous functions for async execution.
    
    Args:
        func: Synchronous function to wrap
        
    Returns:
        Async wrapper function
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        loop = asyncio.get_event_loop()
        partial_func = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(None, partial_func)
    
    return wrapper


def sync_to_async(func: Callable[..., T]) -> Callable[..., Coroutine[Any, Any, T]]:
    """
    Convert a synchronous function to an async function.
    
    Args:
        func: Synchronous function to convert
        
    Returns:
        Async version of the function
    """
    return async_wrapper(func)


class ProgressTracker:
    """
    Progress tracking utility for long-running operations.
    
    Provides a way to track and report progress during async operations.
    """
    
    def __init__(self, total_steps: int = 100):
        """
        Initialize progress tracker.
        
        Args:
            total_steps: Total number of steps for the operation
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.callbacks = []
    
    def add_callback(self, callback: Callable[[int, int, str], Coroutine[Any, Any, None]]) -> None:
        """
        Add a progress callback.
        
        Args:
            callback: Async callback function (current_step, total_steps, message)
        """
        self.callbacks.append(callback)
    
    async def update(self, step: int, message: str = "") -> None:
        """
        Update progress and notify callbacks.
        
        Args:
            step: Current step number
            message: Progress message
        """
        self.current_step = step
        
        for callback in self.callbacks:
            try:
                await callback(self.current_step, self.total_steps, message)
            except Exception:
                pass  # Ignore callback errors
    
    async def increment(self, message: str = "") -> None:
        """
        Increment progress by one step.
        
        Args:
            message: Progress message
        """
        await self.update(self.current_step + 1, message)
    
    def get_percentage(self) -> float:
        """Get current progress as percentage."""
        if self.total_steps == 0:
            return 100.0
        return (self.current_step / self.total_steps) * 100.0


class ResourceManager:
    """
    Resource management utility for async operations.
    
    Manages resources like GPU memory, file handles, and database connections
    in an async context.
    """
    
    def __init__(self):
        """Initialize resource manager."""
        self._resources = {}
        self._lock = asyncio.Lock()
    
    async def acquire_resource(self, resource_id: str, factory: Callable[[], Any]) -> Any:
        """
        Acquire a resource, creating it if necessary.
        
        Args:
            resource_id: Unique identifier for the resource
            factory: Function to create the resource if it doesn't exist
            
        Returns:
            The requested resource
        """
        async with self._lock:
            if resource_id not in self._resources:
                self._resources[resource_id] = factory()
            return self._resources[resource_id]
    
    async def release_resource(self, resource_id: str, cleanup_func: Callable[[Any], None] = None) -> None:
        """
        Release a resource and optionally clean it up.
        
        Args:
            resource_id: Unique identifier for the resource
            cleanup_func: Optional cleanup function for the resource
        """
        async with self._lock:
            if resource_id in self._resources:
                resource = self._resources.pop(resource_id)
                if cleanup_func:
                    cleanup_func(resource)
    
    async def cleanup_all(self) -> None:
        """Clean up all managed resources."""
        async with self._lock:
            self._resources.clear()


# Global instances for convenience
_bridge = AsyncSyncBridge()
_resource_manager = ResourceManager()


async def run_sync_operation(func: Callable[..., T], *args, **kwargs) -> T:
    """
    Convenience function to run synchronous operations asynchronously.
    
    Args:
        func: Synchronous function to run
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Result of the synchronous function
    """
    return await _bridge.run_sync(func, *args, **kwargs)


def get_resource_manager() -> ResourceManager:
    """Get the global resource manager instance."""
    return _resource_manager


async def run_in_executor(func: Callable[..., T], *args, **kwargs) -> T:
    """
    Run a function in a thread pool executor.
    
    Args:
        func: Function to run in executor
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Result of the function
    """
    loop = asyncio.get_event_loop()
    partial_func = functools.partial(func, *args, **kwargs)
    return await loop.run_in_executor(None, partial_func)


def cleanup_async_resources() -> None:
    """Clean up global async resources."""
    _bridge.cleanup()