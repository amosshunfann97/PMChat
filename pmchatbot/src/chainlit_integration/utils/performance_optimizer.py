"""
Performance optimization utilities for Chainlit integration.

This module provides performance optimizations including:
- Async operation optimization
- Resource usage monitoring
- Caching mechanisms
- Memory management
- Connection pooling
- Progress indicators
"""

import asyncio
import time
import psutil
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from functools import wraps
import weakref
from concurrent.futures import ThreadPoolExecutor
import threading


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_start: float = 0.0
    memory_end: Optional[float] = None
    memory_delta: Optional[float] = None
    cpu_percent: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.active_operations: Dict[str, PerformanceMetrics] = {}
        self.process = psutil.Process()
    
    def start_operation(self, operation_name: str) -> str:
        """Start monitoring an operation."""
        operation_id = f"{operation_name}_{int(time.time() * 1000)}"
        
        metric = PerformanceMetrics(
            operation_name=operation_name,
            start_time=time.time(),
            memory_start=self.process.memory_info().rss / 1024 / 1024  # MB
        )
        
        self.active_operations[operation_id] = metric
        return operation_id
    
    def end_operation(self, operation_id: str, success: bool = True, error_message: str = None) -> PerformanceMetrics:
        """End monitoring an operation."""
        if operation_id not in self.active_operations:
            logger.warning(f"Operation {operation_id} not found in active operations")
            return None
        
        metric = self.active_operations.pop(operation_id)
        metric.end_time = time.time()
        metric.duration = metric.end_time - metric.start_time
        metric.memory_end = self.process.memory_info().rss / 1024 / 1024  # MB
        metric.memory_delta = metric.memory_end - metric.memory_start
        metric.cpu_percent = self.process.cpu_percent()
        metric.success = success
        metric.error_message = error_message
        
        self.metrics.append(metric)
        return metric
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics:
            return {"total_operations": 0}
        
        successful_ops = [m for m in self.metrics if m.success]
        failed_ops = [m for m in self.metrics if not m.success]
        
        return {
            "total_operations": len(self.metrics),
            "successful_operations": len(successful_ops),
            "failed_operations": len(failed_ops),
            "average_duration": sum(m.duration for m in successful_ops) / len(successful_ops) if successful_ops else 0,
            "total_memory_usage": sum(m.memory_delta for m in successful_ops if m.memory_delta) if successful_ops else 0,
            "slowest_operation": max(successful_ops, key=lambda x: x.duration) if successful_ops else None,
            "fastest_operation": min(successful_ops, key=lambda x: x.duration) if successful_ops else None
        }


# Global performance monitor
performance_monitor = PerformanceMonitor()


def monitor_performance(operation_name: str = None):
    """Decorator to monitor function performance."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            operation_id = performance_monitor.start_operation(op_name)
            
            try:
                result = await func(*args, **kwargs)
                performance_monitor.end_operation(operation_id, success=True)
                return result
            except Exception as e:
                performance_monitor.end_operation(operation_id, success=False, error_message=str(e))
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            operation_id = performance_monitor.start_operation(op_name)
            
            try:
                result = func(*args, **kwargs)
                performance_monitor.end_operation(operation_id, success=True)
                return result
            except Exception as e:
                performance_monitor.end_operation(operation_id, success=False, error_message=str(e))
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


class AsyncCache:
    """Async-safe caching mechanism with TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Any:
        """Get value from cache."""
        async with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            
            # Check TTL
            if time.time() - entry['timestamp'] > entry['ttl']:
                del self._cache[key]
                del self._access_times[key]
                return None
            
            # Update access time
            self._access_times[key] = time.time()
            return entry['value']
    
    async def set(self, key: str, value: Any, ttl: float = None) -> None:
        """Set value in cache."""
        async with self._lock:
            # Evict if cache is full
            if len(self._cache) >= self.max_size and key not in self._cache:
                await self._evict_lru()
            
            self._cache[key] = {
                'value': value,
                'timestamp': time.time(),
                'ttl': ttl or self.default_ttl
            }
            self._access_times[key] = time.time()
    
    async def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._access_times:
            return
        
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        del self._cache[lru_key]
        del self._access_times[lru_key]
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'hit_ratio': 0.0,  # Would need hit/miss tracking for accurate ratio
            'oldest_entry': min(self._access_times.values()) if self._access_times else None,
            'newest_entry': max(self._access_times.values()) if self._access_times else None
        }


class ResourceManager:
    """Manage system resources and prevent resource exhaustion."""
    
    def __init__(self, max_memory_mb: int = 1024, max_cpu_percent: float = 80.0):
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.process = psutil.Process()
        self._monitoring = False
        self._monitor_task = None
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_task = asyncio.create_task(self._monitor_resources())
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
    
    async def _monitor_resources(self):
        """Monitor system resources."""
        while self._monitoring:
            try:
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                cpu_percent = self.process.cpu_percent()
                
                if memory_mb > self.max_memory_mb:
                    logger.warning(f"High memory usage: {memory_mb:.1f}MB (limit: {self.max_memory_mb}MB)")
                    await self._handle_high_memory()
                
                if cpu_percent > self.max_cpu_percent:
                    logger.warning(f"High CPU usage: {cpu_percent:.1f}% (limit: {self.max_cpu_percent}%)")
                    await self._handle_high_cpu()
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring resources: {e}")
                await asyncio.sleep(10)
    
    async def _handle_high_memory(self):
        """Handle high memory usage."""
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear caches if available
        # This would be implemented based on specific cache instances
        logger.info("Performed memory cleanup due to high usage")
    
    async def _handle_high_cpu(self):
        """Handle high CPU usage."""
        # Add small delay to reduce CPU pressure
        await asyncio.sleep(0.1)
        logger.info("Added CPU throttling due to high usage")
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        return {
            'memory_mb': self.process.memory_info().rss / 1024 / 1024,
            'cpu_percent': self.process.cpu_percent(),
            'memory_percent': self.process.memory_percent(),
            'num_threads': self.process.num_threads()
        }


class ProgressIndicator:
    """Async progress indicator for long-running operations."""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        self._callbacks: List[Callable] = []
    
    def add_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add progress callback."""
        self._callbacks.append(callback)
    
    async def update(self, step: int = None, description: str = None):
        """Update progress."""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        if description:
            self.description = description
        
        progress_data = self.get_progress_data()
        
        # Call all callbacks
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(progress_data)
                else:
                    callback(progress_data)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")
    
    def get_progress_data(self) -> Dict[str, Any]:
        """Get current progress data."""
        elapsed_time = time.time() - self.start_time
        progress_percent = (self.current_step / self.total_steps) * 100 if self.total_steps > 0 else 0
        
        # Estimate remaining time
        if self.current_step > 0:
            time_per_step = elapsed_time / self.current_step
            remaining_steps = self.total_steps - self.current_step
            estimated_remaining = time_per_step * remaining_steps
        else:
            estimated_remaining = 0
        
        return {
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'progress_percent': progress_percent,
            'description': self.description,
            'elapsed_time': elapsed_time,
            'estimated_remaining': estimated_remaining
        }


class ConnectionPool:
    """Connection pool for database and external service connections."""
    
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self._connections: List[Any] = []
        self._available: asyncio.Queue = asyncio.Queue(maxsize=max_connections)
        self._lock = asyncio.Lock()
        self._closed = False
    
    async def get_connection(self) -> Any:
        """Get a connection from the pool."""
        if self._closed:
            raise RuntimeError("Connection pool is closed")
        
        try:
            # Try to get an available connection
            connection = await asyncio.wait_for(self._available.get(), timeout=5.0)
            return connection
        except asyncio.TimeoutError:
            # Create new connection if pool not full
            async with self._lock:
                if len(self._connections) < self.max_connections:
                    connection = await self._create_connection()
                    self._connections.append(connection)
                    return connection
                else:
                    raise RuntimeError("Connection pool exhausted")
    
    async def return_connection(self, connection: Any):
        """Return a connection to the pool."""
        if not self._closed and connection in self._connections:
            await self._available.put(connection)
    
    async def _create_connection(self) -> Any:
        """Create a new connection. Override in subclasses."""
        # This is a placeholder - implement specific connection creation
        return object()
    
    async def close_all(self):
        """Close all connections in the pool."""
        self._closed = True
        
        # Close all connections
        for connection in self._connections:
            try:
                if hasattr(connection, 'close'):
                    if asyncio.iscoroutinefunction(connection.close):
                        await connection.close()
                    else:
                        connection.close()
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
        
        self._connections.clear()


class BatchProcessor:
    """Process items in batches to optimize performance."""
    
    def __init__(self, batch_size: int = 100, max_concurrent: int = 5):
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_items(self, items: List[Any], processor: Callable) -> List[Any]:
        """Process items in batches."""
        results = []
        
        # Split items into batches
        batches = [items[i:i + self.batch_size] for i in range(0, len(items), self.batch_size)]
        
        # Process batches concurrently
        tasks = []
        for batch in batches:
            task = asyncio.create_task(self._process_batch(batch, processor))
            tasks.append(task)
        
        # Wait for all batches to complete
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                logger.error(f"Batch processing error: {batch_result}")
                continue
            results.extend(batch_result)
        
        return results
    
    async def _process_batch(self, batch: List[Any], processor: Callable) -> List[Any]:
        """Process a single batch."""
        async with self._semaphore:
            results = []
            for item in batch:
                try:
                    if asyncio.iscoroutinefunction(processor):
                        result = await processor(item)
                    else:
                        result = processor(item)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing item: {e}")
                    results.append(None)
            return results


# Global instances
resource_manager = ResourceManager()
global_cache = AsyncCache()


def optimize_async_operations():
    """Apply global async operation optimizations."""
    # Set event loop policy for better performance on Windows
    if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Start resource monitoring
    resource_manager.start_monitoring()
    
    logger.info("Async operation optimizations applied")


def get_performance_summary() -> Dict[str, Any]:
    """Get comprehensive performance summary."""
    return {
        'performance_metrics': performance_monitor.get_summary(),
        'resource_usage': resource_manager.get_current_usage(),
        'cache_stats': global_cache.get_stats(),
        'timestamp': time.time()
    }


# Cleanup function
async def cleanup_performance_resources():
    """Cleanup performance monitoring resources."""
    resource_manager.stop_monitoring()
    await global_cache.clear()
    logger.info("Performance resources cleaned up")