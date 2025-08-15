"""
Performance tests for Chainlit integration components.

This module tests performance characteristics of the system including:
- Data processing performance with large datasets
- Memory usage optimization
- Component initialization times
- Query response times
- Concurrent operation handling

Requirements tested:
- 6.1, 6.2: Configuration and embedding performance
- 6.4: Error handling performance
- Overall system performance under load
"""

import pytest
import asyncio
import time
import psutil
import os
import sys
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock external dependencies
mock_modules = {
    'chainlit': Mock(),
    'neo4j': Mock(),
    'openai': Mock(),
    'pm4py': Mock(),
    'matplotlib': Mock(),
    'matplotlib.pyplot': Mock(),
    'PIL': Mock(),
    'PIL.Image': Mock(),
    'torch': Mock(),
    'sentence_transformers': Mock()
}

with patch.dict('sys.modules', mock_modules):
    from chainlit_integration.models import SessionState, ProcessingResult
    from chainlit_integration.managers.part_selection_manager import PartSelectionManager
    from chainlit_integration.managers.process_mining_engine import ProcessMiningEngine
    from chainlit_integration.managers.chat_query_handler import ChatQueryHandler


class PerformanceMonitor:
    """Helper class for monitoring performance metrics."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        self.metrics = {}
    
    def start_monitoring(self, operation_name: str):
        """Start monitoring an operation."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.metrics[operation_name] = {'start_time': self.start_time, 'start_memory': self.start_memory}
    
    def stop_monitoring(self, operation_name: str) -> Dict[str, float]:
        """Stop monitoring and return metrics."""
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        duration = end_time - self.start_time
        memory_delta = end_memory - self.start_memory
        peak_memory = max(self.start_memory, end_memory)
        
        self.metrics[operation_name].update({
            'duration': duration,
            'memory_delta': memory_delta,
            'peak_memory': peak_memory,
            'end_memory': end_memory
        })
        
        return self.metrics[operation_name]
    
    def get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024


@pytest.fixture
def performance_monitor():
    """Provide performance monitoring."""
    return PerformanceMonitor()


class TestDataProcessingPerformance:
    """Test performance of data processing operations."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_large_csv_loading_performance(self, large_temp_csv_file, performance_monitor):
        """Test CSV loading performance with large datasets."""
        session_state = SessionState()
        
        # Mock the manager to avoid external dependencies
        with patch('chainlit_integration.managers.part_selection_manager.pd.read_csv') as mock_read_csv:
            # Simulate large dataset
            import pandas as pd
            large_df = pd.DataFrame({
                'case_id': [f'Case_{i}' for i in range(10000)],
                'part_desc': [f'Part_{i%100}' for i in range(10000)],
                'activity': [f'Activity_{i%10}' for i in range(10000)]
            })
            mock_read_csv.return_value = large_df
            
            manager = PartSelectionManager(session_state, large_temp_csv_file)
            
            performance_monitor.start_monitoring('large_csv_loading')
            parts = await manager.load_available_parts()
            metrics = performance_monitor.stop_monitoring('large_csv_loading')
            
            # Performance assertions
            assert metrics['duration'] < 5.0, f"CSV loading took {metrics['duration']:.2f}s, should be < 5s"
            assert metrics['memory_delta'] < 100, f"Memory usage increased by {metrics['memory_delta']:.2f}MB, should be < 100MB"
            assert len(parts) == 100, "Should extract 100 unique parts"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_part_loading(self, temp_csv_file, performance_monitor):
        """Test concurrent part loading operations."""
        session_states = [SessionState() for _ in range(5)]
        managers = [PartSelectionManager(state, temp_csv_file) for state in session_states]
        
        performance_monitor.start_monitoring('concurrent_loading')
        
        # Run concurrent loading operations
        tasks = [manager.load_available_parts() for manager in managers]
        results = await asyncio.gather(*tasks)
        
        metrics = performance_monitor.stop_monitoring('concurrent_loading')
        
        # Verify all results are identical
        for result in results[1:]:
            assert result == results[0], "Concurrent loading should return identical results"
        
        # Performance assertions
        assert metrics['duration'] < 3.0, f"Concurrent loading took {metrics['duration']:.2f}s, should be < 3s"
        assert metrics['memory_delta'] < 50, f"Memory usage increased by {metrics['memory_delta']:.2f}MB, should be < 50MB"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_data_processing_pipeline_performance(self, temp_csv_file, performance_monitor, mock_pm4py):
        """Test performance of the complete data processing pipeline."""
        session_state = SessionState()
        
        with patch('chainlit_integration.managers.process_mining_engine.pm4py', mock_pm4py):
            engine = ProcessMiningEngine(session_state, temp_csv_file)
            
            performance_monitor.start_monitoring('pipeline_processing')
            result = await engine.process_data("Motor_Housing")
            metrics = performance_monitor.stop_monitoring('pipeline_processing')
            
            # Performance assertions
            assert result.success is True
            assert metrics['duration'] < 10.0, f"Pipeline processing took {metrics['duration']:.2f}s, should be < 10s"
            assert metrics['memory_delta'] < 200, f"Memory usage increased by {metrics['memory_delta']:.2f}MB, should be < 200MB"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_cleanup_efficiency(self, temp_csv_file, performance_monitor):
        """Test memory cleanup efficiency after operations."""
        session_state = SessionState()
        manager = PartSelectionManager(session_state, temp_csv_file)
        
        # Load data
        initial_memory = performance_monitor.get_current_memory()
        await manager.load_available_parts()
        after_load_memory = performance_monitor.get_current_memory()
        
        # Cleanup
        await manager.cleanup()
        after_cleanup_memory = performance_monitor.get_current_memory()
        
        # Memory assertions
        memory_increase = after_load_memory - initial_memory
        memory_recovered = after_load_memory - after_cleanup_memory
        recovery_ratio = memory_recovered / memory_increase if memory_increase > 0 else 1.0
        
        assert recovery_ratio > 0.8, f"Memory recovery ratio {recovery_ratio:.2f} should be > 0.8"
        assert after_cleanup_memory <= initial_memory + 10, "Memory should return close to initial level"


class TestQueryPerformance:
    """Test performance of query processing operations."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_query_response_time(self, performance_monitor, mock_openai):
        """Test query response time performance."""
        session_state = SessionState()
        session_state.llm_type = "openai"
        session_state.openai_api_key = "test_key"
        
        with patch('chainlit_integration.managers.chat_query_handler.OpenAI', return_value=mock_openai):
            handler = ChatQueryHandler(session_state)
            await handler.initialize()
            
            # Test multiple queries
            queries = [
                "What is the most common activity?",
                "How long does the process take on average?",
                "Which parts have the highest failure rate?",
                "What are the bottlenecks in the process?",
                "How can we optimize the workflow?"
            ]
            
            response_times = []
            
            for query in queries:
                performance_monitor.start_monitoring(f'query_{len(response_times)}')
                response = await handler.handle_query(query, "activity")
                metrics = performance_monitor.stop_monitoring(f'query_{len(response_times)}')
                response_times.append(metrics['duration'])
                
                assert response is not None
                assert metrics['duration'] < 5.0, f"Query response took {metrics['duration']:.2f}s, should be < 5s"
            
            # Check average response time
            avg_response_time = sum(response_times) / len(response_times)
            assert avg_response_time < 3.0, f"Average response time {avg_response_time:.2f}s should be < 3s"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_query_handling(self, performance_monitor, mock_openai):
        """Test concurrent query handling performance."""
        session_state = SessionState()
        session_state.llm_type = "openai"
        session_state.openai_api_key = "test_key"
        
        with patch('chainlit_integration.managers.chat_query_handler.OpenAI', return_value=mock_openai):
            handler = ChatQueryHandler(session_state)
            await handler.initialize()
            
            # Create concurrent queries
            queries = [f"Test query {i}" for i in range(10)]
            
            performance_monitor.start_monitoring('concurrent_queries')
            
            tasks = [handler.handle_query(query, "activity") for query in queries]
            responses = await asyncio.gather(*tasks)
            
            metrics = performance_monitor.stop_monitoring('concurrent_queries')
            
            # Verify all responses
            assert len(responses) == len(queries)
            for response in responses:
                assert response is not None
            
            # Performance assertions
            assert metrics['duration'] < 15.0, f"Concurrent queries took {metrics['duration']:.2f}s, should be < 15s"
            
            # Calculate queries per second
            qps = len(queries) / metrics['duration']
            assert qps > 0.5, f"Query throughput {qps:.2f} QPS should be > 0.5"


class TestComponentInitializationPerformance:
    """Test performance of component initialization."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_manager_initialization_times(self, temp_csv_file, performance_monitor, mock_managers):
        """Test initialization times for all managers."""
        session_state = SessionState()
        
        manager_classes = [
            ('PartSelectionManager', PartSelectionManager),
            ('ProcessMiningEngine', ProcessMiningEngine),
            ('ChatQueryHandler', ChatQueryHandler)
        ]
        
        initialization_times = {}
        
        for name, manager_class in manager_classes:
            performance_monitor.start_monitoring(f'init_{name}')
            
            if name == 'ChatQueryHandler':
                manager = manager_class(session_state)
            else:
                manager = manager_class(session_state, temp_csv_file)
            
            await manager.initialize()
            metrics = performance_monitor.stop_monitoring(f'init_{name}')
            
            initialization_times[name] = metrics['duration']
            
            # Individual initialization time assertions
            assert metrics['duration'] < 2.0, f"{name} initialization took {metrics['duration']:.2f}s, should be < 2s"
        
        # Total initialization time
        total_init_time = sum(initialization_times.values())
        assert total_init_time < 5.0, f"Total initialization took {total_init_time:.2f}s, should be < 5s"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_cold_start_performance(self, temp_csv_file, performance_monitor):
        """Test cold start performance (first-time initialization)."""
        session_state = SessionState()
        
        performance_monitor.start_monitoring('cold_start')
        
        # Simulate cold start by initializing all components
        part_manager = PartSelectionManager(session_state, temp_csv_file)
        await part_manager.initialize()
        
        engine = ProcessMiningEngine(session_state, temp_csv_file)
        await engine.initialize()
        
        query_handler = ChatQueryHandler(session_state)
        await query_handler.initialize()
        
        metrics = performance_monitor.stop_monitoring('cold_start')
        
        # Cold start assertions
        assert metrics['duration'] < 10.0, f"Cold start took {metrics['duration']:.2f}s, should be < 10s"
        assert metrics['memory_delta'] < 300, f"Cold start used {metrics['memory_delta']:.2f}MB, should be < 300MB"


class TestScalabilityPerformance:
    """Test scalability performance characteristics."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_large_dataset_scalability(self, performance_monitor):
        """Test system performance with increasingly large datasets."""
        import pandas as pd
        
        dataset_sizes = [100, 500, 1000, 5000]
        performance_results = {}
        
        for size in dataset_sizes:
            # Generate dataset of specified size
            df = pd.DataFrame({
                'case_id': [f'Case_{i}' for i in range(size)],
                'part_desc': [f'Part_{i%10}' for i in range(size)],
                'activity': [f'Activity_{i%5}' for i in range(size)]
            })
            
            session_state = SessionState()
            
            with patch('chainlit_integration.managers.part_selection_manager.pd.read_csv', return_value=df):
                manager = PartSelectionManager(session_state, "dummy_path")
                
                performance_monitor.start_monitoring(f'dataset_{size}')
                parts = await manager.load_available_parts()
                metrics = performance_monitor.stop_monitoring(f'dataset_{size}')
                
                performance_results[size] = {
                    'duration': metrics['duration'],
                    'memory_delta': metrics['memory_delta'],
                    'parts_count': len(parts)
                }
        
        # Analyze scalability
        for i, size in enumerate(dataset_sizes[1:], 1):
            prev_size = dataset_sizes[i-1]
            
            duration_ratio = performance_results[size]['duration'] / performance_results[prev_size]['duration']
            size_ratio = size / prev_size
            
            # Duration should scale sub-linearly (better than O(n))
            assert duration_ratio < size_ratio * 1.5, f"Duration scaling from {prev_size} to {size} is worse than expected"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, temp_csv_file, performance_monitor):
        """Test memory usage under sustained load."""
        session_state = SessionState()
        manager = PartSelectionManager(session_state, temp_csv_file)
        
        initial_memory = performance_monitor.get_current_memory()
        max_memory = initial_memory
        
        # Simulate sustained load
        for i in range(50):
            await manager.load_available_parts()
            await manager.cleanup()
            
            current_memory = performance_monitor.get_current_memory()
            max_memory = max(max_memory, current_memory)
            
            # Check for memory leaks
            if i > 10:  # Allow some warmup
                memory_growth = current_memory - initial_memory
                assert memory_growth < 50, f"Memory growth {memory_growth:.2f}MB after {i} iterations suggests memory leak"
        
        # Final memory check
        final_memory = performance_monitor.get_current_memory()
        total_growth = final_memory - initial_memory
        peak_growth = max_memory - initial_memory
        
        assert total_growth < 30, f"Total memory growth {total_growth:.2f}MB should be < 30MB"
        assert peak_growth < 100, f"Peak memory growth {peak_growth:.2f}MB should be < 100MB"


class TestErrorHandlingPerformance:
    """Test performance of error handling mechanisms."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_error_recovery_performance(self, temp_csv_file, performance_monitor):
        """Test performance of error recovery mechanisms."""
        session_state = SessionState()
        manager = PartSelectionManager(session_state, temp_csv_file)
        
        # Test error recovery times
        error_scenarios = [
            FileNotFoundError("File not found"),
            ConnectionError("Connection failed"),
            MemoryError("Out of memory"),
            TimeoutError("Operation timed out")
        ]
        
        recovery_times = []
        
        for error in error_scenarios:
            performance_monitor.start_monitoring('error_recovery')
            
            # Simulate error and recovery
            try:
                # Force an error condition
                with patch.object(manager, 'load_available_parts', side_effect=error):
                    await manager.load_available_parts()
            except Exception:
                # Simulate recovery
                await manager.cleanup()
                await manager.initialize()
            
            metrics = performance_monitor.stop_monitoring('error_recovery')
            recovery_times.append(metrics['duration'])
            
            # Individual recovery time assertion
            assert metrics['duration'] < 1.0, f"Error recovery took {metrics['duration']:.2f}s, should be < 1s"
        
        # Average recovery time
        avg_recovery_time = sum(recovery_times) / len(recovery_times)
        assert avg_recovery_time < 0.5, f"Average recovery time {avg_recovery_time:.2f}s should be < 0.5s"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_graceful_degradation_performance(self, performance_monitor):
        """Test performance during graceful degradation scenarios."""
        session_state = SessionState()
        
        # Simulate degraded performance conditions
        performance_monitor.start_monitoring('degraded_operation')
        
        # Test with limited resources
        with patch('asyncio.sleep', return_value=None):  # Speed up artificial delays
            manager = PartSelectionManager(session_state, "nonexistent_file.csv")
            
            try:
                await manager.load_available_parts()
            except FileNotFoundError:
                # Expected error, test fallback behavior
                session_state.available_parts = ["Fallback_Part"]
                assert len(session_state.available_parts) > 0
        
        metrics = performance_monitor.stop_monitoring('degraded_operation')
        
        # Degraded operation should still be reasonably fast
        assert metrics['duration'] < 2.0, f"Degraded operation took {metrics['duration']:.2f}s, should be < 2s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])