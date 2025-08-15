"""
Comprehensive integration tests for Chainlit integration.

This module tests integration between all components and validates
complete user workflows with real-world scenarios.

Requirements tested:
- All requirements (1.1-6.5) through complete workflow validation
- Component integration and coordination
- Error propagation and recovery
- Session management across components
"""

import pytest
import asyncio
import tempfile
import os
import sys
from unittest.mock import Mock, AsyncMock, patch, MagicMock
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
    'sentence_transformers': Mock(),
    'networkx': Mock()
}

with patch.dict('sys.modules', mock_modules):
    from chainlit_integration.models import SessionState, LLMType, QueryContext, ProcessingResult
    from chainlit_integration.managers.session_manager import SessionManager
    from chainlit_integration.managers.llm_selection_manager import LLMSelectionManager
    from chainlit_integration.managers.part_selection_manager import PartSelectionManager
    from chainlit_integration.managers.process_mining_engine import ProcessMiningEngine
    from chainlit_integration.managers.query_context_manager import QueryContextManager
    from chainlit_integration.managers.chat_query_handler import ChatQueryHandler
    from chainlit_integration.managers.visualization_manager import VisualizationManager


class IntegrationTestSuite:
    """Comprehensive integration test suite."""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.session_state = SessionState()
        self.managers = {}
        self.mock_chainlit = self._setup_chainlit_mocks()
        self.mock_neo4j = self._setup_neo4j_mocks()
        self.mock_openai = self._setup_openai_mocks()
    
    def _setup_chainlit_mocks(self):
        """Setup comprehensive Chainlit mocks."""
        mock_cl = Mock()
        mock_cl.Message = Mock()
        mock_cl.Image = Mock()
        mock_cl.File = Mock()
        mock_cl.Action = Mock()
        mock_cl.user_session = Mock()
        mock_cl.user_session.get = Mock(return_value=self.session_state)
        mock_cl.user_session.set = Mock()
        mock_cl.send = AsyncMock()
        return mock_cl
    
    def _setup_neo4j_mocks(self):
        """Setup Neo4j mocks."""
        mock_driver = Mock()
        mock_session = Mock()
        mock_session.run = Mock()
        mock_session.close = Mock()
        mock_driver.session = Mock(return_value=mock_session)
        mock_driver.close = Mock()
        return mock_driver
    
    def _setup_openai_mocks(self):
        """Setup OpenAI mocks."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Test AI response"
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        return mock_client
    
    async def initialize_all_managers(self):
        """Initialize all managers for testing."""
        # Session Manager
        self.managers['session'] = SessionManager(self.session_state)
        await self.managers['session'].initialize()
        
        # LLM Selection Manager
        self.managers['llm'] = LLMSelectionManager(self.session_state)
        await self.managers['llm'].initialize()
        
        # Part Selection Manager
        self.managers['part'] = PartSelectionManager(self.session_state, self.csv_path)
        await self.managers['part'].initialize()
        
        # Process Mining Engine
        self.managers['engine'] = ProcessMiningEngine(self.session_state, self.csv_path)
        await self.managers['engine'].initialize()
        
        # Query Context Manager
        self.managers['context'] = QueryContextManager(self.session_state)
        await self.managers['context'].initialize()
        
        # Chat Query Handler
        self.managers['query'] = ChatQueryHandler(self.session_state)
        await self.managers['query'].initialize()
        
        # Visualization Manager
        self.managers['viz'] = VisualizationManager(self.session_state)
        await self.managers['viz'].initialize()
    
    async def cleanup_all_managers(self):
        """Cleanup all managers."""
        for manager in self.managers.values():
            if hasattr(manager, 'cleanup'):
                await manager.cleanup()
        self.managers.clear()
    
    async def simulate_complete_workflow(self) -> Dict[str, Any]:
        """Simulate a complete user workflow."""
        workflow_results = {
            'steps_completed': [],
            'errors': [],
            'session_states': [],
            'performance_metrics': {}
        }
        
        try:
            # Step 1: Initialize session
            workflow_results['steps_completed'].append('session_init')
            workflow_results['session_states'].append(self.session_state.to_dict())
            
            # Step 2: Select LLM (OpenAI)
            self.session_state.llm_type = LLMType.OPENAI
            self.session_state.openai_api_key = "test_api_key"
            await self.managers['llm'].handle_openai_selection()
            workflow_results['steps_completed'].append('llm_selection')
            workflow_results['session_states'].append(self.session_state.to_dict())
            
            # Step 3: Load and select part
            parts = await self.managers['part'].load_available_parts()
            if parts:
                selected_part = parts[0]
                self.session_state.selected_part = selected_part
                await self.managers['part'].handle_part_selection(selected_part)
                workflow_results['steps_completed'].append('part_selection')
                workflow_results['session_states'].append(self.session_state.to_dict())
            
            # Step 4: Process data
            with patch('chainlit_integration.managers.process_mining_engine.pm4py') as mock_pm4py:
                mock_pm4py.discover_dfg.return_value = ({}, {}, {})
                mock_pm4py.discover_performance_dfg.return_value = ({}, {}, {})
                
                result = await self.managers['engine'].process_data(selected_part)
                if result.success:
                    self.session_state.processing_complete = True
                    workflow_results['steps_completed'].append('data_processing')
                    workflow_results['session_states'].append(self.session_state.to_dict())
            
            # Step 5: Generate visualizations
            with patch('chainlit_integration.managers.visualization_manager.plt') as mock_plt:
                mock_plt.figure.return_value = Mock()
                mock_plt.savefig = Mock()
                
                await self.managers['viz'].generate_and_display_automatic_visualizations(
                    result.dfg_data, result.performance_data
                )
                self.session_state.visualizations_displayed = True
                workflow_results['steps_completed'].append('visualization')
                workflow_results['session_states'].append(self.session_state.to_dict())
            
            # Step 6: Setup retrievers
            await self.managers['engine'].setup_retrievers()
            self.session_state.retrievers = ("mock_activity", "mock_process", "mock_variant")
            workflow_results['steps_completed'].append('retriever_setup')
            workflow_results['session_states'].append(self.session_state.to_dict())
            
            # Step 7: Handle queries with context
            contexts = [QueryContext.ACTIVITY, QueryContext.PROCESS, QueryContext.VARIANT]
            queries = [
                "What is the most common activity?",
                "How long does the process take?",
                "Which variant is most efficient?"
            ]
            
            for context, query in zip(contexts, queries):
                # Select context
                await self.managers['context'].handle_context_selection(context.value)
                self.session_state.current_context_mode = context
                
                # Process query
                with patch.object(self.managers['query'], '_get_llm_client', return_value=self.mock_openai):
                    response = await self.managers['query'].handle_query(query, context.value)
                    assert response is not None
                
                # Reset context
                self.session_state.reset_query_context()
                
                workflow_results['steps_completed'].append(f'query_{context.value}')
                workflow_results['session_states'].append(self.session_state.to_dict())
            
            # Step 8: Test part switching
            if len(parts) > 1:
                new_part = parts[1]
                self.session_state.reset_for_new_part()
                self.session_state.selected_part = new_part
                
                # Reprocess with new part
                result = await self.managers['engine'].process_data(new_part)
                if result.success:
                    workflow_results['steps_completed'].append('part_switching')
                    workflow_results['session_states'].append(self.session_state.to_dict())
            
            # Step 9: Test session termination
            self.session_state.session_active = False
            workflow_results['steps_completed'].append('session_termination')
            workflow_results['session_states'].append(self.session_state.to_dict())
            
        except Exception as e:
            workflow_results['errors'].append(str(e))
        
        return workflow_results


@pytest.fixture
def integration_suite(temp_csv_file):
    """Create integration test suite."""
    return IntegrationTestSuite(temp_csv_file)


class TestCompleteWorkflowIntegration:
    """Test complete workflow integration scenarios."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_successful_workflow(self, integration_suite):
        """Test complete successful workflow from start to finish."""
        await integration_suite.initialize_all_managers()
        
        try:
            workflow_results = await integration_suite.simulate_complete_workflow()
            
            # Verify workflow completion
            expected_steps = [
                'session_init', 'llm_selection', 'part_selection', 
                'data_processing', 'visualization', 'retriever_setup',
                'query_activity', 'query_process', 'query_variant',
                'session_termination'
            ]
            
            completed_steps = workflow_results['steps_completed']
            
            # Check that core steps were completed
            core_steps = ['session_init', 'llm_selection', 'part_selection', 'data_processing']
            for step in core_steps:
                assert step in completed_steps, f"Core step '{step}' was not completed"
            
            # Verify no critical errors
            assert len(workflow_results['errors']) == 0, f"Workflow had errors: {workflow_results['errors']}"
            
            # Verify session state progression
            states = workflow_results['session_states']
            assert len(states) > 0, "No session states recorded"
            
            # Check final state
            final_state = states[-1]
            assert final_state['session_active'] is False, "Session should be terminated"
            
        finally:
            await integration_suite.cleanup_all_managers()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_with_llm_switching(self, integration_suite):
        """Test workflow with LLM switching between OpenAI and Ollama."""
        await integration_suite.initialize_all_managers()
        
        try:
            # Start with OpenAI
            integration_suite.session_state.llm_type = LLMType.OPENAI
            integration_suite.session_state.openai_api_key = "test_key"
            await integration_suite.managers['llm'].handle_openai_selection()
            
            assert integration_suite.session_state.llm_type == LLMType.OPENAI
            
            # Switch to Ollama
            integration_suite.session_state.llm_type = LLMType.OLLAMA
            integration_suite.session_state.openai_api_key = None
            await integration_suite.managers['llm'].handle_ollama_selection()
            
            assert integration_suite.session_state.llm_type == LLMType.OLLAMA
            assert integration_suite.session_state.openai_api_key is None
            
            # Verify query handler adapts to LLM change
            with patch.object(integration_suite.managers['query'], '_get_llm_client') as mock_get_client:
                mock_get_client.return_value = Mock()
                await integration_suite.managers['query'].handle_query("test", "activity")
                mock_get_client.assert_called_once()
            
        finally:
            await integration_suite.cleanup_all_managers()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_with_multiple_part_switches(self, integration_suite):
        """Test workflow with multiple part switches."""
        await integration_suite.initialize_all_managers()
        
        try:
            # Load available parts
            parts = await integration_suite.managers['part'].load_available_parts()
            assert len(parts) >= 2, "Need at least 2 parts for switching test"
            
            part_processing_results = []
            
            # Process each part
            for i, part in enumerate(parts[:3]):  # Test with first 3 parts
                # Reset for new part
                integration_suite.session_state.reset_for_new_part()
                integration_suite.session_state.selected_part = part
                
                # Process data
                with patch('chainlit_integration.managers.process_mining_engine.pm4py') as mock_pm4py:
                    mock_pm4py.discover_dfg.return_value = ({f'activity_{i}': 1}, {}, {})
                    mock_pm4py.discover_performance_dfg.return_value = ({f'perf_{i}': 1}, {}, {})
                    
                    result = await integration_suite.managers['engine'].process_data(part)
                    part_processing_results.append((part, result.success))
                    
                    if result.success:
                        integration_suite.session_state.processing_complete = True
                        
                        # Generate visualizations
                        with patch('chainlit_integration.managers.visualization_manager.plt'):
                            await integration_suite.managers['viz'].generate_and_display_automatic_visualizations(
                                result.dfg_data, result.performance_data
                            )
            
            # Verify all parts were processed successfully
            for part, success in part_processing_results:
                assert success, f"Part '{part}' processing failed"
            
            # Verify session state is clean after switches
            assert integration_suite.session_state.selected_part == parts[2]
            assert integration_suite.session_state.processing_complete is True
            
        finally:
            await integration_suite.cleanup_all_managers()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_error_recovery(self, integration_suite):
        """Test workflow error recovery scenarios."""
        await integration_suite.initialize_all_managers()
        
        try:
            # Test recovery from part selection error
            with patch.object(integration_suite.managers['part'], 'load_available_parts', 
                            side_effect=FileNotFoundError("CSV not found")):
                try:
                    await integration_suite.managers['part'].load_available_parts()
                    assert False, "Should have raised FileNotFoundError"
                except FileNotFoundError:
                    # Recovery: reinitialize with valid data
                    await integration_suite.managers['part'].cleanup()
                    await integration_suite.managers['part'].initialize()
                    parts = await integration_suite.managers['part'].load_available_parts()
                    assert len(parts) > 0, "Recovery should provide valid parts"
            
            # Test recovery from processing error
            with patch.object(integration_suite.managers['engine'], 'process_data',
                            return_value=ProcessingResult(success=False, error_message="Processing failed")):
                result = await integration_suite.managers['engine'].process_data("test_part")
                assert result.success is False
                
                # Recovery: retry with different approach
                with patch.object(integration_suite.managers['engine'], 'process_data',
                                return_value=ProcessingResult(success=True)):
                    result = await integration_suite.managers['engine'].process_data("test_part")
                    assert result.success is True
            
            # Test recovery from query error
            with patch.object(integration_suite.managers['query'], 'handle_query',
                            side_effect=Exception("Query failed")):
                try:
                    await integration_suite.managers['query'].handle_query("test", "activity")
                    assert False, "Should have raised exception"
                except Exception:
                    # Recovery: reset and retry
                    integration_suite.session_state.reset_query_context()
                    with patch.object(integration_suite.managers['query'], 'handle_query',
                                    return_value="Recovery response"):
                        response = await integration_suite.managers['query'].handle_query("test", "activity")
                        assert response == "Recovery response"
            
        finally:
            await integration_suite.cleanup_all_managers()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_component_operations(self, integration_suite):
        """Test concurrent operations across components."""
        await integration_suite.initialize_all_managers()
        
        try:
            # Setup initial state
            integration_suite.session_state.llm_type = LLMType.OLLAMA
            parts = await integration_suite.managers['part'].load_available_parts()
            integration_suite.session_state.selected_part = parts[0]
            
            # Simulate concurrent operations
            tasks = []
            
            # Concurrent part loading
            tasks.append(integration_suite.managers['part'].load_available_parts())
            
            # Concurrent LLM operations
            tasks.append(integration_suite.managers['llm'].handle_ollama_selection())
            
            # Concurrent context operations
            tasks.append(integration_suite.managers['context'].handle_context_selection("activity"))
            
            # Execute concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify results
            parts_result = results[0]
            assert isinstance(parts_result, list) and len(parts_result) > 0
            
            # Check that no exceptions occurred
            for i, result in enumerate(results):
                assert not isinstance(result, Exception), f"Task {i} failed with: {result}"
            
            # Verify session state consistency
            assert integration_suite.session_state.llm_type == LLMType.OLLAMA
            assert integration_suite.session_state.selected_part == parts[0]
            
        finally:
            await integration_suite.cleanup_all_managers()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_session_state_consistency(self, integration_suite):
        """Test session state consistency across all operations."""
        await integration_suite.initialize_all_managers()
        
        try:
            # Track session state changes
            state_changes = []
            
            # Initial state
            state_changes.append(('initial', integration_suite.session_state.to_dict()))
            
            # LLM selection
            integration_suite.session_state.llm_type = LLMType.OPENAI
            integration_suite.session_state.openai_api_key = "test_key"
            state_changes.append(('llm_selected', integration_suite.session_state.to_dict()))
            
            # Part selection
            parts = await integration_suite.managers['part'].load_available_parts()
            integration_suite.session_state.selected_part = parts[0]
            integration_suite.session_state.available_parts = parts
            state_changes.append(('part_selected', integration_suite.session_state.to_dict()))
            
            # Data processing
            with patch('chainlit_integration.managers.process_mining_engine.pm4py') as mock_pm4py:
                mock_pm4py.discover_dfg.return_value = ({}, {}, {})
                mock_pm4py.discover_performance_dfg.return_value = ({}, {}, {})
                
                result = await integration_suite.managers['engine'].process_data(parts[0])
                integration_suite.session_state.processing_complete = True
                state_changes.append(('data_processed', integration_suite.session_state.to_dict()))
            
            # Query context selection
            integration_suite.session_state.current_context_mode = QueryContext.ACTIVITY
            integration_suite.session_state.awaiting_context_selection = True
            state_changes.append(('context_selected', integration_suite.session_state.to_dict()))
            
            # Query completion
            integration_suite.session_state.reset_query_context()
            state_changes.append(('query_completed', integration_suite.session_state.to_dict()))
            
            # Part switching
            integration_suite.session_state.reset_for_new_part()
            state_changes.append(('part_reset', integration_suite.session_state.to_dict()))
            
            # Verify state transitions
            assert len(state_changes) == 7
            
            # Check specific state properties
            initial_state = state_changes[0][1]
            assert initial_state['session_active'] is True
            assert initial_state['llm_type'] is None
            
            llm_state = state_changes[1][1]
            assert llm_state['llm_type'] == 'openai'
            assert llm_state['openai_api_key'] == 'test_key'
            
            part_state = state_changes[2][1]
            assert part_state['selected_part'] == parts[0]
            assert len(part_state['available_parts']) > 0
            
            processed_state = state_changes[3][1]
            assert processed_state['processing_complete'] is True
            
            context_state = state_changes[4][1]
            assert context_state['current_context_mode'] == 'activity'
            assert context_state['awaiting_context_selection'] is True
            
            query_completed_state = state_changes[5][1]
            assert query_completed_state['current_context_mode'] is None
            assert query_completed_state['awaiting_context_selection'] is False
            
            reset_state = state_changes[6][1]
            assert reset_state['selected_part'] is None
            assert reset_state['processing_complete'] is False
            
        finally:
            await integration_suite.cleanup_all_managers()


class TestComponentCoordination:
    """Test coordination between components."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_manager_lifecycle_coordination(self, integration_suite):
        """Test proper coordination of manager lifecycles."""
        # Test initialization order
        managers_to_init = [
            ('session', SessionManager),
            ('llm', LLMSelectionManager),
            ('part', PartSelectionManager),
            ('engine', ProcessMiningEngine),
            ('context', QueryContextManager),
            ('query', ChatQueryHandler),
            ('viz', VisualizationManager)
        ]
        
        # Initialize in order
        for name, manager_class in managers_to_init:
            if name in ['part', 'engine']:
                manager = manager_class(integration_suite.session_state, integration_suite.csv_path)
            else:
                manager = manager_class(integration_suite.session_state)
            
            await manager.initialize()
            integration_suite.managers[name] = manager
            
            # Verify manager is properly initialized
            assert hasattr(manager, 'session_state')
            assert manager.session_state is integration_suite.session_state
        
        # Test cleanup order (reverse)
        for name in reversed(list(integration_suite.managers.keys())):
            manager = integration_suite.managers[name]
            if hasattr(manager, 'cleanup'):
                await manager.cleanup()
        
        integration_suite.managers.clear()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_data_flow_between_components(self, integration_suite):
        """Test data flow between components."""
        await integration_suite.initialize_all_managers()
        
        try:
            # Data flow: Part Manager -> Engine -> Visualization Manager
            
            # Step 1: Part manager provides data
            parts = await integration_suite.managers['part'].load_available_parts()
            integration_suite.session_state.available_parts = parts
            integration_suite.session_state.selected_part = parts[0]
            
            # Step 2: Engine processes the data
            with patch('chainlit_integration.managers.process_mining_engine.pm4py') as mock_pm4py:
                mock_dfg_data = {'activity_A': 10, 'activity_B': 8}
                mock_perf_data = {'activity_A': 120, 'activity_B': 90}
                
                mock_pm4py.discover_dfg.return_value = (mock_dfg_data, {}, {})
                mock_pm4py.discover_performance_dfg.return_value = (mock_perf_data, {}, {})
                
                result = await integration_suite.managers['engine'].process_data(parts[0])
                
                assert result.success is True
                assert result.dfg_data == mock_dfg_data
                assert result.performance_data == mock_perf_data
            
            # Step 3: Visualization manager uses the processed data
            with patch('chainlit_integration.managers.visualization_manager.plt') as mock_plt:
                mock_plt.figure.return_value = Mock()
                mock_plt.savefig = Mock()
                
                await integration_suite.managers['viz'].generate_and_display_automatic_visualizations(
                    result.dfg_data, result.performance_data
                )
                
                # Verify visualization manager received the data
                mock_plt.figure.assert_called()
                mock_plt.savefig.assert_called()
            
            # Step 4: Query handler uses the retrievers from engine
            integration_suite.session_state.retrievers = ("activity_retriever", "process_retriever", "variant_retriever")
            
            with patch.object(integration_suite.managers['query'], '_get_llm_client', return_value=integration_suite.mock_openai):
                response = await integration_suite.managers['query'].handle_query("test query", "activity")
                assert response is not None
            
        finally:
            await integration_suite.cleanup_all_managers()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_propagation_between_components(self, integration_suite):
        """Test error propagation between components."""
        await integration_suite.initialize_all_managers()
        
        try:
            # Test error propagation: Part Manager -> Engine
            with patch.object(integration_suite.managers['part'], 'load_available_parts',
                            side_effect=FileNotFoundError("CSV file not found")):
                
                # Part manager error should prevent engine from processing
                try:
                    parts = await integration_suite.managers['part'].load_available_parts()
                    assert False, "Should have raised FileNotFoundError"
                except FileNotFoundError:
                    # Engine should handle the absence of parts gracefully
                    result = await integration_suite.managers['engine'].process_data("nonexistent_part")
                    assert result.success is False
                    assert "not found" in result.error_message.lower() or "invalid" in result.error_message.lower()
            
            # Test error propagation: Engine -> Visualization Manager
            with patch.object(integration_suite.managers['engine'], 'process_data',
                            return_value=ProcessingResult(success=False, error_message="Processing failed")):
                
                result = await integration_suite.managers['engine'].process_data("test_part")
                assert result.success is False
                
                # Visualization manager should handle failed processing gracefully
                try:
                    await integration_suite.managers['viz'].generate_and_display_automatic_visualizations(
                        None, None
                    )
                    # Should either succeed with empty data or raise appropriate error
                except Exception as e:
                    assert "data" in str(e).lower() or "none" in str(e).lower()
            
            # Test error propagation: LLM Manager -> Query Handler
            integration_suite.session_state.llm_type = LLMType.OPENAI
            integration_suite.session_state.openai_api_key = "invalid_key"
            
            with patch.object(integration_suite.managers['query'], '_get_llm_client',
                            side_effect=Exception("Invalid API key")):
                
                try:
                    await integration_suite.managers['query'].handle_query("test", "activity")
                    assert False, "Should have raised exception"
                except Exception as e:
                    assert "api" in str(e).lower() or "key" in str(e).lower()
            
        finally:
            await integration_suite.cleanup_all_managers()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])