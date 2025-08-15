"""
Pytest configuration and shared fixtures for comprehensive test suite.
"""

import pytest
import asyncio
import tempfile
import pandas as pd
import os
import sys
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock external dependencies before importing our modules
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


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_csv_data():
    """Generate sample CSV data for testing."""
    return """case_id,activity,timestamp,part_desc,resource,cost
Case_001,Order_Received,2024-01-01 08:00:00,Motor_Housing,System,0
Case_001,Material_Preparation,2024-01-01 09:15:00,Motor_Housing,Worker_A,50
Case_001,Turning_Milling_Machine_4,2024-01-01 10:30:00,Motor_Housing,Machine_4,120
Case_001,Quality_Control,2024-01-01 14:20:00,Motor_Housing,QC_Inspector,30
Case_001,Packaging,2024-01-01 15:45:00,Motor_Housing,Worker_B,25
Case_001,Shipping,2024-01-01 16:30:00,Motor_Housing,System,15
Case_002,Order_Received,2024-01-01 08:30:00,Cable_Head,System,0
Case_002,Material_Preparation,2024-01-01 10:00:00,Cable_Head,Worker_A,45
Case_002,Laser_Marking_Machine_7,2024-01-01 11:15:00,Cable_Head,Machine_7,80
Case_002,Quality_Control,2024-01-01 13:30:00,Cable_Head,QC_Inspector,30
Case_002,Packaging,2024-01-01 14:15:00,Cable_Head,Worker_B,25
Case_002,Shipping,2024-01-01 15:00:00,Cable_Head,System,15
Case_003,Order_Received,2024-01-01 09:00:00,Connector_Pin,System,0
Case_003,Material_Preparation,2024-01-01 10:30:00,Connector_Pin,Worker_C,40
Case_003,Lapping_Machine_1,2024-01-01 12:00:00,Connector_Pin,Machine_1,100
Case_003,Quality_Control,2024-01-01 15:45:00,Connector_Pin,QC_Inspector,30
Case_003,Packaging,2024-01-01 16:30:00,Connector_Pin,Worker_B,25
Case_003,Shipping,2024-01-01 17:15:00,Connector_Pin,System,15
Case_004,Order_Received,2024-01-02 08:00:00,Motor_Housing,System,0
Case_004,Material_Preparation,2024-01-02 09:30:00,Motor_Housing,Worker_A,50
Case_004,Turning_Milling_Machine_4,2024-01-02 11:00:00,Motor_Housing,Machine_4,120
Case_004,Rework_Required,2024-01-02 14:30:00,Motor_Housing,QC_Inspector,0
Case_004,Turning_Milling_Machine_4,2024-01-02 15:15:00,Motor_Housing,Machine_4,60
Case_004,Quality_Control,2024-01-02 16:45:00,Motor_Housing,QC_Inspector,30
Case_004,Packaging,2024-01-02 17:30:00,Motor_Housing,Worker_B,25
Case_004,Shipping,2024-01-02 18:15:00,Motor_Housing,System,15
Case_005,Order_Received,2024-01-02 08:15:00,Cable_Head,System,0
Case_005,Material_Preparation,2024-01-02 09:45:00,Cable_Head,Worker_A,45
Case_005,Laser_Marking_Machine_7,2024-01-02 11:30:00,Cable_Head,Machine_7,80
Case_005,Quality_Control,2024-01-02 13:15:00,Cable_Head,QC_Inspector,30
Case_005,Packaging,2024-01-02 14:00:00,Cable_Head,Worker_B,25
Case_005,Shipping,2024-01-02 14:45:00,Cable_Head,System,15"""


@pytest.fixture
def large_sample_csv_data():
    """Generate larger sample CSV data for performance testing."""
    import random
    from datetime import datetime, timedelta
    
    parts = ["Motor_Housing", "Cable_Head", "Connector_Pin", "Gear_Assembly", "Control_Unit"]
    activities = [
        "Order_Received", "Material_Preparation", "Turning_Milling_Machine_4",
        "Laser_Marking_Machine_7", "Lapping_Machine_1", "Quality_Control",
        "Rework_Required", "Packaging", "Shipping"
    ]
    resources = ["System", "Worker_A", "Worker_B", "Worker_C", "Machine_4", "Machine_7", "Machine_1", "QC_Inspector"]
    
    rows = ["case_id,activity,timestamp,part_desc,resource,cost"]
    
    for case_num in range(1, 1001):  # 1000 cases
        case_id = f"Case_{case_num:04d}"
        part = random.choice(parts)
        start_time = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 30))
        
        # Generate process flow for this case
        case_activities = ["Order_Received"]
        if random.random() > 0.7:  # 30% chance of rework
            case_activities.extend(["Material_Preparation", random.choice(activities[2:5]), "Rework_Required", random.choice(activities[2:5])])
        else:
            case_activities.extend(["Material_Preparation", random.choice(activities[2:5])])
        case_activities.extend(["Quality_Control", "Packaging", "Shipping"])
        
        current_time = start_time
        for activity in case_activities:
            resource = random.choice(resources)
            cost = random.randint(0, 150) if activity != "Order_Received" else 0
            
            rows.append(f"{case_id},{activity},{current_time.strftime('%Y-%m-%d %H:%M:%S')},{part},{resource},{cost}")
            current_time += timedelta(hours=random.randint(1, 4), minutes=random.randint(0, 59))
    
    return "\n".join(rows)


@pytest.fixture
def temp_csv_file(sample_csv_data):
    """Create a temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(sample_csv_data)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def large_temp_csv_file(large_sample_csv_data):
    """Create a temporary CSV file with large sample data for performance testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(large_sample_csv_data)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def session_state():
    """Create a fresh session state for testing."""
    return SessionState()


@pytest.fixture
def mock_chainlit():
    """Mock Chainlit components."""
    mock_cl = Mock()
    mock_cl.Message = Mock()
    mock_cl.Image = Mock()
    mock_cl.File = Mock()
    mock_cl.Action = Mock()
    mock_cl.user_session = Mock()
    mock_cl.user_session.get = Mock(return_value=SessionState())
    mock_cl.user_session.set = Mock()
    return mock_cl


@pytest.fixture
def mock_neo4j():
    """Mock Neo4j driver and session."""
    mock_driver = Mock()
    mock_session = Mock()
    mock_session.run = Mock()
    mock_session.close = Mock()
    mock_driver.session = Mock(return_value=mock_session)
    mock_driver.close = Mock()
    return mock_driver


@pytest.fixture
def mock_openai():
    """Mock OpenAI client."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "Test response"
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    return mock_client


@pytest.fixture
def mock_pm4py():
    """Mock PM4py components."""
    mock_pm4py = Mock()
    mock_pm4py.discover_dfg = Mock(return_value=({}, {}, {}))
    mock_pm4py.discover_performance_dfg = Mock(return_value=({}, {}, {}))
    mock_pm4py.view_dfg = Mock()
    mock_pm4py.save_vis_dfg = Mock()
    return mock_pm4py


@pytest.fixture
def performance_metrics():
    """Fixture to collect performance metrics during tests."""
    metrics = {
        'execution_times': [],
        'memory_usage': [],
        'component_init_times': {},
        'query_response_times': []
    }
    return metrics


@pytest.fixture
def mock_managers():
    """Create mock manager instances for testing."""
    managers = {}
    
    manager_configs = {
        "session_manager": ["initialize", "cleanup"],
        "llm_manager": ["initialize", "handle_ollama_selection", "handle_openai_selection", "cleanup"],
        "part_manager": ["initialize", "handle_part_selection", "load_available_parts", "cleanup"],
        "engine": ["initialize", "process_data", "setup_retrievers", "cleanup_resources", "cleanup"],
        "context_manager": ["initialize", "handle_context_selection", "cleanup"],
        "query_handler": ["initialize", "handle_query", "cleanup"],
        "viz_manager": ["initialize", "generate_and_display_automatic_visualizations", "cleanup_temp_files", "cleanup"]
    }
    
    for name, methods in manager_configs.items():
        manager = Mock()
        for method in methods:
            if method == "process_data":
                setattr(manager, method, AsyncMock(return_value=ProcessingResult(success=True)))
            elif method == "load_available_parts":
                setattr(manager, method, AsyncMock(return_value=["Motor_Housing", "Cable_Head", "Connector_Pin"]))
            elif method == "handle_query":
                setattr(manager, method, AsyncMock(return_value="Test response"))
            else:
                setattr(manager, method, AsyncMock(return_value=True))
        
        managers[name] = manager
    
    return managers


@pytest.fixture
def error_scenarios():
    """Define common error scenarios for testing."""
    return {
        'file_not_found': FileNotFoundError("CSV file not found"),
        'connection_error': ConnectionError("Neo4j connection failed"),
        'api_error': Exception("OpenAI API error"),
        'processing_error': Exception("PM4py processing failed"),
        'memory_error': MemoryError("Insufficient memory"),
        'timeout_error': TimeoutError("Operation timed out")
    }


class TestDataGenerator:
    """Helper class for generating test data."""
    
    @staticmethod
    def generate_process_variants(num_variants: int = 5) -> List[Dict[str, Any]]:
        """Generate different process variants for testing."""
        variants = []
        for i in range(num_variants):
            variants.append({
                'variant_id': f'variant_{i}',
                'activities': [f'activity_{j}' for j in range(3, 8)],
                'frequency': 10 + i * 5,
                'avg_duration': 120 + i * 30
            })
        return variants
    
    @staticmethod
    def generate_dfg_data() -> Dict[str, Any]:
        """Generate sample DFG data."""
        return {
            'activities': ['Start', 'A', 'B', 'C', 'End'],
            'edges': [('Start', 'A'), ('A', 'B'), ('B', 'C'), ('C', 'End')],
            'frequencies': {('Start', 'A'): 100, ('A', 'B'): 95, ('B', 'C'): 90, ('C', 'End'): 85}
        }
    
    @staticmethod
    def generate_performance_data() -> Dict[str, Any]:
        """Generate sample performance data."""
        return {
            'activities': ['Start', 'A', 'B', 'C', 'End'],
            'durations': {'A': 30, 'B': 45, 'C': 60},
            'avg_times': {('Start', 'A'): 15, ('A', 'B'): 30, ('B', 'C'): 45, ('C', 'End'): 20}
        }


@pytest.fixture
def test_data_generator():
    """Provide test data generator."""
    return TestDataGenerator()