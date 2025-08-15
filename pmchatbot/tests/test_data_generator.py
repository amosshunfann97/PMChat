"""
Test data generator for comprehensive testing.

This module generates various types of test data for validating
the Chainlit integration system with different scenarios.

Features:
- CSV data generation with various patterns
- Process mining test scenarios
- Performance test datasets
- Error condition simulation data
"""

import pandas as pd
import random
import tempfile
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json


class ProcessTestDataGenerator:
    """Generator for process mining test data."""
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducible data."""
        random.seed(seed)
        self.seed = seed
    
    def generate_basic_process_data(self, num_cases: int = 100) -> pd.DataFrame:
        """Generate basic process data for testing."""
        activities = [
            "Order_Received",
            "Material_Preparation", 
            "Manufacturing",
            "Quality_Control",
            "Packaging",
            "Shipping"
        ]
        
        parts = ["Motor_Housing", "Cable_Head", "Connector_Pin", "Gear_Assembly"]
        resources = ["System", "Worker_A", "Worker_B", "Machine_1", "Machine_2", "QC_Inspector"]
        
        data = []
        
        for case_num in range(1, num_cases + 1):
            case_id = f"Case_{case_num:04d}"
            part = random.choice(parts)
            
            # Generate process flow
            start_time = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 30))
            current_time = start_time
            
            for activity in activities:
                resource = random.choice(resources)
                cost = random.randint(10, 150) if activity != "Order_Received" else 0
                
                data.append({
                    'case_id': case_id,
                    'activity': activity,
                    'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'part_desc': part,
                    'resource': resource,
                    'cost': cost
                })
                
                # Add time between activities
                current_time += timedelta(
                    hours=random.randint(1, 8),
                    minutes=random.randint(0, 59)
                )
        
        return pd.DataFrame(data)
    
    def generate_complex_process_data(self, num_cases: int = 500) -> pd.DataFrame:
        """Generate complex process data with variants and rework."""
        base_activities = [
            "Order_Received",
            "Material_Preparation",
            "Manufacturing_Step_1",
            "Manufacturing_Step_2", 
            "Quality_Control",
            "Packaging",
            "Shipping"
        ]
        
        rework_activities = [
            "Rework_Required",
            "Defect_Analysis",
            "Corrective_Action"
        ]
        
        parts = [
            "Motor_Housing", "Cable_Head", "Connector_Pin", 
            "Gear_Assembly", "Control_Unit", "Sensor_Module"
        ]
        
        resources = [
            "System", "Worker_A", "Worker_B", "Worker_C",
            "Machine_1", "Machine_2", "Machine_3", "Machine_4",
            "QC_Inspector_1", "QC_Inspector_2", "Supervisor"
        ]
        
        data = []
        
        for case_num in range(1, num_cases + 1):
            case_id = f"Case_{case_num:04d}"
            part = random.choice(parts)
            
            # Determine process variant
            has_rework = random.random() < 0.2  # 20% chance of rework
            has_expedited = random.random() < 0.1  # 10% chance of expedited
            
            start_time = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 60))
            current_time = start_time
            
            # Generate main process flow
            activities_to_execute = base_activities.copy()
            
            if has_expedited:
                activities_to_execute.insert(1, "Expedited_Processing")
            
            for i, activity in enumerate(activities_to_execute):
                resource = random.choice(resources)
                
                # Simulate different costs for different activities
                if activity == "Order_Received":
                    cost = 0
                elif "Manufacturing" in activity:
                    cost = random.randint(80, 200)
                elif activity == "Quality_Control":
                    cost = random.randint(20, 50)
                else:
                    cost = random.randint(10, 60)
                
                data.append({
                    'case_id': case_id,
                    'activity': activity,
                    'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'part_desc': part,
                    'resource': resource,
                    'cost': cost
                })
                
                # Add rework after quality control if needed
                if activity == "Quality_Control" and has_rework:
                    rework_time = current_time + timedelta(hours=1)
                    
                    for rework_activity in rework_activities:
                        data.append({
                            'case_id': case_id,
                            'activity': rework_activity,
                            'timestamp': rework_time.strftime('%Y-%m-%d %H:%M:%S'),
                            'part_desc': part,
                            'resource': random.choice(resources),
                            'cost': random.randint(30, 100)
                        })
                        rework_time += timedelta(hours=random.randint(1, 3))
                    
                    # Repeat manufacturing after rework
                    manufacturing_activity = random.choice(["Manufacturing_Step_1", "Manufacturing_Step_2"])
                    data.append({
                        'case_id': case_id,
                        'activity': manufacturing_activity,
                        'timestamp': rework_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'part_desc': part,
                        'resource': random.choice(resources),
                        'cost': random.randint(40, 120)
                    })
                    
                    current_time = rework_time + timedelta(hours=2)
                else:
                    # Normal time progression
                    if has_expedited and "Manufacturing" in activity:
                        time_delta = timedelta(hours=random.randint(1, 3))  # Faster
                    else:
                        time_delta = timedelta(hours=random.randint(2, 8))
                    
                    current_time += time_delta
        
        return pd.DataFrame(data)
    
    def generate_performance_test_data(self, num_cases: int = 5000) -> pd.DataFrame:
        """Generate large dataset for performance testing."""
        activities = [
            "Start", "Task_A", "Task_B", "Task_C", "Task_D", 
            "Task_E", "Task_F", "Task_G", "Task_H", "End"
        ]
        
        parts = [f"Part_{i:03d}" for i in range(1, 51)]  # 50 different parts
        resources = [f"Resource_{i:02d}" for i in range(1, 21)]  # 20 resources
        
        data = []
        
        for case_num in range(1, num_cases + 1):
            case_id = f"Case_{case_num:06d}"
            part = random.choice(parts)
            
            start_time = datetime(2024, 1, 1) + timedelta(
                days=random.randint(0, 90),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            current_time = start_time
            
            # Generate random process variant (skip some activities)
            activities_to_execute = ["Start"]
            middle_activities = activities[1:-1]
            num_activities = random.randint(3, len(middle_activities))
            selected_activities = random.sample(middle_activities, num_activities)
            activities_to_execute.extend(sorted(selected_activities))
            activities_to_execute.append("End")
            
            for activity in activities_to_execute:
                resource = random.choice(resources)
                cost = random.randint(5, 200) if activity not in ["Start", "End"] else 0
                
                data.append({
                    'case_id': case_id,
                    'activity': activity,
                    'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'part_desc': part,
                    'resource': resource,
                    'cost': cost
                })
                
                # Variable time between activities
                current_time += timedelta(
                    hours=random.randint(0, 12),
                    minutes=random.randint(0, 59)
                )
        
        return pd.DataFrame(data)
    
    def generate_error_condition_data(self) -> Dict[str, pd.DataFrame]:
        """Generate data that triggers various error conditions."""
        error_datasets = {}
        
        # 1. Empty dataset
        error_datasets['empty'] = pd.DataFrame(columns=['case_id', 'activity', 'timestamp', 'part_desc'])
        
        # 2. Missing required columns
        error_datasets['missing_columns'] = pd.DataFrame({
            'case_id': ['Case_001'],
            'activity': ['Test_Activity']
            # Missing timestamp and part_desc
        })
        
        # 3. Invalid timestamps
        error_datasets['invalid_timestamps'] = pd.DataFrame({
            'case_id': ['Case_001', 'Case_002'],
            'activity': ['Activity_A', 'Activity_B'],
            'timestamp': ['invalid_date', '2024-13-45 25:70:80'],  # Invalid formats
            'part_desc': ['Part_A', 'Part_B']
        })
        
        # 4. Duplicate case-activity combinations
        error_datasets['duplicates'] = pd.DataFrame({
            'case_id': ['Case_001', 'Case_001', 'Case_001'],
            'activity': ['Activity_A', 'Activity_A', 'Activity_B'],  # Duplicate
            'timestamp': ['2024-01-01 10:00:00', '2024-01-01 10:00:00', '2024-01-01 11:00:00'],
            'part_desc': ['Part_A', 'Part_A', 'Part_A']
        })
        
        # 5. Single case (edge case)
        error_datasets['single_case'] = pd.DataFrame({
            'case_id': ['Case_001'],
            'activity': ['Single_Activity'],
            'timestamp': ['2024-01-01 10:00:00'],
            'part_desc': ['Single_Part']
        })
        
        # 6. Very long activity names (edge case)
        long_activity = "Very_Long_Activity_Name_That_Might_Cause_Issues_" * 5
        error_datasets['long_names'] = pd.DataFrame({
            'case_id': ['Case_001'],
            'activity': [long_activity],
            'timestamp': ['2024-01-01 10:00:00'],
            'part_desc': ['Part_With_Very_Long_Name_' * 10]
        })
        
        return error_datasets
    
    def save_test_datasets(self, output_dir: str = None) -> Dict[str, str]:
        """Save all test datasets to files."""
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="test_data_")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Generate and save datasets
        datasets = {
            'basic_100': self.generate_basic_process_data(100),
            'basic_500': self.generate_basic_process_data(500),
            'complex_500': self.generate_complex_process_data(500),
            'complex_1000': self.generate_complex_process_data(1000),
            'performance_5000': self.generate_performance_test_data(5000)
        }
        
        # Save main datasets
        for name, df in datasets.items():
            file_path = output_path / f"{name}.csv"
            df.to_csv(file_path, index=False)
            saved_files[name] = str(file_path)
        
        # Save error condition datasets
        error_datasets = self.generate_error_condition_data()
        for name, df in error_datasets.items():
            file_path = output_path / f"error_{name}.csv"
            df.to_csv(file_path, index=False)
            saved_files[f"error_{name}"] = str(file_path)
        
        # Save metadata
        metadata = {
            'generated_at': datetime.now().isoformat(),
            'seed': self.seed,
            'datasets': {name: {'rows': len(df), 'columns': list(df.columns)} 
                        for name, df in datasets.items()},
            'error_datasets': {name: {'rows': len(df), 'columns': list(df.columns)} 
                              for name, df in error_datasets.items()},
            'files': saved_files
        }
        
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        saved_files['metadata'] = str(metadata_path)
        
        return saved_files


class ChainlitTestDataGenerator:
    """Generator for Chainlit-specific test data."""
    
    def __init__(self):
        self.process_generator = ProcessTestDataGenerator()
    
    def generate_session_state_scenarios(self) -> List[Dict[str, Any]]:
        """Generate various session state scenarios for testing."""
        scenarios = []
        
        # 1. Fresh session
        scenarios.append({
            'name': 'fresh_session',
            'state': {
                'session_active': True,
                'llm_type': None,
                'openai_api_key': None,
                'selected_part': None,
                'available_parts': [],
                'processing_complete': False,
                'visualizations_displayed': False,
                'retrievers': None,
                'current_context_mode': None,
                'awaiting_context_selection': False
            }
        })
        
        # 2. LLM selected (OpenAI)
        scenarios.append({
            'name': 'openai_selected',
            'state': {
                'session_active': True,
                'llm_type': 'openai',
                'openai_api_key': 'test_api_key_123',
                'selected_part': None,
                'available_parts': ['Motor_Housing', 'Cable_Head', 'Connector_Pin'],
                'processing_complete': False,
                'visualizations_displayed': False,
                'retrievers': None,
                'current_context_mode': None,
                'awaiting_context_selection': False
            }
        })
        
        # 3. Part selected and processed
        scenarios.append({
            'name': 'part_processed',
            'state': {
                'session_active': True,
                'llm_type': 'ollama',
                'openai_api_key': None,
                'selected_part': 'Motor_Housing',
                'available_parts': ['Motor_Housing', 'Cable_Head', 'Connector_Pin'],
                'processing_complete': True,
                'visualizations_displayed': True,
                'retrievers': ('activity_retriever', 'process_retriever', 'variant_retriever'),
                'current_context_mode': None,
                'awaiting_context_selection': False
            }
        })
        
        # 4. Ready for queries
        scenarios.append({
            'name': 'ready_for_queries',
            'state': {
                'session_active': True,
                'llm_type': 'openai',
                'openai_api_key': 'test_api_key_456',
                'selected_part': 'Cable_Head',
                'available_parts': ['Motor_Housing', 'Cable_Head', 'Connector_Pin'],
                'processing_complete': True,
                'visualizations_displayed': True,
                'retrievers': ('activity_retriever', 'process_retriever', 'variant_retriever'),
                'current_context_mode': 'activity',
                'awaiting_context_selection': True
            }
        })
        
        # 5. Error state
        scenarios.append({
            'name': 'error_state',
            'state': {
                'session_active': True,
                'llm_type': 'openai',
                'openai_api_key': 'invalid_key',
                'selected_part': 'Invalid_Part',
                'available_parts': [],
                'processing_complete': False,
                'visualizations_displayed': False,
                'retrievers': None,
                'current_context_mode': None,
                'awaiting_context_selection': False
            }
        })
        
        return scenarios
    
    def generate_query_test_cases(self) -> List[Dict[str, Any]]:
        """Generate query test cases for different contexts."""
        test_cases = []
        
        contexts = ['activity', 'process', 'variant', 'combined']
        
        query_templates = {
            'activity': [
                "What is the most common activity?",
                "Which activity takes the longest time?",
                "How many times does {activity} occur?",
                "What activities come after {activity}?",
                "Which resources perform {activity}?"
            ],
            'process': [
                "What is the average process duration?",
                "How many cases are in the process?",
                "What is the process efficiency?",
                "Which process variant is most common?",
                "What are the process bottlenecks?"
            ],
            'variant': [
                "How many process variants exist?",
                "Which variant is most efficient?",
                "What is the difference between variants?",
                "Which variant has the highest cost?",
                "How do variants compare in duration?"
            ],
            'combined': [
                "Compare activity performance across variants",
                "Which activities are common to all variants?",
                "How do process costs vary by activity and variant?",
                "What is the overall process optimization potential?",
                "Analyze the relationship between activities and variants"
            ]
        }
        
        for context in contexts:
            for template in query_templates[context]:
                test_cases.append({
                    'context': context,
                    'query': template,
                    'expected_type': 'analysis_response',
                    'should_succeed': True
                })
        
        # Add edge cases
        edge_cases = [
            {
                'context': 'activity',
                'query': '',  # Empty query
                'expected_type': 'error',
                'should_succeed': False
            },
            {
                'context': 'process',
                'query': 'x' * 1000,  # Very long query
                'expected_type': 'response',
                'should_succeed': True
            },
            {
                'context': 'variant',
                'query': 'What is the meaning of life?',  # Unrelated query
                'expected_type': 'response',
                'should_succeed': True
            }
        ]
        
        test_cases.extend(edge_cases)
        return test_cases
    
    def generate_workflow_test_scenarios(self) -> List[Dict[str, Any]]:
        """Generate complete workflow test scenarios."""
        scenarios = []
        
        # 1. Happy path workflow
        scenarios.append({
            'name': 'happy_path',
            'steps': [
                {'action': 'start_session', 'expected_state': 'session_active'},
                {'action': 'select_llm', 'params': {'type': 'openai', 'api_key': 'test_key'}},
                {'action': 'load_parts', 'expected_result': 'parts_loaded'},
                {'action': 'select_part', 'params': {'part': 'Motor_Housing'}},
                {'action': 'process_data', 'expected_result': 'processing_complete'},
                {'action': 'generate_visualizations', 'expected_result': 'visualizations_displayed'},
                {'action': 'setup_retrievers', 'expected_result': 'retrievers_ready'},
                {'action': 'select_context', 'params': {'context': 'activity'}},
                {'action': 'submit_query', 'params': {'query': 'What is the most common activity?'}},
                {'action': 'end_session', 'expected_state': 'session_inactive'}
            ]
        })
        
        # 2. Part switching workflow
        scenarios.append({
            'name': 'part_switching',
            'steps': [
                {'action': 'start_session'},
                {'action': 'select_llm', 'params': {'type': 'ollama'}},
                {'action': 'select_part', 'params': {'part': 'Motor_Housing'}},
                {'action': 'process_data'},
                {'action': 'switch_part', 'params': {'part': 'Cable_Head'}},
                {'action': 'process_data'},
                {'action': 'submit_query', 'params': {'query': 'Compare the two parts'}},
                {'action': 'end_session'}
            ]
        })
        
        # 3. Error recovery workflow
        scenarios.append({
            'name': 'error_recovery',
            'steps': [
                {'action': 'start_session'},
                {'action': 'select_llm', 'params': {'type': 'openai', 'api_key': 'invalid_key'}},
                {'action': 'handle_error', 'expected_result': 'error_handled'},
                {'action': 'select_llm', 'params': {'type': 'ollama'}},
                {'action': 'select_part', 'params': {'part': 'Invalid_Part'}},
                {'action': 'handle_error', 'expected_result': 'error_handled'},
                {'action': 'select_part', 'params': {'part': 'Motor_Housing'}},
                {'action': 'process_data'},
                {'action': 'end_session'}
            ]
        })
        
        return scenarios


def create_test_data_suite(output_dir: str = None) -> Dict[str, Any]:
    """Create a complete test data suite."""
    process_gen = ProcessTestDataGenerator()
    chainlit_gen = ChainlitTestDataGenerator()
    
    # Generate and save process data
    saved_files = process_gen.save_test_datasets(output_dir)
    
    # Generate Chainlit test scenarios
    session_scenarios = chainlit_gen.generate_session_state_scenarios()
    query_test_cases = chainlit_gen.generate_query_test_cases()
    workflow_scenarios = chainlit_gen.generate_workflow_test_scenarios()
    
    # Save Chainlit test data
    if output_dir:
        output_path = Path(output_dir)
        
        # Save session scenarios
        with open(output_path / "session_scenarios.json", 'w') as f:
            json.dump(session_scenarios, f, indent=2)
        saved_files['session_scenarios'] = str(output_path / "session_scenarios.json")
        
        # Save query test cases
        with open(output_path / "query_test_cases.json", 'w') as f:
            json.dump(query_test_cases, f, indent=2)
        saved_files['query_test_cases'] = str(output_path / "query_test_cases.json")
        
        # Save workflow scenarios
        with open(output_path / "workflow_scenarios.json", 'w') as f:
            json.dump(workflow_scenarios, f, indent=2)
        saved_files['workflow_scenarios'] = str(output_path / "workflow_scenarios.json")
    
    return {
        'files': saved_files,
        'session_scenarios': session_scenarios,
        'query_test_cases': query_test_cases,
        'workflow_scenarios': workflow_scenarios
    }


if __name__ == "__main__":
    # Generate test data suite
    output_dir = "test_data_output"
    test_suite = create_test_data_suite(output_dir)
    
    print(f"✅ Test data suite generated in: {output_dir}")
    print(f"📁 Generated {len(test_suite['files'])} files")
    print(f"🎭 Created {len(test_suite['session_scenarios'])} session scenarios")
    print(f"❓ Created {len(test_suite['query_test_cases'])} query test cases")
    print(f"🔄 Created {len(test_suite['workflow_scenarios'])} workflow scenarios")