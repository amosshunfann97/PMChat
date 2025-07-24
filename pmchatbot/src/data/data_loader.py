import pandas as pd
import os
from config.settings import Config

config = Config()

def load_csv_data():
    """Load and validate CSV data"""
    csv_file_path = config.CSV_FILE_PATH
    
    if not csv_file_path:
        raise ValueError("CSV_FILE_PATH environment variable not set!")
    
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV file not found at {csv_file_path}")
    
    print(f"Loading data from {csv_file_path}...")
    
    # Try reading with semicolon delimiter first, fallback to comma if it fails
    try:
        df = pd.read_csv(csv_file_path, sep=';')
        if df.shape[1] == 1:  # Only one column, likely wrong delimiter
            raise ValueError("Only one column detected, trying comma delimiter.")
    except Exception:
        df = pd.read_csv(csv_file_path, sep=',')
    print(f"Loaded dataset with {len(df)} events")

    # Standardize column types
    if 'case_id' in df.columns:
        df['case_id'] = df['case_id'].astype(str)
    elif 'case:concept:name' in df.columns:
        df['case:concept:name'] = df['case:concept:name'].astype(str)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%y %H:%M')
        df = df.sort_values(['case_id', 'timestamp'])
    elif 'time:timestamp' in df.columns:
        df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
    
    print(f"Dataset contains {df['case_id'].nunique()} unique cases")
    print(f"Activities: {df['activity'].unique()}")
    
    return df

def build_activity_case_map(df):
    """Build mapping of activities to case IDs"""
    from collections import defaultdict
    
    activity_case_map = defaultdict(set)
    for _, row in df.iterrows():
        activity_case_map[row['activity']].add(str(row['case_id']))
    
    # Sort the case IDs for each activity
    return {k: sorted(list(v), key=lambda x: int(x) if x.isdigit() else x) 
            for k, v in activity_case_map.items()}

def list_part_descs(df):
    """List all unique part_desc values in the DataFrame."""
    if 'part_desc' not in df.columns:
        raise ValueError("No 'part_desc' column found in the data.")
    return sorted(df['part_desc'].unique())

def filter_by_part_desc(df, part_desc):
    """Filter the DataFrame by a specific part_desc value."""
    if 'part_desc' not in df.columns:
        raise ValueError("No 'part_desc' column found in the data.")
    return df[df['part_desc'] == part_desc].copy()