import pm4py

def prepare_pm4py_log(df):
    """Convert DataFrame to PM4Py event log"""
    log_df = df.copy()
    log_df = log_df.rename(columns={
        'case_id': 'case:concept:name',
        'activity': 'concept:name',
        'timestamp': 'time:timestamp'
    })
    log = pm4py.convert_to_event_log(log_df)
    return log

def discover_process_model(log):
    """Discover process model and performance metrics using PM4Py"""
    print("Discovering process model...")
    dfg, start_activities, end_activities = pm4py.discover_dfg(log)
    
    print("Discovering performance DFG...")
    performance_dfg_mean, _, _ = pm4py.discover_performance_dfg(log, perf_aggregation_key="mean")
    performance_dfg_min, _, _ = pm4py.discover_performance_dfg(log, perf_aggregation_key="min")
    performance_dfg_max, _, _ = pm4py.discover_performance_dfg(log, perf_aggregation_key="max")
    
    performance_dfgs = {
        'mean': performance_dfg_mean,
        'min': performance_dfg_min,
        'max': performance_dfg_max
    }
    
    print(f"Process discovered:")
    print(f"   - DFG edges: {len(dfg)}")
    print(f"   - Performance DFG edges (mean): {len(performance_dfg_mean)}")
    print(f"   - Performance DFG edges (min): {len(performance_dfg_min)}")
    print(f"   - Performance DFG edges (max): {len(performance_dfg_max)}")
    print(f"   - Start activities: {list(start_activities.keys())}")
    print(f"   - End activities: {list(end_activities.keys())}")
    
    return dfg, start_activities, end_activities, performance_dfgs