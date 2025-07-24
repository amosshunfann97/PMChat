import pm4py

def format_duration(seconds):
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    parts = []
    if days > 0:
        parts.append(f"{days} days")
    if hours > 0 or days > 0:
        parts.append(f"{hours} hrs")
    if minutes > 0 or hours > 0 or days > 0:
        parts.append(f"{minutes} mins")
    parts.append(f"{secs} secs")
    return " ".join(parts)

def preprocess_performance_dfg(performance_dfg):
    # Convert all durations to formatted strings
    return {k: format_duration(v) for k, v in performance_dfg.items()}

def visualize_performance_dfg(performance_dfg, start_activities, end_activities, output_path="performance_dfg_pm4py.png"):
    """Visualize and save the performance DFG as PNG using PM4Py."""
    try:
        # Pass the original numeric values, not formatted strings
        pm4py.view_performance_dfg(performance_dfg, start_activities, end_activities)
        pm4py.save_vis_performance_dfg(performance_dfg, start_activities, end_activities, output_path)
        print(f"Performance DFG visualization saved to {output_path}")
    except Exception as e:
        print(f"Performance DFG visualization failed: {e}")