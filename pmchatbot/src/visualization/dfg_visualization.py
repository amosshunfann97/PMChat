import pm4py
import pandas as pd

def visualize_dfg(dfg, start_activities, end_activities, output_path="dfg_pm4py.png"):
    """Visualize and save the DFG using PM4Py."""
    try:
        pm4py.view_dfg(dfg, start_activities, end_activities)
        pm4py.save_vis_dfg(dfg, start_activities, end_activities, output_path)
        print(f"DFG visualization saved to {output_path}")
    except Exception as e:
        print(f"DFG visualization failed: {e}")

def export_dfg_data(dfg, start_activities, end_activities, output_path="dfg_relationships.csv"):
    """Export DFG data to CSV."""
    dfg_data = []
    for (source, target), count in dfg.items():
        dfg_data.append({
            'source_activity': source,
            'target_activity': target,
            'frequency': count,
            'relationship_type': 'directly_follows'
        })
    for activity, count in start_activities.items():
        dfg_data.append({
            'source_activity': 'START',
            'target_activity': activity,
            'frequency': count,
            'relationship_type': 'start_activity'
        })
    for activity, count in end_activities.items():
        dfg_data.append({
            'source_activity': activity,
            'target_activity': 'END',
            'frequency': count,
            'relationship_type': 'end_activity'
        })
    dfg_df = pd.DataFrame(dfg_data)
    dfg_df.to_csv(output_path, index=False)
    print(f"DFG data exported to {output_path}")