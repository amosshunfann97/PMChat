import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_loader import load_csv_data, list_part_descs, filter_by_part_desc
import pm4py
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.stats import get_start_activities, get_end_activities, get_case_duration
from pm4py.discovery import discover_performance_dfg
from pm4py.util import constants
from pm4py.statistics.variants.log.get import get_variants_along_with_case_durations

def format_duration(seconds):
    years = int(seconds // 31536000)  # 365 days
    months = int((seconds % 31536000) // 2592000)  # 30 days
    days = int((seconds % 2592000) // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    parts = []
    if years > 0:
        parts.append(f"{years} years")
    if months > 0 or years > 0:
        parts.append(f"{months} months")
    if days > 0 or months > 0 or years > 0:
        parts.append(f"{days} days")
    if hours > 0 or days > 0 or months > 0 or years > 0:
        parts.append(f"{hours} hrs")
    if minutes > 0 or hours > 0 or days > 0 or months > 0 or years > 0:
        parts.append(f"{minutes} mins")
    if secs > 0 or not parts:
        parts.append(f"{secs} secs")
    return " ".join(parts)

def main():
    df = load_csv_data()
    # --- PART SELECTION ---
    if 'part_desc' in df.columns:
        parts = list_part_descs(df)
        print(f"Available parts: {parts}")
        while True:
            selected_part = input("Enter part_desc to filter (or leave blank for all): ").strip()
            if not selected_part:
                print("No filtering applied.")
                break
            if selected_part not in parts:
                print(f"Typo detected: '{selected_part}' not found in available parts. Please try again.")
            else:
                df = filter_by_part_desc(df, selected_part)
                print(f"Filtered to part_desc: {selected_part} ({len(df)} events)")
                break

    # Rename columns for PM4Py compatibility
    df = df.rename(columns={
        'case_id': 'case:concept:name',
        'activity': 'concept:name',
        'timestamp': 'time:timestamp'
    })

    # Save a copy of the DataFrame for variant path duration analysis
    df_for_variants = df.copy()

    # Convert to PM4Py event log
    event_log = pm4py.convert_to_event_log(df)

    if not event_log or len(event_log) == 0:
        print("Event log is empty. Nothing to analyze.")
        return

    # Discover DFG
    dfg, freq_start_activities, freq_end_activities = pm4py.discover_dfg(event_log)
        
    # Performance DFG
    performance_dfg, perf_start_activities, perf_end_activities = discover_performance_dfg(event_log, perf_aggregation_key="mean")

    print("=== DFG (frequency, mean seconds and ymdhms) ===")
    for (src, tgt), mean_seconds in performance_dfg.items():
        freq = dfg.get((src, tgt), 'N/A')
        print(f"{src} -> {tgt} (frequency = {freq}  performance = {mean_seconds:.2f} seconds = {format_duration(mean_seconds)})")

    print("\n=== Start Activities ===")
    try:
        print(get_start_activities(event_log))
    except Exception as e:
        print(f"Error getting start activities: {e}")

    print("\n=== End Activities ===")
    try:
        print(get_end_activities(event_log))
    except Exception as e:
        print(f"Error getting end activities: {e}")
        
    print("\n=== Individual Case Durations ===")
    for trace in event_log[:5]:  # First 5 cases
        case_id = trace.attributes['concept:name']
        duration = get_case_duration(event_log, case_id)
        print(f"{case_id}: {duration:.2f} seconds ({format_duration(duration)})")

    print("\n=== Variants with Case Durations ===")
    try:
        variants_dict, durations_dict = get_variants_along_with_case_durations(event_log)
        for i, (variant, traces) in enumerate(variants_dict.items()):
            variant_str = " → ".join(variant)
            durations = durations_dict[variant]
            avg_duration = durations.mean() if len(durations) > 0 else 0.0
            min_duration = durations.min() if len(durations) > 0 else 0.0
            max_duration = durations.max() if len(durations) > 0 else 0.0
            case_ids = [trace.attributes['concept:name'] for trace in traces]
            print(f"Variant {i+1}: {variant_str}")
            print(f"  Cases: {len(traces)}")
            print(f"  Case IDs: {', '.join(str(cid) for cid in case_ids)}")
            print(f"  Avg duration: {avg_duration:.2f} seconds ({format_duration(avg_duration)})")
            print(f"  Min duration: {min_duration:.2f} seconds ({format_duration(min_duration)})")
            print(f"  Max duration: {max_duration:.2f} seconds ({format_duration(max_duration)})")
    except Exception as e:
        print(f"Error getting variants with case durations: {e}")


if __name__ == "__main__":
    main()