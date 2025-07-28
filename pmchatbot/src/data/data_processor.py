import pm4py
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.statistics.variants.log import get as variants_get
from utils.logging_utils import log
from pm4py.stats import get_rework_cases_per_activity

def prepare_pm4py_log(df):
    """Convert DataFrame to PM4Py event log"""
    log_df = df.copy()
    log_df = log_df.rename(columns={
        'case_id': 'case:concept:name',
        'activity': 'concept:name',
        'timestamp': 'time:timestamp'
    })
    event_log = pm4py.convert_to_event_log(log_df)
    return event_log

def discover_process_model(event_log):
    """Discover process model and performance metrics using PM4Py"""
    log("Discovering DFG...", level="info")
    dfg, start_activities, end_activities = pm4py.discover_dfg(event_log)

    log("Discovering performance DFG...", level="info")
    performance_dfg_mean, _, _ = pm4py.discover_performance_dfg(event_log, perf_aggregation_key="mean")
    performance_dfg_min, _, _ = pm4py.discover_performance_dfg(event_log, perf_aggregation_key="min")
    performance_dfg_max, _, _ = pm4py.discover_performance_dfg(event_log, perf_aggregation_key="max")

    performance_dfgs = {
        'mean': performance_dfg_mean,
        'min': performance_dfg_min,
        'max': performance_dfg_max
    }

    # Compute rework cases per activity
    try:
        rework_cases = get_rework_cases_per_activity(event_log)
    except Exception as e:
        log(f"Error computing rework cases per activity: {e}", level="error")
        rework_cases = {}

    log(f"Process discovered:", level="debug")
    log(f"   - DFG edges: {len(dfg)}", level="debug")
    log(f"   - Performance DFG edges (mean): {len(performance_dfg_mean)}", level="debug")
    log(f"   - Performance DFG edges (min): {len(performance_dfg_min)}", level="debug")
    log(f"   - Performance DFG edges (max): {len(performance_dfg_max)}", level="debug")
    log(f"   - Start activities: {list(start_activities.keys())}", level="debug")
    log(f"   - End activities: {list(end_activities.keys())}", level="debug")
    log(f"   - Activities with rework: {list(rework_cases.keys())}", level="debug")

    return dfg, start_activities, end_activities, performance_dfgs, rework_cases

def extract_case_variants(event_log, min_cases_per_variant=1):
    """
    Extract case variants and their durations using PM4Py event log.
    """

    variants_dict, durations_dict = variants_get.get_variants_along_with_case_durations(event_log)

    # Convert variants_dict keys to tuple for consistency
    variant_groups = {tuple(variant): [trace.attributes.get("concept:name", str(idx))
                                       for idx, trace in enumerate(traces)]
                      for variant, traces in variants_dict.items()}

    frequent_variants = {variant: cases for variant, cases in variant_groups.items()
                         if len(cases) >= min_cases_per_variant}
    sorted_variants = sorted(frequent_variants.items(), key=lambda x: len(x[1]), reverse=True)

    log(f"   Found {len(sorted_variants)} case variants", level="debug")
    log("   Top 3 most frequent variants:", level="debug")
    for i, (variant, cases) in enumerate(sorted_variants[:3], 1):
        variant_str = " → ".join(variant)
        avg_duration = durations_dict[variant].mean() if len(durations_dict[variant]) > 0 else 0.0
        log(f"   {i}. {variant_str} ({len(cases)} cases, avg duration: {avg_duration:.2f}s)", level="debug")

    case_durations = case_statistics.get_all_case_durations(event_log)
    log(str(case_durations), level="debug")

    variant_stats = []
    for variant, cases in sorted_variants:
        durations = durations_dict[variant]
        avg_duration = durations.mean() if len(durations) > 0 else 0.0
        min_duration = durations.min() if len(durations) > 0 else 0.0
        max_duration = durations.max() if len(durations) > 0 else 0.0
        total_duration = durations.sum() if len(durations) > 0 else 0.0
        stats = {
            'variant': variant,
            'cases': cases,
            'frequency': len(cases),
            'avg_activities': len(variant),
            'avg_unique_activities': len(set(variant)),
            'avg_duration': avg_duration,
            'min_duration': min_duration,
            'max_duration': max_duration,
            'total_duration': total_duration
        }
        variant_stats.append(stats)

    return variant_stats

def extract_process_paths(dfg, performance_dfgs, min_frequency=1):
    """Extract 2-activity process paths from the DFG with performance metrics"""
    
    frequent_paths = {}
    path_performance = {}

    for (src, tgt), count in dfg.items():
        if count >= min_frequency:
            path = (src, tgt)
            frequent_paths[path] = count

            # Get performance metrics from performance_dfgs
            path_performance[path] = {
                'mean': performance_dfgs['mean'].get((src, tgt), 0.0),
                'min': performance_dfgs['min'].get((src, tgt), 0.0),
                'max': performance_dfgs['max'].get((src, tgt), 0.0),
                'count': count
            }

    # Sort paths by frequency
    sorted_frequent_paths = sorted(frequent_paths.items(), key=lambda x: x[1], reverse=True)

    log(f"   Found {len(sorted_frequent_paths)} process paths", level="debug")
    log("   Top 3 most frequent 2-activity transitions:", level="debug")
    for i, (path, freq) in enumerate(sorted_frequent_paths[:3], 1):
        path_str = " → ".join(path)
        perf = path_performance[path]
        log(f"   {i}. {path_str} (frequency: {freq}, avg time: {perf['mean']:.2f}s, min time: {perf['min']:.2f}s, max time: {perf['max']:.2f}s)", level="debug")

    return sorted_frequent_paths, path_performance