from collections import defaultdict

def extract_process_paths(df, performance_dfgs, min_frequency=1):
    """Extract 2-activity process paths from the event log with performance metrics"""
    print(f"Extracting 2-activity process paths with performance (min frequency: {min_frequency})...")
    
    # Group by case and get activity sequences with timestamps
    case_sequences = []
    case_timestamps = []
    for case_id, group in df.groupby('case_id'):
        # Sort by timestamp to get correct order
        sequence = group.sort_values('timestamp')['activity'].tolist()
        timestamps = group.sort_values('timestamp')['timestamp'].tolist()
        case_sequences.append(sequence)
        case_timestamps.append(timestamps)
    
    # Extract only 2-activity paths (direct transitions) with performance
    path_frequencies = defaultdict(int)
    path_durations = defaultdict(list)
    
    for sequence, timestamps in zip(case_sequences, case_timestamps):
        # Extract only paths of length 2 (direct activity transitions)
        for i in range(len(sequence) - 1):
            path = tuple(sequence[i:i + 2])  # Only 2-activity paths
            path_frequencies[path] += 1
            
            # Calculate duration between activities
            duration = (timestamps[i + 1] - timestamps[i]).total_seconds()
            path_durations[path].append(duration)
    
    # Filter by minimum frequency and calculate performance statistics
    frequent_paths = {}
    path_performance = {}
    for path, freq in path_frequencies.items():
        if freq >= min_frequency:
            frequent_paths[path] = freq
            durations = path_durations[path]
            path_performance[path] = {
                'mean': sum(durations) / len(durations),
                'min': min(durations),
                'max': max(durations),
                'count': len(durations)
            }
    
    # Sort by frequency
    sorted_paths = sorted(frequent_paths.items(), key=lambda x: x[1], reverse=True)
    
    print(f"   Found {len(sorted_paths)} frequent 2-activity paths")
    print("   Top 10 most frequent 2-activity transitions with performance:")
    for i, (path, freq) in enumerate(sorted_paths[:10], 1):
        path_str = " → ".join(path)
        perf = path_performance[path]
        print(f"   {i}. {path_str} (frequency: {freq}, avg time: {perf['mean']:.2f}s)")
    
    # Calculate performance metrics for all paths using PM4Py data
    path_performance = {}
    for path, freq in frequent_paths.items():
        src, tgt = path
        path_performance[path] = {
            'mean': performance_dfgs['mean'].get((src, tgt), 0.0),
            'min': performance_dfgs['min'].get((src, tgt), 0.0),
            'max': performance_dfgs['max'].get((src, tgt), 0.0),
            'count': freq
        }
    
    # Sort paths by frequency
    sorted_frequent_paths = sorted(frequent_paths.items(), key=lambda x: x[1], reverse=True)
    
    print(f"   Found {len(sorted_frequent_paths)} frequent 2-activity paths with performance metrics")
    print("   Top 10 most frequent 2-activity transitions with performance metrics:")
    for i, (path, freq) in enumerate(sorted_frequent_paths[:10], 1):
        path_str = " → ".join(path)
        perf = path_performance[path]
        print(f"   {i}. {path_str} (frequency: {freq}, avg time: {perf['mean']:.2f}s, min time: {perf['min']:.2f}s, max time: {perf['max']:.2f}s)")
    
    return sorted_frequent_paths, path_performance

def generate_process_based_chunks(dfg, start_activities, end_activities, frequent_paths, path_performance):
    """Generate process-based chunks for RAG"""
    print("Generating process-based process model chunks with performance metrics...")
    chunks = []
    
    if frequent_paths:
        frequencies = [freq for _, freq in frequent_paths]
        max_freq = max(frequencies)
        min_freq = min(frequencies)
        all_times = [path_performance[path]['mean'] for path, _ in frequent_paths]
        min_time = min(all_times)
        max_time = max(all_times)
    else:
        max_freq = min_freq = None
        min_time = max_time = None

    for i, (path, frequency) in enumerate(frequent_paths):
        path_str = " → ".join(path)
        perf = path_performance[path]
        text = (
            f"Process '{path_str}' is a process that occurs {frequency} times. "
            f"This process takes on average {format_duration(perf['mean'])} to complete. "
            f"(min: {format_duration(perf['min'])}, max: {format_duration(perf['max'])}). "
        )
        # Add this block for loop pattern
        if len(path) == 2 and path[0] == path[1]:
            text += "This is a looping pattern, where the activity repeats within a process instance. "

        if perf['mean'] == min_time:
            text += "This is the fastest process. "
        if perf['mean'] == max_time:
            text += "This is the slowest process. "
            text += "This process is the bottleneck due to its long average execution time. "
            
        total_paths = len(frequent_paths)
        rank = i + 1
        text += f"This is the {rank} most frequent process out of {total_paths} processes. "

        if frequency == max_freq:
            text += f"This process has the highest frequency or execution count. "
        if frequency == min_freq:
            text += f"This process has the lowest frequency or execution count. "

        process_model = {
            "path": path,
            "path_string": path_str,
            "frequency": frequency,
            "length": len(path),
            "rank": rank,
            "performance": perf,
        }
        
        chunks.append({
            "text": text.strip(),
            "type": "process_chunk",
            "path_string": path_str,
            "source": "process_based_chunking",
            "data": process_model
        })
    
    return chunks

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