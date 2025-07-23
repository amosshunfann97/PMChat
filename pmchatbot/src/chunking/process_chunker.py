from collections import defaultdict

def extract_process_paths(dfg, performance_dfgs, min_frequency=1):
    """Extract 2-activity process paths from the DFG with performance metrics"""
    print(f"Extracting 2-activity process paths with performance (min frequency: {min_frequency})...")
    
    # Use DFG directly for frequencies (same as activity chunker)
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
    
    print(f"   Found {len(sorted_frequent_paths)} frequent 2-activity paths with performance metrics")
    print("   Top 10 most frequent 2-activity transitions with performance metrics:")
    for i, (path, freq) in enumerate(sorted_frequent_paths[:10], 1):
        path_str = " → ".join(path)
        perf = path_performance[path]
        print(f"   {i}. {path_str} (frequency: {freq}, avg time: {perf['mean']:.2f}s, min time: {perf['min']:.2f}s, max time: {perf['max']:.2f}s)")
    
    return sorted_frequent_paths, path_performance

def generate_process_based_chunks(frequent_paths, path_performance):
    """Generate process-based chunks for RAG with self-loop detection"""
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

    # Self-loop detection and ranking
    self_loops = {path: freq for path, freq in frequent_paths if path[0] == path[1] and freq > 0}
    self_loops_sorted = sorted(self_loops.items(), key=lambda x: x[1], reverse=True)
    total_self_loops = len(self_loops_sorted)

    for i, (path, frequency) in enumerate(frequent_paths):
        path_str = " → ".join(path)
        perf = path_performance[path]
        text = (
            f"Process '{path_str}' is a process that occurs {frequency} times. "
            f"This process takes on average {format_duration(perf['mean'])} to complete. "
            f"(min: {format_duration(perf['min'])}, max: {format_duration(perf['max'])}). "
        )
        # Self-loop detection and ranking
        if len(path) == 2 and path[0] == path[1]:
            # Find rank of current self-loop among all self-loops
            rank = next((idx + 1 for idx, (loop_path, _) in enumerate(self_loops_sorted) if loop_path == path), None)
            text += (
                f"It has a self-loop pattern, directly followed by itself {frequency} times "
                f"(rank {rank}/{total_self_loops} self-loop transitions). "
            )

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