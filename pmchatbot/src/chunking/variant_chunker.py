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

def generate_variant_based_chunks(dfg, start_activities, end_activities, variant_stats):
    """Generate variant-based chunks for RAG"""
    print("Generating variant-based process model chunks with performance metrics...")
    chunks = []
    total_variants = len(variant_stats)
    total_cases = sum(stats['frequency'] for stats in variant_stats)
    lengths = [len(stats['variant']) for stats in variant_stats]
    max_length = max(lengths)
    min_length = min(lengths)
    durations = [stats['avg_duration'] for stats in variant_stats]
    min_duration = min(durations)
    max_duration = max(durations)

    for i, stats in enumerate(variant_stats):
        variant = stats['variant']
        cases = stats['cases']
        frequency = stats['frequency']
        variant_length = len(variant)
        variant_str = " → ".join(variant)
        text = (
            f"Process variant '{variant_str}' represents an execution pattern found in {frequency} cases. "
            f"This variant consists of {variant_length} activities and takes on average {format_duration(stats['avg_duration'])} to complete. "
            f"(min: {format_duration(stats['min_duration'])}, max: {format_duration(stats['max_duration'])}). "
        )
        if stats['avg_duration'] == min_duration:
            text += "This is the fastest variant. "
        if stats['avg_duration'] == max_duration:
            text += "This is the slowest variant. "
        
        if variant_length == max_length:
            text += "This is the longest variant. "
        if variant_length == min_length:
            text += "This is the shortest variant. "
        
        start_act, end_act = variant[0], variant[-1]
        text += f"This variant starts with {start_act}, ends with {end_act}. "

        rank = i + 1
        percentage = (frequency / total_cases) * 100
        text += f"This variant represents {percentage:.1f}% among all variants. "

        example_cases = cases
        text += f"Cases following this variant: {', '.join(example_cases)}. "
        
        if rank == 1:
            text += f"This is the most common process execution pattern (rank {rank} of {total_variants}). "
        elif rank == total_variants:
            text += f"This is the least common process execution pattern (rank {rank} of {total_variants}). "
        else:
            text += f"This is a mid-ranked variant (rank {rank} of {total_variants}). "
            
        unique_activities = set(variant)
        common_activities = set()
        for other_stats in variant_stats:
            if other_stats != stats:
                common_activities.update(set(other_stats['variant']))
        variant_specific = unique_activities - common_activities
        if variant_specific:
            text += f"This variant includes unique activities not found in other common variants: {', '.join(variant_specific)}. "
            
        variant_model = {
            "variant": variant,
            "variant_string": variant_str,
            "cases": cases,
            "frequency": frequency,
            "percentage": percentage,
            "length": variant_length,
            "rank": rank,
            "performance": {
                'avg_duration': stats['avg_duration'],
                'min_duration': stats['min_duration'],
                'max_duration': stats['max_duration'],
                'total_duration': stats['total_duration']
            },
            "starts_process": variant[0] in start_activities,
            "ends_process": variant[-1] in end_activities,
            "avg_activities": stats['avg_activities'],
            "avg_unique_activities": stats['avg_unique_activities'],
            "unique_activities": list(variant_specific)
        }
        
        chunks.append({
            "text": text.strip(),
            "type": "variant_chunk",
            "variant_string": variant_str,
            "source": "variant_based_chunking",
            "data": variant_model
        })
    
    return chunks