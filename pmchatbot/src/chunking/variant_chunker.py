from collections import defaultdict

def extract_case_variants(df, performance_dfgs, min_cases_per_variant=1):
    """Extract case variants with performance metrics"""
    print(f"Extracting case variants with performance (min cases per variant: {min_cases_per_variant})...")
    case_sequences = {}
    case_metadata = {}
    case_durations = {}

    for case_id, group in df.groupby('case_id'):
        sorted_group = group.sort_values('timestamp')
        sequence = tuple(sorted_group['activity'].tolist())
        timestamps = sorted_group['timestamp'].tolist()
        case_sequences[case_id] = sequence

        case_duration = 0.0
        for i in range(len(sequence) - 1):
            transition = (sequence[i], sequence[i + 1])
            case_duration += performance_dfgs['mean'].get(transition, 0.0)
        case_durations[case_id] = case_duration

        case_metadata[case_id] = {
            'num_activities': len(sequence),
            'unique_activities': len(set(sequence)),
            'duration': case_duration
        }

    variant_groups = defaultdict(list)
    for case_id, sequence in case_sequences.items():
        variant_groups[sequence].append(case_id)

    frequent_variants = {variant: cases for variant, cases in variant_groups.items() 
                        if len(cases) >= min_cases_per_variant}
    sorted_variants = sorted(frequent_variants.items(), key=lambda x: len(x[1]), reverse=True)

    print(f"   Found {len(sorted_variants)} case variants")
    print("   Top 10 most frequent variants with performance:")
    for i, (variant, cases) in enumerate(sorted_variants[:10], 1):
        variant_str = " → ".join(variant)
        avg_duration = sum(case_durations[c] for c in cases) / len(cases)
        print(f"   {i}. {variant_str} ({len(cases)} cases, avg duration: {avg_duration:.2f}s)")

    variant_stats = []
    for variant, cases in sorted_variants:
        durations = [case_durations[c] for c in cases]
        variant_min_duration = 0.0
        variant_max_duration = 0.0
        for i in range(len(variant) - 1):
            transition = (variant[i], variant[i + 1])
            variant_min_duration += performance_dfgs['min'].get(transition, 0.0)
            variant_max_duration += performance_dfgs['max'].get(transition, 0.0)
        stats = {
            'variant': variant,
            'cases': cases,
            'frequency': len(cases),
            'avg_activities': sum(case_metadata[c]['num_activities'] for c in cases) / len(cases),
            'avg_unique_activities': sum(case_metadata[c]['unique_activities'] for c in cases) / len(cases),
            'avg_duration': sum(durations) / len(durations),
            'min_duration': variant_min_duration,
            'max_duration': variant_max_duration,
            'total_duration': sum(durations)
        }
        variant_stats.append(stats)

    return variant_stats

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
            f"This variant consists of {variant_length} activities and takes on average {stats['avg_duration']:.2f} seconds to complete. (min: {stats['min_duration']:.2f} seconds, max: {stats['max_duration']:.2f} seconds). "
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