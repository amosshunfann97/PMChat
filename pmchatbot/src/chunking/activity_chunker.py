def generate_activity_based_chunks(dfg, start_activities, end_activities, activity_case_map):
    """Generate activity-based chunks for RAG"""
    print("Generating activity-based process model chunks...")
    chunks = []
    all_activities = set()
    
    for (src, tgt), count in dfg.items():
        all_activities.add(src)
        all_activities.add(tgt)

    execution_counts = {}
    for activity in all_activities:
        incoming = [(src, count) for (src, tgt), count in dfg.items() if tgt == activity]
        outgoing = [(tgt, count) for (src, tgt), count in dfg.items() if src == activity]
        total_incoming = sum(count for _, count in incoming)
        total_outgoing = sum(count for _, count in outgoing)
        
        if activity in start_activities:
            execution_count = total_outgoing if total_outgoing > 0 else start_activities[activity]
        elif activity in end_activities:
            execution_count = total_incoming if total_incoming > 0 else end_activities[activity]
        else:
            execution_count = max(total_incoming, total_outgoing) if total_incoming > 0 or total_outgoing > 0 else 0
        execution_counts[activity] = execution_count

    max_count = max(execution_counts.values())
    min_count = min(execution_counts.values())

    # Calculate self-loop activities and their counts
    self_loops = {a: c for (a, b), c in dfg.items() if a == b and c > 0}
    # Sort activities by self-loop count descending
    self_loops_sorted = sorted(self_loops.items(), key=lambda x: x[1], reverse=True)
    total_self_loops = len(self_loops_sorted)

    for activity in all_activities:
        incoming = [(src, count) for (src, tgt), count in dfg.items() if tgt == activity]
        outgoing = [(tgt, count) for (src, tgt), count in dfg.items() if src == activity]
        
        total_incoming = sum(count for _, count in incoming)
        total_outgoing = sum(count for _, count in outgoing)
        
        if activity in start_activities:
            execution_count = total_outgoing if total_outgoing > 0 else start_activities[activity]
        elif activity in end_activities:
            execution_count = total_incoming if total_incoming > 0 else end_activities[activity]
        else:
            execution_count = max(total_incoming, total_outgoing) if total_incoming > 0 or total_outgoing > 0 else 0
        execution_counts[activity] = execution_count
        

        text = f"{activity} is an activity in this workflow. "
        
        if execution_count == max_count:
            text += f"This activity has the highest/most frequency or execution count among all activities. "
        if execution_count == min_count:
            text += f"This activity has the lowest/least frequency or execution count among all activities. "
            
        text += f"This activity executed {execution_count} times. "
        
        if activity in start_activities:
            text += f"{activity} is a starting activity that begins the workflow ({start_activities[activity]} cases start here). "
        if activity in end_activities:
            text += f"{activity} is an ending activity that concludes the workflow ({end_activities[activity]} cases end here). "
        
        if incoming:
            incoming_desc = [f"{src} ({count} times)" for src, count in incoming]
            text += f"{activity} is preceded by: {', '.join(incoming_desc)}. "
        if outgoing:
            outgoing_desc = [f"{tgt} ({count} times)" for tgt, count in outgoing]
            text += f"{activity} is followed by: {', '.join(outgoing_desc)}. "
            
        case_ids = activity_case_map.get(activity, [])
        text += f"This activity appears in {len(case_ids)} cases. "
        # if case_ids:
        #     text += f"Case IDs linked to this activity: {', '.join(case_ids)}. "
        
        # Self-loop detection
        self_loop_count = dfg.get((activity, activity), 0)
        if self_loop_count > 0:
            # Find rank of current activity among self-loop activities
            rank = next((i + 1 for i, (act, _) in enumerate(self_loops_sorted) if act == activity), None)
            text += (
                f"{activity} has a self-loop, directly followed by itself {self_loop_count} times "
                f"(rank {rank}/{total_self_loops} activities with rework within itself). "
            )

        activity_model = {
            "activity": activity,
            "incoming": incoming,
            "outgoing": outgoing,
            "is_start": activity in start_activities,
            "is_end": activity in end_activities,
            "start_frequency": start_activities.get(activity, 0),
            "end_frequency": end_activities.get(activity, 0),
            "execution_count": execution_count,
            "case_ids": case_ids
        }
        
        chunks.append({
            "text": text.strip(),
            "type": "activity_chunk",
            "activity_name": activity,
            "source": "activity_based_chunking",
            "data": activity_model
        })
    
    return chunks