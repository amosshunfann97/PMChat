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
        text += f"This activity executed {execution_count} times. "
        
        if execution_count == max_count:
            text += f"This activity has the highest frequency among all activities. "
        if execution_count == min_count:
            text += f"This activity has the lowest frequency among all activities. "
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
        if case_ids:
            text += f"Related Case IDs: {', '.join(case_ids)}. "
            
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