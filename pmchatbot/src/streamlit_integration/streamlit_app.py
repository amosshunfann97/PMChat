import streamlit as st
from pmchatbot.src.data.data_loader import load_csv_data, list_part_descs, filter_by_part_desc
from pmchatbot.src.data.data_processor import prepare_pm4py_log, discover_process_model, extract_case_variants, extract_process_paths
from pmchatbot.src.chunking.activity_chunker import generate_activity_based_chunks
from pmchatbot.src.chunking.process_chunker import generate_process_based_chunks
from pmchatbot.src.chunking.variant_chunker import generate_variant_based_chunks
from pmchatbot.src.embeddings.local_embedder import get_local_embedder
from pmchatbot.src.database.neo4j_manager import connect_neo4j
from pmchatbot.src.database.data_storage import store_chunks_in_neo4j
from pmchatbot.src.visualization.dfg_visualization import visualize_dfg
from pmchatbot.src.visualization.performance_dfg_visualization import visualize_performance_dfg

st.title("Process Mining GraphRAG Frontend")

try:
    df = load_csv_data()
    part_options = list_part_descs(df)
    selected_part = st.selectbox("Select a part", part_options)

    if selected_part:
        filtered_df = filter_by_part_desc(df, selected_part)

        if st.button("Process Selected Part"):
            with st.spinner("Processing and loading chunks..."):
                event_log = prepare_pm4py_log(filtered_df)
                dfg, start_activities, end_activities, performance_dfgs, rework_cases = discover_process_model(event_log)
                
                # Terminal logs for process discovery
                print("DFG Edges:", len(dfg))
                print("Start Activities:", len(start_activities))
                print("End Activities:", len(end_activities))
                print("Performance DFGs:", len(performance_dfgs))
                print("Rework Cases:", len(rework_cases))

                # Activity-based chunking
                activity_case_map = filtered_df.groupby('activity')['case_id'].apply(list).to_dict()
                activity_chunks = generate_activity_based_chunks(dfg, start_activities, end_activities, activity_case_map, rework_cases)

                # Process-based chunking
                frequent_paths, path_performance = extract_process_paths(dfg, performance_dfgs, min_frequency=1)
                process_chunks = generate_process_based_chunks(frequent_paths, path_performance)

                # Variant-based chunking
                variant_stats = extract_case_variants(event_log, min_cases_per_variant=1)
                variant_chunks = generate_variant_based_chunks(dfg, start_activities, end_activities, variant_stats)

                # Terminal logs for chunking
                print(f"Activity Chunks: {len(activity_chunks)}")
                print(f"Process Chunks: {len(process_chunks)}")
                print(f"Variant Chunks: {len(variant_chunks)}")

                # Store chunks in Neo4j
                driver = connect_neo4j()
                local_embedder = get_local_embedder(device="cpu")
                if driver and local_embedder:
                    store_chunks_in_neo4j(
                        driver, dfg, start_activities, end_activities,
                        activity_chunks, process_chunks, variant_chunks,
                        frequent_paths, variant_stats, local_embedder
                    )
                    print("Chunks stored in Neo4j successfully.")
                else:
                    print("Neo4j connection or local embedder failed.")

            st.success("Process mining, chunking, and Neo4j storage complete! Visualizing DFGs...")

            # Visualize DFG and Performance DFG
            with st.spinner("Generating DFG visualizations..."):
                visualize_dfg(dfg, start_activities, end_activities, output_path="dfg_pm4py.png")
                visualize_performance_dfg(performance_dfgs['mean'], start_activities, end_activities, output_path="performance_dfg_pm4py.png")
                st.image("dfg_pm4py.png", caption="Directly-Follows Graph (DFG)")
                st.image("performance_dfg_pm4py.png", caption="Performance DFG (Mean Duration)")

except Exception as e:
    st.error(str(e))