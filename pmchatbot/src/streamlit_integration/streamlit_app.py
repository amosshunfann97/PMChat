import streamlit as st
from pmchatbot.src.data.data_loader import load_csv_data, list_part_descs, filter_by_part_desc

st.title("Process Mining GraphRAG Frontend")

try:
    df = load_csv_data()
    part_options = list_part_descs(df)
    selected_part = st.selectbox("Select a part", part_options)
    st.write(f"Selected part: {selected_part}")
    
    filtered_df = filter_by_part_desc(df, selected_part)
    # For example, display the filtered dataframe:
    st.dataframe(filtered_df)
except Exception as e:
    st.error(str(e))