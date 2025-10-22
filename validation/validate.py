import streamlit as st
import json
import pandas as pd # Optional, for displaying results nicely
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()

data_directory = os.getenv("POND_DATA_PATH")
main_directory = os.getenv("POND_PATH")


filename = "../data/pond_results.csv"
outfile = "../data/pond_results_validated.csv"


df = pd.read_csv(filename)
#df = df.sample(n=5, random_state=42).reset_index(drop=True)

data_df = df.loc[:, df.columns != 'context']
data_points = data_df.to_dict(orient='records')
data_points = [{k: v for k, v in d.items()} for d in data_points]
markdown_content = df.loc[:, 'context'].tolist()

# Combine markdown and data into a single list of dictionaries
sample_dataset = [
    {"markdown_text": md, "data": data}
    for md, data in zip(markdown_content, data_points)
]


st.set_page_config(layout="wide", page_title="Dataset Validator")

st.title("Dataset Validation")
st.write("Review each data point and its associated Markdown, then mark it as valid or invalid.")

# Initialize session state for storing results and current index
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'validation_results' not in st.session_state:
    st.session_state.validation_results = []
if 'dataset' not in st.session_state:
    st.session_state.dataset = sample_dataset # Load your actual dataset here

dataset_len = len(st.session_state.dataset)

def record_validation(status):
    """Records the validation status for the current data point."""
    current_item = st.session_state.dataset[st.session_state.current_index]
    st.session_state.validation_results.append(status)
    st.session_state.current_index += 1

def save_results():
    """Saves the validation results to CSV file."""
    df['validation_status'] = st.session_state.validation_results
    df.to_csv(outfile, index=False)

# Track results
valid = 0
hallucinated = 0
mis_interpreted = 0
parsing_errors = 0
skipped = 0

# --- Display Current Data Point ---
if st.session_state.current_index < dataset_len:
    current_item = st.session_state.dataset[st.session_state.current_index]

    st.subheader(f"Data Point {st.session_state.current_index + 1} of {dataset_len}")

    st.write("### Is this data point valid?")
    col_btns1, col_btns2, col_btns3, col_btns4, col_btns5 = st.columns(5)
    with col_btns1:
        if st.button("✅ Valid", key="valid_btn"):
            record_validation("valid")
            valid += 1
            st.rerun()
    with col_btns2:
        if st.button("Hallucination", key="hallucination_btn"):
            record_validation("hallucination")
            hallucinated += 1
            st.rerun()
    with col_btns3:
        if st.button("Mis-interpretation", key="inter_btn"):
            record_validation("mis-interpretation")
            mis_interpreted += 1
            st.rerun()
    with col_btns3:
        if st.button("Parsing Error", key="parse_btn"):
            record_validation("parsing_error")
            parsing_errors += 1
            st.rerun()
    with col_btns5:
        if st.button("⏭️ Skip", key="skip_btn"):
            record_validation(None)
            skipped += 1
            st.rerun()

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.warning("Markdown Text:")
        st.markdown(current_item['markdown_text'])

    with col2:
        st.info("Extracted Data:")
        st.json(current_item['data'])

    st.markdown("---")


else:
    st.success("All data points have been reviewed!")
    st.write("### Summary of Validation Results")
    #st.write(f"Data points matched to ground truth and removed before validation: {n_matched}")
    st.write(f"Total Data Points Reviewed: {len(st.session_state.validation_results)}")
    st.write(f"✅ Valid: {st.session_state.validation_results.count('valid')}")
    st.write(f"❌ Hallucinations: {st.session_state.validation_results.count('hallucination')}")
    st.write(f"❌ Mis-interpretations: {st.session_state.validation_results.count('mis-interpretation')}")
    st.write(f"❌ Parsing Errors: {st.session_state.validation_results.count('parsing_error')}")
    st.write(f"⏭️ Skipped: {st.session_state.validation_results.count(None)}")

    if st.session_state.validation_results:
        # save to CSV
        save_results()
        st.write(f"Validation results saved to `{outfile}`.")
    else:
        st.write("No validation decisions were made.")

    if st.button("Start Over"):
        st.session_state.current_index = 0
        st.session_state.validation_results = []
        st.rerun()

####################################################################################################