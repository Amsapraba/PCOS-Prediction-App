import streamlit as st
import pandas as pd

# Set page title and layout
st.set_page_config(page_title="PCOS Data Viewer", layout="wide")

# Title
st.title("PCOS Dataset Viewer")

# File uploader
uploaded_file = st.file_uploader("Upload your PCOS dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)

    # Display dataset
    st.subheader("Full Dataset")
    st.dataframe(df)  # Displays full dataset with scrollbars

    # Show basic statistics
    st.subheader("Basic Statistics")
    st.write(df.describe())

    # Show column info
    st.subheader("Dataset Information")
    buffer = []
    df.info(buf=buffer)
    info_str = "\n".join([str(line) for line in buffer])
    st.text(info_str)

else:
    st.warning("Please upload a CSV file to proceed.")
