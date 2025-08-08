"""
data_upload.py - Data Upload Module
===================================
Handles file upload (CSV/Excel), parsing, type detection, and metadata generation.
Saves data and column metadata for downstream use in the app.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os


def app():
    """
    Main function for the 'Upload Data' page.
    Allows users to upload a CSV or Excel file, previews it, detects column types,
    and saves both the cleaned data and metadata for later stages.
    """
    st.title("📁 Upload Data")
    st.markdown("### Upload your dataset for analysis")
    st.write("Supports **CSV** and **Excel** (`.xlsx`) files.")

    # --- File Uploader ---
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx"],
        help="Upload a structured data file (CSV or Excel) to begin."
    )

    if uploaded_file is None:
        st.info("Please upload a file to proceed.")
        st.stop()

    # --- Load Data Based on File Type ---
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            st.stop()
    except Exception as e:
        st.error("Failed to read the uploaded file.")
        st.code(f"Error: {e}")
        st.stop()

    # Validate non-empty data
    if df.empty:
        st.warning("The uploaded file is empty.")
        st.stop()

    # --- Display Data Preview ---
    st.markdown("### 📊 Data Preview")
    st.dataframe(df.head(100), use_container_width=True)

    st.markdown(f"**Shape:** `{df.shape[0]}` rows × `{df.shape[1]}` columns")

    # --- Detect Column Types ---
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Handle columns not caught by dtype detection (e.g., date-like strings)
    all_cols = set(df.columns)
    detected_cols = set(numeric_cols + categorical_cols)
    remaining_cols = all_cols - detected_cols

    # Classify remaining columns as categorical by default
    for col in remaining_cols:
        categorical_cols.append(col)

    # Generate metadata
    columns = []
    for col in df.columns:
        if col in numeric_cols:
            col_type = "numerical"
        elif col in categorical_cols:
            col_type = "categorical"
        else:
            col_type = "categorical"  # fallback
        columns.append({"column_name": col, "type": col_type})

    columns_df = pd.DataFrame(columns)

    # --- Save Data and Metadata ---
    try:
        # Create directories if they don't exist
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/metadata", exist_ok=True)

        # Save raw data
        df.to_csv("data/main_data.csv", index=False)

        # Save metadata
        columns_df.to_csv("data/metadata/column_type_desc.csv", index=False)
    except Exception as e:
        st.error("Failed to save data or metadata.")
        st.code(f"Save Error: {e}")
        st.stop()

    # --- Display Detected Column Types ---
    st.markdown("### 🔍 Detected Column Types")
    for i, row in enumerate(columns_df.itertuples()):
        st.markdown(f"**{i+1}. `{row.column_name}`** — `{row.type}`")

    st.markdown("""
    These types are auto-detected.  
    You can modify them in the **Change Metadata** section.
    """)

    # --- Success Confirmation ---
    st.success("✅ Data uploaded and metadata saved successfully!")
    st.session_state["data_uploaded"] = True