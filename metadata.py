"""
metadata.py - Metadata & Data Cleaning (Fixed, Proper Apply Flow)
==================================================
- Edit column types
- Configure cleaning rules
- Click 'Apply Cleaning' to process
- See Before/After
- Save cleaned data
"""

import streamlit as st
import pandas as pd
import numpy as np
import os


def count_issues(series):
    """Count NaN and zero values."""
    n_null = series.isna().sum()
    n_zero = (series == 0).sum() if pd.api.types.is_numeric_dtype(series) else 0
    return n_null, n_zero


def clean_column(series, method, custom_value=None):
    """Clean column with specified method."""
    s = series.copy()

    if method == "skip" or method == "drop":
        return s
    elif method == "mean" and pd.api.types.is_numeric_dtype(s):
        return s.fillna(s.mean())
    elif method == "median" and pd.api.types.is_numeric_dtype(s):
        return s.fillna(s.median())
    elif method == "mode":
        mode_val = s.mode()
        fill_val = mode_val.iloc[0] if len(mode_val) > 0 else None
        return s.fillna(fill_val)
    elif method == "custom":
        return s.fillna(custom_value)
    else:
        return s


def app():
    st.title("🔧 Metadata & Data Cleaning")

    # --- Check Files ---
    data_path = "data/main_data.csv"
    metadata_path = "data/metadata/column_type_desc.csv"

    if not os.path.exists(data_path):
        st.warning("No data found. Go to **Upload Data** to load a dataset.")
        st.stop()

    if not os.path.exists(metadata_path):
        st.warning("Metadata not found. Please reload your data.")
        st.stop()

    # --- Load Data ---
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        st.error("Failed to load dataset.")
        st.code(e)
        st.stop()

    try:
        metadata = pd.read_csv(metadata_path)
    except Exception as e:
        st.error("Failed to load metadata.")
        st.code(e)
        st.stop()

    st.markdown(f"**Dataset:** `{df.shape[0]}` × `{df.shape[1]}`")
    st.dataframe(df.head(100), use_container_width=True)

    # --- 1. Edit Column Types ---
    st.markdown("### 🛠️ 1. Edit Column Types")
    col1, col2 = st.columns([2, 1])

    with col1:
        selected_col = st.selectbox(
            "Select Column to Edit",
            options=metadata["column_name"].tolist(),
            key="edit_col_type"
        )

    with col2:
        curr_type = metadata.loc[metadata["column_name"] == selected_col, "type"].values[0]
        new_type = st.selectbox(
            "New Type",
            options=["numerical", "categorical"],
            index=0 if curr_type == "numerical" else 1,
            key="new_type_select"
        )

    if st.button("✅ Update Column Type", key="update_type_btn"):
        metadata.loc[metadata["column_name"] == selected_col, "type"] = new_type
        metadata.to_csv(metadata_path, index=False)
        st.success(f"✅ '{selected_col}' updated to '{new_type}'")
        st.rerun()  # Refresh to reflect change

    st.markdown("---")

    # --- 2. Data Cleaning Setup ---
    st.markdown("### 🧼 2. Configure Data Cleaning")

    # Initialize session state for cleaning config
    if "clean_config" not in st.session_state:
        st.session_state.clean_config = {}

    col_types = metadata.set_index("column_name")["type"].to_dict()

    for col in df.columns:
        null_count, zero_count = count_issues(df[col])
        if null_count == 0 and zero_count == 0:
            continue  # Skip if no issues

        col_type = col_types.get(col, "categorical")

        with st.expander(f"🧹 Clean `{col}` | Missing: {null_count} | Zero: {zero_count}"):
            col1, col2 = st.columns([3, 2])

            if col_type == "numerical":
                options = ["skip", "mean", "median", "mode", "custom", "drop"]
                default = 1  # mean
            else:
                options = ["skip", "mode", "custom", "drop"]
                default = 1  # mode

            # Load saved choice or default
            current = st.session_state.clean_config.get(col, {"method": "skip", "custom": ""})

            method = col1.selectbox(
                f"Action for `{col}`",
                options,
                index=options.index(current["method"]) if current["method"] in options else default,
                key=f"method_{col}"
            )

            custom_value = ""
            if method == "custom":
                custom_value = col2.text_input(
                    "Custom value",
                    value=current["custom"],
                    key=f"custom_{col}"
                )

            # Save to session state
            st.session_state.clean_config[col] = {
                "method": method,
                "custom": custom_value
            }

    if not st.session_state.clean_config:
        st.info("✅ No missing or zero values found — your data is already clean!")
        st.stop()

    st.markdown("---")

    # --- 3. Apply Cleaning Button ---
    if st.button("🧹 Apply Cleaning", type="primary", key="apply_clean_btn"):
        with st.spinner("Cleaning data..."):
            cleaned_df = df.copy()
            drop_rows_cols = []

            # Apply cleaning per column
            for col, config in st.session_state.clean_config.items():
                method = config["method"]
                custom_val = config["custom"]

                if method == "drop":
                    drop_rows_cols.append(col)
                else:
                    cleaned_df[col] = clean_column(
                        cleaned_df[col],
                        method,
                        custom_val if method == "custom" else None
                    )

            # Drop rows with missing in 'drop' columns
            if drop_rows_cols:
                cleaned_df = cleaned_df.dropna(subset=drop_rows_cols)

            # Save to session state
            st.session_state.cleaned_df = cleaned_df

            st.success("✅ Cleaning applied!")
            st.session_state.cleaning_applied = True

    # --- 4. Before & After Preview (Only After Apply) ---
    if hasattr(st.session_state, "cleaning_applied") and st.session_state.cleaning_applied:
        st.markdown("### 🔍 3. Before vs After Cleaning")

        colA, colB = st.columns(2)
        colA.markdown("#### 🟦 Before")
        colA.dataframe(df.head(25), use_container_width=True)

        colB.markdown("#### 🟩 After")
        cleaned_df = st.session_state.cleaned_df
        colB.dataframe(cleaned_df.head(25), use_container_width=True)

        # Metrics
        col1, col2 = st.columns(2)
        col1.metric("Before Rows", df.shape[0])
        col2.metric("After Rows", cleaned_df.shape[0])

        # --- Save Cleaned Data ---
        st.markdown("---")
        if st.button("💾 Save Cleaned Data", key="save_cleaned_btn"):
            os.makedirs("data", exist_ok=True)
            cleaned_df.to_csv("data/cleaned_data.csv", index=False)
            st.success("✅ Cleaned data saved to `data/cleaned_data.csv`")

            # Download button
            csv = cleaned_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download Cleaned Data",
                data=csv,
                file_name="cleaned_data.csv",
                mime="text/csv",
                key="dl_cleaned_csv"
            )

    # --- Final Note ---
    st.markdown("---")
    st.markdown(
        "<small>All changes are local. Raw data remains untouched in `data/main_data.csv`.</small>",
        unsafe_allow_html=True
    )