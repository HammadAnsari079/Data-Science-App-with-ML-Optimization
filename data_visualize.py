"""
data_visualize.py - Data Analysis & Visualization Module
========================================================
Generates interactive visualizations including:
- Pie charts for categorical distributions
- Correlation heatmap for numerical features
- Grouped bar charts for category-specific trends
- Descriptive statistics by category

All outputs are based on the uploaded data and saved metadata.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def app():
    """
    Main function for the 'Data Analysis' page.
    Loads data, detects column types, and provides interactive visualizations
    including pie charts, correlation matrices, and grouped bar charts.
    """
    st.title("📊 Data Analysis")

    # --- Check if Data Exists ---
    data_path = "data/main_data.csv"
    metadata_path = "data/metadata/column_type_desc.csv"

    if not os.path.exists(data_path):
        st.warning("No data found. Please go to **Upload Data** to load a dataset.")
        st.stop()

    if not os.path.exists(metadata_path):
        st.warning("Metadata not found. Please reload your data in **Upload Data**.")
        st.stop()

    # --- Load Data and Metadata ---
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        st.error("Failed to load the dataset.")
        st.code(f"Error: {e}")
        st.stop()

    try:
        metadata = pd.read_csv(metadata_path)
    except Exception as e:
        st.error("Failed to load column metadata.")
        st.code(f"Error: {e}")
        st.stop()

    if df.empty:
        st.info("The dataset is empty.")
        st.stop()

    # --- Extract Column Types ---
    def get_column_types(meta_df):
        categorical = meta_df[meta_df["type"] == "categorical"]["column_name"].tolist()
        numerical = meta_df[meta_df["type"] == "numerical"]["column_name"].tolist()
        # Treat any unknown type as categorical
        unknown = meta_df[~meta_df["type"].isin(["categorical", "numerical"])]["column_name"].tolist()
        categorical.extend(unknown)
        return categorical, numerical

    categorical_cols, numerical_cols = get_column_types(metadata)

    if not numerical_cols:
        st.warning("No numerical columns available for analysis.")
        st.info("Consider updating column types in **Change Metadata**.")
        st.stop()

    if not categorical_cols:
        st.warning("No categorical columns available.")
        st.info("Some visualizations may be limited.")
        # Continue anyway for correlation

    # --- Categorical Distribution Pie Chart ---
    st.markdown("### 🥧 Categorical Distribution")
    if categorical_cols:
        selected_cat = st.selectbox("Choose a categorical column", categorical_cols)

        value_counts = df[selected_cat].value_counts(dropna=False)
        sizes = value_counts.values
        labels = value_counts.index.astype(str)

        # Highlight the largest slice
        max_idx = np.argmax(sizes)
        explode = tuple(0.1 if i == max_idx else 0 for i in range(len(sizes)))

        fig1, ax1 = plt.subplots(figsize=(6, 6))
        ax1.pie(
            sizes,
            explode=explode,
            labels=labels,
            autopct="%1.1f%%",
            shadow=True,
            startangle=90,
            textprops={"fontsize": 10}
        )
        ax1.axis("equal")  # Equal aspect ratio
        ax1.set_title(f"Distribution of '{selected_cat}'", fontsize=14, pad=20)
        st.pyplot(fig1)
    else:
        st.info("No categorical columns to visualize.")

    # --- Correlation Heatmap ---
    st.markdown("### 🔗 Numerical Correlation Matrix")
    numeric_df = df[numerical_cols]
    if numeric_df.shape[1] < 2:
        st.info("Need at least 2 numerical columns for correlation.")
    else:
        try:
            corr = numeric_df.corr(method="pearson")

            fig2, ax2 = plt.subplots(figsize=(8, 6))
            mask = np.triu(np.ones_like(corr, dtype=bool))
            cmap = sns.diverging_palette(240, 10, as_cmap=True)

            sns.heatmap(
                corr,
                mask=mask,
                annot=True,
                fmt=".2f",
                cmap=cmap,
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                ax=ax2
            )
            ax2.set_title("Pearson Correlation Heatmap", fontsize=16, pad=20)
            st.pyplot(fig2)
        except Exception as e:
            st.error("Could not compute correlation matrix.")
            st.code(f"Error: {e}")

    # --- Grouped Analysis (if categorical column selected) ---
    if categorical_cols:
        st.markdown("### 📈 Grouped Numerical Analysis")

        col1, col2 = st.columns(2)

        with col1:
            category = st.selectbox("Group by", categorical_cols, key="group_cat")

        unique_vals = df[category].dropna().unique()
        try:
            unique_vals = sorted(unique_vals)
        except TypeError:
            unique_vals = [str(v) for v in unique_vals]  # Fallback for mixed types

        with col2:
            selected_level = st.selectbox(f"Select level of '{category}'", unique_vals)

        # Filter data for selected group
        group_data = df[df[category] == selected_level]

        if group_data.empty:
            st.warning(f"No data found for {category} = {selected_level}")
        else:
            st.markdown(f"#### Summary Statistics for `{category} = {selected_level}`")
            st.dataframe(group_data[numerical_cols].describe(), use_container_width=True)

            # Bar chart for a selected numerical column
            st.markdown("#### Bar Chart for Selected Numerical Column")
            selected_num = st.selectbox("Choose numerical column", numerical_cols, key="num_bar")

            fig3, ax3 = plt.subplots(figsize=(6, 4))
            group_data.boxplot(column=selected_num, ax=ax3)
            ax3.set_title(f"Distribution of '{selected_num}'")
            ax3.set_ylabel(selected_num)
            st.pyplot(fig3)

    else:
        st.info("No categorical columns to group by.")

    # --- Final Info ---
    st.markdown("---")
    st.markdown("<small>Visualizations generated locally. No data leaves your machine.</small>", unsafe_allow_html=True)