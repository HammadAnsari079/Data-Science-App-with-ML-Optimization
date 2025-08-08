"""
optimization.py - Feature Optimization Module
================================================
Identifies and removes redundant (highly correlated) columns from the dataset.
Helps improve model performance by reducing multicollinearity.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os


def get_redundant_columns(corr_matrix: pd.DataFrame, target_column: str, threshold: float = 0.9) -> list:
    """
    Identifies redundant columns based on high correlation with each other,
    excluding those strongly tied to the target (y).

    Args:
        corr_matrix (pd.DataFrame): Correlation matrix of numerical features.
        target_column (str): The target variable to preserve correlation with.
        threshold (float): Correlation threshold to consider as redundant (e.g., 0.95).

    Returns:
        list: List of column names to drop.
    """
    # Focus on absolute correlation
    corr_abs = corr_matrix.abs()

    # Get correlation of features with target
    target_corr = corr_abs[target_column].drop(target_column)

    # Sort features by importance to target (descending)
    sorted_by_target = target_corr.sort_values(ascending=False).index.tolist()

    # Track columns to keep
    keep, redundant = [], []

    for col in sorted_by_target:
        # Check if this column is highly correlated with any already kept column
        is_redundant = False
        for kept_col in keep:
            if corr_abs.loc[col, kept_col] > threshold:
                is_redundant = True
                break
        if is_redundant:
            redundant.append(col)
        else:
            keep.append(col)

    return redundant


def app():
    """
    Main function for the 'Y-Parameter Optimization' page.
    Allows users to remove redundant features based on correlation with respect to the target.
    """
    st.title("⚡ Feature Optimization")

    # --- Check if Data Exists ---
    data_path = "data/main_data.csv"
    metadata_path = "data/metadata/column_type_desc.csv"

    if not os.path.exists(data_path):
        st.warning("No data found. Please go to **Upload Data** to load a dataset.")
        st.stop()

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        st.error("Failed to load the dataset.")
        st.code(f"Error: {e}")
        st.stop()

    if df.empty:
        st.warning("The dataset is empty.")
        st.stop()

    st.markdown(f"**Dataset Shape:** `{df.shape[0]}` rows × `{df.shape[1]}` columns")
    st.dataframe(df.head(100), use_container_width=True)

    # --- Get Numerical Columns ---
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numerical_cols) < 2:
        st.info("Need at least 2 numerical columns to detect redundancy.")
        st.stop()

    # --- User Inputs ---
    st.markdown("### ⚙️ Configure Optimization")
    y_var = st.selectbox(
        "Select target variable (y)",
        options=numerical_cols,
        help="The column you want to predict. Features highly correlated with it will be preserved."
    )

    threshold = st.slider(
        "Correlation Threshold",
        min_value=0.50,
        max_value=0.99,
        value=0.90,
        step=0.01,
        format="%.2f",
        help="Columns with correlation above this value are considered redundant."
    )

    # --- Compute Redundancy ---
    if st.button("🔍 Find Redundant Columns", type="primary"):
        try:
            # Compute correlation matrix
            corr = df[numerical_cols].corr(method='pearson')

            # Identify redundant columns
            redundant_cols = get_redundant_columns(corr, target_column=y_var, threshold=threshold)

            # Show results
            st.markdown("### 🧹 Optimization Results")

            if not redundant_cols:
                st.success("✅ No redundant columns found. All features are relatively independent.")
            else:
                st.warning(f"Found `{len(redundant_cols)}` redundant column(s) that can be removed:")
                for col in redundant_cols:
                    st.markdown(f"- `{col}`")

                # Generate optimized dataset
                optimized_df = df.drop(columns=redundant_cols)

                st.markdown("### 📊 Optimized Dataset Preview")
                st.dataframe(optimized_df.head(100), use_container_width=True)

                # Offer download
                csv = optimized_df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Optimized Data (CSV)",
                    data=csv,
                    file_name="optimized_data.csv",
                    mime="text/csv"
                )

                # Optional: Save optimized data
                try:
                    os.makedirs("data/optimized", exist_ok=True)
                    optimized_df.to_csv("data/optimized/optimized_data.csv", index=False)
                    st.info("Optimized data saved to `data/optimized/optimized_data.csv`")
                except Exception as e:
                    st.warning("Could not save optimized data.")
                    st.code(f"Save error: {e}")

        except Exception as e:
            st.error("An error occurred during optimization.")
            st.code(f"Error: {e}")

    # --- Correlation Heatmap (Optional) ---
    with st.expander("View Correlation Matrix"):
        import matplotlib.pyplot as plt
        import seaborn as sns

        corr = df[numerical_cols].corr(method='pearson')
        fig, ax = plt.subplots(figsize=(6, 4))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            ax=ax
        )
        ax.set_title("Feature Correlation Matrix")
        st.pyplot(fig)

    # --- Final Note ---
    st.markdown("---")
    st.markdown(
        "<small>Optimization helps reduce multicollinearity. "
        "All operations are local and private.</small>",
        unsafe_allow_html=True
    )