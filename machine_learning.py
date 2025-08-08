"""
machine_learning.py - Machine Learning Module
=============================================
A no-code interface for running regression and classification models.
Users select target (y) and features (X), choose prediction type,
and the app trains models, evaluates performance, and saves the best one.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


def is_numerical(series: pd.Series) -> bool:
    """
    Check if a pandas Series is numerical.
    """
    try:
        pd.to_numeric(series.dropna())
        return True
    except (ValueError, TypeError):
        return False


def app():
    """
    Main function for the 'Machine Learning' page.
    Enables users to build ML models without coding.
    Supports regression and classification with automatic encoding and model selection.
    """
    st.title("🤖 Machine Learning")

    # --- Check if Data Exists ---
    data_path = "data/main_data.csv"
    metadata_dir = "data/metadata"

    if not os.path.exists(data_path):
        st.warning("No data found. Please go to **Upload Data** to load a dataset.")
        st.stop()

    # Ensure metadata directory exists
    os.makedirs(metadata_dir, exist_ok=True)

    # --- Load Data ---
    try:
        data = pd.read_csv(data_path)
    except Exception as e:
        st.error("Failed to load the dataset.")
        st.code(f"Error: {e}")
        st.stop()

    if data.empty:
        st.warning("The dataset is empty.")
        st.stop()

    st.markdown(f"**Dataset Shape:** `{data.shape[0]}` rows × `{data.shape[1]}` columns")
    st.markdown("---")

    # --- Model Configuration ---
    st.markdown("### ⚙️ Configure Your Model")
    col1, col2 = st.columns(2)

    with col1:
        y_var = st.selectbox(
            "Select target variable (y)",
            options=data.columns.tolist(),
            help="This is the column the model will predict."
        )

    with col2:
        X_options = [col for col in data.columns if col != y_var]
        X_var = st.multiselect(
            "Select feature variables (X)",
            options=X_options,
            help="Choose one or more columns to use as input features."
        )

    # Validate X variables
    if not X_var:
        st.error("You must select at least one feature (X) for prediction.")
        st.stop()

    if y_var in X_var:
        st.error("The target variable (y) cannot be included in the features (X).")
        st.stop()

    pred_type = st.radio(
        "Select prediction type",
        options=["Regression", "Classification"],
        help="Regression: Predict continuous numbers. Classification: Predict categories."
    )

    # Confirm setup
    st.markdown("---")
    st.markdown("### ✅ Model Summary")
    st.write(f"**Prediction Type:** {pred_type}")
    st.write(f"**Target (y):** `{y_var}`")
    st.write(f"**Features (X):** `{', '.join(X_var)}`")

    # Store parameters
    model_params = {
        'X': X_var,
        'y': y_var,
        'pred_type': pred_type,
        'random_state': 42
    }

    st.markdown("---")

    # --- Train Test Split ---
    st.markdown("### 📊 Train/Test Split")
    test_size = st.slider(
        "Testing set size (%)",
        min_value=10,
        max_value=50,
        value=20,
        step=10,
        help="Percentage of data used for testing. Default: 20%"
    ) / 100.0

    X = data[X_var].copy()
    y = data[y_var].copy()

    # One-hot encode X
    X = pd.get_dummies(X, drop_first=True)

    # Encode y if classification
    if pred_type == "Classification":
        if is_numerical(y):
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            classes = le.classes_
        else:
            y_encoded = y.copy()
            classes = y.dropna().unique()
            # Re-encode as labels
            y_encoded = LabelEncoder().fit_transform(y)
            classes = pd.Categorical(y).categories.tolist()

        st.markdown("### 🔤 Target Class Encoding")
        for idx, cls in enumerate(classes):
            st.write(f"**{cls}** → `{idx}`")

        y = y_encoded

    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y if pred_type == "Classification" else None
        )
    except Exception as e:
        st.error("Could not split data.")
        st.code(f"Error: {e}")
        st.stop()

    st.write(f"- Training samples: `{X_train.shape[0]}`")
    st.write(f"- Testing samples: `{X_test.shape[0]}`")
    st.write(f"- Number of features after encoding: `{X_train.shape[1]}`")

    # Save model parameters
    try:
        with open(os.path.join(metadata_dir, "model_params.json"), "w") as f:
            json.dump(model_params, f, indent=4)
    except Exception as e:
        st.warning("Could not save model parameters.")
        st.code(f"Save error: {e}")

    st.markdown("---")

    # --- Model Training ---
    st.markdown("### 🏁 Running Models")

    if pred_type == "Regression":
        results = []

        # Linear Regression
        try:
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            r2 = lr.score(X_test, y_test)
            results.append(["Linear Regression", r2])
        except Exception as e:
            results.append(["Linear Regression", np.nan])
            st.warning("LR failed: " + str(e))

        # Decision Tree Regressor
        try:
            dt = DecisionTreeRegressor(random_state=42)
            dt.fit(X_train, y_train)
            r2_dt = dt.score(X_test, y_test)
            results.append(["Decision Tree Regressor", r2_dt])
        except Exception as e:
            results.append(["Decision Tree Regressor", np.nan])
            st.warning("DT Regressor failed: " + str(e))

        # Filter out failed models
        results = [r for r in results if not np.isnan(r[1])]
        if not results:
            st.error("All regression models failed to train.")
            st.stop()

        # Sort and display
        results_df = pd.DataFrame(results, columns=["Model", "R² Score"]).round(4)
        results_df = results_df.sort_values("R² Score", ascending=False).reset_index(drop=True)

        st.dataframe(results_df, use_container_width=True)

        # Save best model
        best_model = dt if results_df.iloc[0]["Model"] == "Decision Tree Regressor" else lr
        model_path = os.path.join(metadata_dir, "model_reg.sav")
        try:
            joblib.dump(best_model, model_path)
            st.success("✅ Best regression model saved.")
        except Exception as e:
            st.warning("Could not save model: " + str(e))

    elif pred_type == "Classification":
        results = []

        # Logistic Regression
        try:
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(X_train, y_train)
            acc = lr.score(X_test, y_test)
            results.append(["Logistic Regression", acc])
        except Exception as e:
            results.append(["Logistic Regression", np.nan])
            st.warning("Logistic Regression failed: " + str(e))

        # Decision Tree Classifier
        try:
            dt = DecisionTreeClassifier(random_state=42)
            dt.fit(X_train, y_train)
            acc_dt = dt.score(X_test, y_test)
            results.append(["Decision Tree Classifier", acc_dt])
        except Exception as e:
            results.append(["Decision Tree Classifier", np.nan])
            st.warning("DT Classifier failed: " + str(e))

        # Filter failed models
        results = [r for r in results if not np.isnan(r[1])]
        if not results:
            st.error("All classification models failed to train.")
            st.stop()

        # Sort and display
        results_df = pd.DataFrame(results, columns=["Model", "Accuracy"]).round(4)
        results_df = results_df.sort_values("Accuracy", ascending=False).reset_index(drop=True)

        st.dataframe(results_df, use_container_width=True)

        # Save best model
        best_model = dt if results_df.iloc[0]["Model"] == "Decision Tree Classifier" else lr
        model_path = os.path.join(metadata_dir, "model_classification.sav")
        try:
            joblib.dump(best_model, model_path)
            st.success("✅ Best classification model saved.")
        except Exception as e:
            st.warning("Could not save model: " + str(e))

    # --- Final Notes ---
    st.markdown("---")
    st.markdown("<small>Models trained locally. No data leaves your machine.</small>", unsafe_allow_html=True)