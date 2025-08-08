"""
modelUtils.py - Model Saving & Loading Utility
================================================
Provides safe, reliable functions to save and load machine learning models using joblib.
Uses a standardized local directory (`data/models`) to ensure compatibility with the app.
"""

import os
import joblib
from typing import Any


# Define the model directory relative to the project
MODEL_DIR = "data/models"


def ensure_model_directory() -> bool:
    """
    Ensures the model directory exists. Creates it if not.

    Returns:
        bool: True if directory exists or is created successfully.
    """
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        return True
    except Exception as e:
        print(f"Failed to create model directory '{MODEL_DIR}': {e}")
        return False


def save_model(model: Any, filename: str = None) -> str:
    """
    Saves a trained model to disk using joblib.

    Args:
        model (Any): The trained scikit-learn model to save.
        filename (str, optional): Name of the file (without extension). 
                                  If None, auto-generates a name.

    Returns:
        str: Full path to the saved model file.

    Raises:
        IOError: If saving fails due to permissions or disk issues.
    """
    # Ensure the model directory exists
    if not ensure_model_directory():
        raise IOError("Model directory could not be created.")

    # Generate filename
    if filename is None:
        existing = [f for f in os.listdir(MODEL_DIR) if f.endswith(".joblib")]
        idx = len(existing) + 1
        filename = f"model_{idx}"
    elif not filename.endswith(".joblib"):
        filename = f"{filename}.joblib"

    filepath = os.path.join(MODEL_DIR, filename)

    # Save the model
    try:
        joblib.dump(model, filepath)
        return filepath
    except Exception as e:
        raise IOError(f"Failed to save model to {filepath}: {e}")


def load_model(filepath: str) -> Any:
    """
    Loads a saved model from disk.

    Args:
        filepath (str): Full path to the model file.

    Returns:
        Any: The loaded model object.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If loading fails.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")

    try:
        model = joblib.load(filepath)
        return model
    except Exception as e:
        raise IOError(f"Failed to load model from {filepath}: {e}")


def list_saved_models() -> list:
    """
    Lists all saved model files in the model directory.

    Returns:
        list: List of model filenames.
    """
    if not os.path.exists(MODEL_DIR):
        return []
    return [f for f in os.listdir(MODEL_DIR) if f.endswith(".joblib")]


