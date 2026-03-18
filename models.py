# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================
import os
import pickle
import json
import streamlit as st


@st.cache_data
def load_model():
    """
    Load the trained XGBoost model from pickle file.

    Uses Streamlit's cache_data decorator to load the model only once
    and reuse it across reruns, improving performance.

    Returns:
        model: Trained XGBoost model object or None if file not found
    """
    possible_paths = [
        'artifacts/xgb2_model.pkl',
        'xgb2_model.pkl',
        '../xgb2_model.pkl',
        '../models/xgb2_model.pkl'
    ]

    for model_path in possible_paths:
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                return model
            except Exception as e:
                st.error(f"Error loading model from {model_path}: {str(e)}")
                continue

    st.error("Model file not found. Please check the model path.")
    return None


@st.cache_data
def load_model_metadata():
    """
    Load model metadata including state-wise RMSE values.

    Returns:
        dict: Model metadata dictionary or None if file not found
    """
    possible_paths = [
        'artifacts/model_metadata.json',
        'model_metadata.json',
        '../model_metadata.json',
        './model_metadata.json'
    ]

    for metadata_path in possible_paths:
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                return metadata
            except Exception as e:
                st.warning(f"Error loading metadata from {metadata_path}: {str(e)}")
                continue

    return None


def get_state_rmse(state_name, metadata):
    """
    Get state-specific RMSE from metadata.

    Parameters:
        state_name (str): Name of the state
        metadata (dict): Model metadata dictionary

    Returns:
        float: RMSE value for the state, or None if not found
    """
    if metadata is None or 'state_rmse' not in metadata:
        return None

    state_rmse_dict = metadata['state_rmse']

    if state_name in state_rmse_dict:
        return state_rmse_dict[state_name]

    # Handle state name variations
    if state_name == "Jammu and Kashmir":
        if "J&K" in state_rmse_dict:
            return state_rmse_dict["J&K"]
    elif state_name == "J&K":
        if "Jammu and Kashmir" in state_rmse_dict:
            return state_rmse_dict["Jammu and Kashmir"]

    return None
