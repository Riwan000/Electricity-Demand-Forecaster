"""
Smart Energy Forecasting Platform - Streamlit Dashboard
Stage 1.5: Forecast + Explain

This application provides an interactive dashboard for electricity demand forecasting
using machine learning models and real-time weather data.
"""

# ============================================================================
# IMPORTS
# ============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import holidays
import os
import requests  # For weather API calls
import json
from scipy import stats  # For confidence interval calculation
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# RAG imports
from utils.embeddings import get_embeddings, get_openrouter_api_key
from utils.vector_store import VectorStore
from utils.rag_builder import (
    load_model_metadata as load_model_metadata_rag,
    build_model_metrics_documents,
    build_feature_importance_documents,
    build_forecast_summary_documents
)
from utils.rag_engine import RAGEngine

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Smart Energy Forecasting Platform",
    page_icon="⚡",
    layout="wide"
)

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# State to coordinates mapping (approximate centers of Indian states)
# These coordinates are used to fetch weather data from APIs
STATE_COORDINATES = {
    "Kerala": {"lat": 10.8505, "lon": 76.2711},
    "Goa": {"lat": 15.2993, "lon": 74.1240},
    "Maharashtra": {"lat": 19.7515, "lon": 75.7139},
    "Karnataka": {"lat": 15.3173, "lon": 75.7139},
    "Tamil Nadu": {"lat": 11.1271, "lon": 78.6569},
    "Gujarat": {"lat": 23.0225, "lon": 72.5714},
    "Rajasthan": {"lat": 27.0238, "lon": 74.2179},
    "Uttar Pradesh": {"lat": 26.8467, "lon": 80.9462},
    "West Bengal": {"lat": 22.9868, "lon": 87.8550},
    "Andhra Pradesh": {"lat": 15.9129, "lon": 79.7400},
    "Telangana": {"lat": 18.1124, "lon": 79.0193},
    "Punjab": {"lat": 31.1471, "lon": 75.3412},
    "Haryana": {"lat": 29.0588, "lon": 76.0856},
    "Madhya Pradesh": {"lat": 22.9734, "lon": 78.6569},
    "Bihar": {"lat": 25.0961, "lon": 85.3131},
    "Odisha": {"lat": 20.9517, "lon": 85.0985},
    "Assam": {"lat": 26.2006, "lon": 92.9376},
    "Jharkhand": {"lat": 23.6102, "lon": 85.2799},
    "Chhattisgarh": {"lat": 21.2787, "lon": 81.8661},
    "Uttarakhand": {"lat": 30.0668, "lon": 79.0193},
    "Himachal Pradesh": {"lat": 31.1048, "lon": 77.1734},
    "Jammu and Kashmir": {"lat": 34.0837, "lon": 74.7973},
    "Manipur": {"lat": 24.6637, "lon": 93.9063},
    "Meghalaya": {"lat": 25.4670, "lon": 91.3662},
    "Mizoram": {"lat": 23.1645, "lon": 92.9376},
    "Nagaland": {"lat": 26.1584, "lon": 94.5624},
    "Tripura": {"lat": 23.9408, "lon": 91.9882},
    "Arunachal Pradesh": {"lat": 28.2180, "lon": 94.7278},
    "Sikkim": {"lat": 27.5330, "lon": 88.5122},
    "NCT of Delhi": {"lat": 28.6139, "lon": 77.2090},
    "Puducherry": {"lat": 11.9416, "lon": 79.8083},
    "Chandigarh": {"lat": 30.7333, "lon": 76.7794},
    "Dadra and Nagar Haveli": {"lat": 20.1809, "lon": 73.0169},
    "Daman and Diu": {"lat": 20.3974, "lon": 72.8328},
    "Lakshadweep": {"lat": 10.5667, "lon": 72.6417},
    "Andaman and Nicobar": {"lat": 11.7401, "lon": 92.6586},
}

# List of all Indian states for dropdown
ALL_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jammu and Kashmir",
    "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra",
    "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal", "NCT of Delhi",
    "Puducherry", "Chandigarh", "Dadra and Nagar Haveli", "Daman and Diu",
    "Lakshadweep", "Andaman and Nicobar"
]

# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

@st.cache_data
def load_model():
    """
    Load the trained XGBoost model from pickle file.
    
    Uses Streamlit's cache_data decorator to load the model only once
    and reuse it across reruns, improving performance.
    
    Returns:
        model: Trained XGBoost model object or None if file not found
    """
    # Try multiple possible model paths
    possible_paths = [
        #'../xgb1_model.pkl',  # From frontend directory
        #'xgb1_model.pkl',      # From root directory
        #'../models/xgb1_model.pkl',  # If models folder exists
        '../xgb2_model.pkl',  # From frontend directory
        'xgb2_model.pkl',      # From root directory
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
    
    # Try exact match first
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

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data
def load_historical_data(state_name):
    """
    Load historical weather data for a specific state from CSV files.
    
    This function loads the historical weather data that was used to train
    the model. It's used as a fallback when weather API fails or for
    climatology-based forecasting.
    
    Parameters:
        state_name (str): Name of the Indian state
        
    Returns:
        pd.DataFrame: Historical weather data with date column, or None if not found
    """
    # Mapping of state names to their CSV file names
    state_file_map = {
        'Arunachal Pradesh': 'Arunachal_Pradesh.csv',
        'Assam': 'Assam.csv',
        'Bihar': 'Bihar.csv',
        'Chhattisgarh': 'Chhattisgarh.csv',
        'Goa': 'Goa.csv',
        'Gujarat': 'Gujarat.csv',
        'Haryana': 'Haryana.csv',
        'Himachal Pradesh': 'Himachal_Pradesh.csv',
        'Jammu and Kashmir': 'Jammu_and_Kashmir.csv',
        'Jharkhand': 'Jharkhand.csv',
        'Karnataka': 'Karnataka.csv',
        'Kerala': 'Kerala.csv',
        'Madhya Pradesh': 'Madhya_Pradesh.csv',
        'Maharashtra': 'Maharashtra.csv',
        'Manipur': 'Manipur.csv',
        'Meghalaya': 'Meghalaya.csv',
        'Mizoram': 'Mizoram.csv',
        'Nagaland': 'Nagaland.csv',
        'Odisha': 'Odisha.csv',
        'Punjab': 'Punjab.csv',
        'Rajasthan': 'Rajasthan.csv',
        'Sikkim': 'Sikkim.csv',
        'Tamil Nadu': 'Tamil_Nadu.csv',
        'Telangana': 'Telangana.csv',
        'Tripura': 'Tripura.csv',
        'Uttar Pradesh': 'Uttar_Pradesh.csv',
        'Uttarakhand': 'Uttarakhand.csv',
        'West Bengal': 'West_Bengal.csv',
        'Andhra Pradesh': 'Andhra_Pradesh.csv',
        'NCT of Delhi': 'NCT_of_Delhi.csv',
        'Puducherry': 'Puducherry.csv',
        'Chandigarh': 'Chandigarh.csv',
        'Dadra and Nagar Haveli': 'Dadra_and_Nagar_Haveli.csv',
        'Daman and Diu': 'Daman_and_Diu.csv',
        'Lakshadweep': 'Lakshadweep.csv',
        'Andaman and Nicobar': 'Andaman_and_Nicobar.csv'
    }

    # Construct file path - try multiple possible locations
    possible_paths = [
        f"../data/14983362/{state_file_map.get(state_name, 'Kerala.csv')}",
        f"data/14983362/{state_file_map.get(state_name, 'Kerala.csv')}",
    ]
    
    for file_path in possible_paths:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df['date'] = pd.to_datetime(df['date'])
                return df
            except Exception as e:
                st.warning(f"Error loading {file_path}: {str(e)}")
                continue
    
    return None


@st.cache_data
def load_historical_demand(state_name):
    """
    Load historical actual demand data for a specific state from CSV files.
    
    This function loads the actual electricity demand data that was used to train
    the model. It's used for residual analysis and baseline comparisons.
    
    Parameters:
        state_name (str): Name of the Indian state
        
    Returns:
        pd.DataFrame: Historical demand data with date and actual_demand_MU columns, or None if not found
    """
    # Mapping of state names to CSV column names in daily_energy_met_MU.csv
    state_column_map = {
        'Andhra Pradesh': 'Andhra Pradesh',
        'Arunachal Pradesh': 'Arunachal Pradesh',
        'Assam': 'Assam',
        'Bihar': 'Bihar',
        'Chhattisgarh': 'Chhattisgarh',
        'Goa': 'Goa',
        'Gujarat': 'Gujarat',
        'Haryana': 'Haryana',
        'Himachal Pradesh': 'Himachal Pradesh',
        'Jammu and Kashmir': 'Jammu and Kashmir',
        'Jharkhand': 'Jharkhand',
        'Karnataka': 'Karnataka',
        'Kerala': 'Kerala',
        'Madhya Pradesh': 'Madhya Pradesh',
        'Maharashtra': 'Maharashtra',
        'Manipur': 'Manipur',
        'Meghalaya': 'Meghalaya',
        'Mizoram': 'Mizoram',
        'Nagaland': 'Nagaland',
        'Odisha': 'Odisha',
        'Punjab': 'Punjab',
        'Rajasthan': 'Rajasthan',
        'Sikkim': 'Sikkim',
        'Tamil Nadu': 'Tamil Nadu',
        'Telangana': 'Telangana',
        'Tripura': 'Tripura',
        'Uttar Pradesh': 'Uttar Pradesh',
        'Uttarakhand': 'Uttarakhand',
        'West Bengal': 'West Bengal',
        'NCT of Delhi': 'Delhi',
        'Puducherry': 'Puducherry',
        'Chandigarh': 'Chandigarh',
        'Dadra and Nagar Haveli': 'Dadra and Nagar Haveli',
        'Daman and Diu': 'Daman and Diu',
        'Lakshadweep': 'Lakshadweep',
        'Andaman and Nicobar': 'Andaman and Nicobar Islands'
    }
    
    # Get column name for the state
    column_name = state_column_map.get(state_name)
    if column_name is None:
        return None
    
    # Try multiple possible file paths
    possible_paths = [
        'data/daily_energy_met_MU.csv',
        '../data/daily_energy_met_MU.csv',
        'data/14983362/daily_energy_met_MU.csv',
        '../data/14983362/daily_energy_met_MU.csv'
    ]
    
    # #region agent log
    import json
    log_path_debug = os.path.join(os.path.dirname(__file__), ".cursor", "debug.log")
    os.makedirs(os.path.dirname(log_path_debug), exist_ok=True)
    with open(log_path_debug, 'a', encoding='utf-8') as f:
        f.write(json.dumps({"sessionId": "debug-session", "runId": "deployment-debug", "hypothesisId": "D", "location": "app.py:load_historical_demand", "message": "Checking file paths", "data": {"state_name": state_name, "possible_paths": possible_paths, "paths_exist": [os.path.exists(p) for p in possible_paths], "cwd": os.getcwd(), "script_dir": os.path.dirname(__file__) if '__file__' in globals() else "unknown"}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
    # #endregion
    
    for file_path in possible_paths:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                
                # Check if the column exists
                if column_name not in df.columns:
                    # Try alternative column name
                    if state_name == 'NCT of Delhi' and 'Delhi' in df.columns:
                        column_name = 'Delhi'
                    elif state_name == 'Andaman and Nicobar' and 'Andaman and Nicobar Islands' in df.columns:
                        column_name = 'Andaman and Nicobar Islands'
                    else:
                        continue
                
                # Extract date and demand columns
                date_col = 'Date' if 'Date' in df.columns else 'date'
                if date_col not in df.columns:
                    continue
                
                demand_df = pd.DataFrame({
                    'date': pd.to_datetime(df[date_col]),
                    'actual_demand_MU': pd.to_numeric(df[column_name], errors='coerce')
                })
                
                # Remove rows with missing values
                demand_df = demand_df.dropna()
                
                if len(demand_df) > 0:
                    # #region agent log
                    with open(log_path_debug, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "deployment-debug", "hypothesisId": "D", "location": "app.py:load_historical_demand:success", "message": "Successfully loaded historical demand", "data": {"file_path": file_path, "demand_df_len": len(demand_df), "column_name": column_name}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
                    # #endregion
                    return demand_df
                    
            except Exception as e:
                # #region agent log
                with open(log_path_debug, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "deployment-debug", "hypothesisId": "D", "location": "app.py:load_historical_demand:error", "message": "Error loading demand data", "data": {"file_path": file_path, "error": str(e)}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
                # #endregion
                st.warning(f"Error loading demand data from {file_path}: {str(e)}")
                continue
    
    return None


def get_fallback_baseline(state_name):
    """
    Get fallback baseline for a state when historical data is unavailable.
    These are approximate typical daily demand values in MU.
    """
    fallback_baselines = {
        'Maharashtra': 400.0, 'Gujarat': 250.0, 'Tamil Nadu': 300.0, 'Karnataka': 200.0,
        'Uttar Pradesh': 350.0, 'West Bengal': 200.0, 'Rajasthan': 150.0, 'Madhya Pradesh': 180.0,
        'Andhra Pradesh': 200.0, 'Telangana': 150.0, 'Punjab': 120.0, 'Haryana': 100.0,
        'Bihar': 200.0, 'Odisha': 120.0, 'Kerala': 100.0, 'Assam': 80.0,
        'Jharkhand': 100.0, 'Chhattisgarh': 100.0, 'Uttarakhand': 50.0, 'Himachal Pradesh': 40.0,
        'Jammu and Kashmir': 50.0, 'NCT of Delhi': 150.0, 'Goa': 15.0, 'Puducherry': 10.0,
        'Chandigarh': 8.0, 'Tripura': 30.0, 'Manipur': 25.0, 'Meghalaya': 20.0,
        'Mizoram': 15.0, 'Nagaland': 20.0, 'Arunachal Pradesh': 15.0, 'Sikkim': 10.0,
        'Dadra and Nagar Haveli': 5.0, 'Daman and Diu': 5.0, 'Lakshadweep': 2.0,
        'Andaman and Nicobar': 5.0
    }
    return fallback_baselines.get(state_name, 100.0)  # Default to 100 MU if state not found


def calculate_state_30d_baseline(historical_demand, date):
    """
    Calculate 30-day rolling baseline for a specific date.
    
    Parameters:
        historical_demand (pd.DataFrame): DataFrame with date and actual_demand_MU columns
        date (pd.Timestamp): Date for which to calculate baseline
        
    Returns:
        float: 30-day rolling average baseline, or None if insufficient data
    """
    if historical_demand is None or len(historical_demand) == 0:
        return None
    
    # Filter to dates before the target date
    historical_demand_sorted = historical_demand.sort_values('date')
    before_date = historical_demand_sorted[historical_demand_sorted['date'] < date]
    
    if len(before_date) < 30:
        # Use available data if less than 30 days
        if len(before_date) > 0:
            return before_date['actual_demand_MU'].tail(len(before_date)).mean()
        return None
    
    # Calculate 30-day rolling average
    baseline = before_date['actual_demand_MU'].tail(30).mean()
    return baseline


def denormalize_predictions(predictions, dates, state_name, historical_demand=None):
    """
    Convert log-space predictions to absolute MU values.
    
    Model (XGB2_Log) outputs log1p-transformed predictions. This function:
    1. Applies expm1 to convert from log space to processed space
    2. Multiplies by state_30d_baseline to get absolute MU values
    
    Parameters:
        predictions (np.array): Log-space predictions from model
        dates (pd.Series): Dates corresponding to predictions
        state_name (str): Name of the state
        historical_demand (pd.DataFrame): Optional historical demand data for baseline calculation
        
    Returns:
        np.array: Absolute predictions in MU
    """
    # #region agent log
    import json
    import os
    log_path = os.path.join(os.path.dirname(__file__), ".cursor", "debug.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps({"sessionId": "debug-session", "runId": "deployment-debug", "hypothesisId": "C", "location": "app.py:denormalize_predictions:entry", "message": "denormalize_predictions called", "data": {"state_name": state_name, "historical_demand_provided": historical_demand is not None, "dates_len": len(dates) if dates is not None else 0}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
    # #endregion
    
    # First, convert from log space to processed space (expm1 reverses log1p)
    predictions_processed = np.expm1(predictions)
    
    if historical_demand is None:
        historical_demand = load_historical_demand(state_name)
    
    # #region agent log
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps({"sessionId": "debug-session", "runId": "deployment-debug", "hypothesisId": "D", "location": "app.py:denormalize_predictions:after_load", "message": "After load_historical_demand", "data": {"historical_demand_is_none": historical_demand is None, "historical_demand_len": len(historical_demand) if historical_demand is not None else 0}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
    # #endregion
    
    if historical_demand is None or len(historical_demand) == 0:
        # #region agent log
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "deployment-debug", "hypothesisId": "D", "location": "app.py:denormalize_predictions:no_historical", "message": "No historical demand - using fallback baseline", "data": {"predictions_processed_mean": float(np.mean(predictions_processed)), "state_name": state_name}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
        # #endregion
        
        # FIX: Use fallback baseline when historical data unavailable (deployment scenario)
        fallback_baseline = get_fallback_baseline(state_name)
        
        # #region agent log
        with open(log_path, 'a', encoding='utf-8') as f:
            result = predictions_processed * fallback_baseline
            f.write(json.dumps({"sessionId": "debug-session", "runId": "deployment-debug", "hypothesisId": "D", "location": "app.py:denormalize_predictions:fallback_applied", "message": "Fallback baseline applied", "data": {"fallback_baseline": fallback_baseline, "result_mean": float(np.mean(result)), "result_min": float(np.min(result)), "result_max": float(np.max(result))}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
        # #endregion
        
        return predictions_processed * fallback_baseline
    
    # For future dates (forecasting), use most recent 30-day baseline
    # Convert both to Timestamp for comparison (handles both Timestamp and date objects)
    if len(dates) > 0:
        first_date = pd.Timestamp(dates.iloc[0])
        current_date = pd.Timestamp.now()
        
        # #region agent log
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "deployment-debug", "hypothesisId": "B", "location": "app.py:denormalize_predictions:date_check", "message": "Date comparison in denormalize", "data": {"first_date": str(first_date), "current_date": str(current_date), "first_date_gt_current": bool(first_date > current_date), "first_date_type": str(type(first_date)), "current_date_type": str(type(current_date))}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
        # #endregion
        
        if first_date > current_date:
            # Forecasting future dates - use recent baseline
            if len(historical_demand) >= 30:
                baseline = historical_demand['actual_demand_MU'].tail(30).mean()
            elif len(historical_demand) > 0:
                baseline = historical_demand['actual_demand_MU'].mean()
            else:
                # #region agent log
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "deployment-debug", "hypothesisId": "C", "location": "app.py:denormalize_predictions:no_baseline", "message": "No baseline in historical_demand - using fallback", "data": {"predictions_processed_mean": float(np.mean(predictions_processed)), "state_name": state_name}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
                # #endregion
                # FIX: Use fallback baseline when historical_demand exists but has no data
                fallback_baseline = get_fallback_baseline(state_name)
                return predictions_processed * fallback_baseline
            
            # #region agent log
            result = predictions_processed * baseline
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "deployment-debug", "hypothesisId": "C", "location": "app.py:denormalize_predictions:future_dates", "message": "Future dates path - baseline applied", "data": {"baseline": float(baseline), "result_mean": float(np.mean(result)), "result_min": float(np.min(result)), "result_max": float(np.max(result))}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
            # #endregion
            return result
        else:
            # Historical dates - calculate baseline for each date
            baselines = dates.apply(lambda d: calculate_state_30d_baseline(historical_demand, d))
            # Fill NaN baselines with mean
            if baselines.isna().any():
                mean_baseline = baselines.mean() if baselines.notna().any() else historical_demand['actual_demand_MU'].mean()
                baselines = baselines.fillna(mean_baseline)
            
            # #region agent log
            result = predictions_processed * baselines.values
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "deployment-debug", "hypothesisId": "C", "location": "app.py:denormalize_predictions:historical_dates", "message": "Historical dates path - baselines applied", "data": {"baselines_mean": float(baselines.mean()), "result_mean": float(np.mean(result)), "result_min": float(np.min(result)), "result_max": float(np.max(result))}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
            # #endregion
            return result
    else:
        # #region agent log
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "deployment-debug", "hypothesisId": "C", "location": "app.py:denormalize_predictions:no_dates", "message": "No dates provided - using fallback baseline", "data": {"predictions_processed_mean": float(np.mean(predictions_processed)), "state_name": state_name}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
        # #endregion
        # FIX: No dates provided - use fallback baseline instead of returning processed predictions
        fallback_baseline = get_fallback_baseline(state_name)
        return predictions_processed * fallback_baseline


# ============================================================================
# WEATHER API FUNCTIONS
# ============================================================================

def get_weather_forecast_openmeteo(state_name, start_date, num_days):
    """
    Get weather forecast using Open-Meteo API (FREE, no API key required).
    
    Open-Meteo is a free weather API that provides up to 16 days of forecast
    data without requiring registration or API keys. This makes it ideal for
    development and production use.
    
    Parameters:
        state_name (str): Name of the Indian state
        start_date (datetime.date): Start date for forecast
        num_days (int): Number of days to forecast (max 16 for free tier)
        
    Returns:
        pd.DataFrame: Weather data matching model's expected format
        
    Raises:
        ValueError: If state coordinates not found
        Exception: If API request fails
    """
    # Get coordinates for the state
    if state_name not in STATE_COORDINATES:
        raise ValueError(f"Coordinates not found for state: {state_name}")
    
    coords = STATE_COORDINATES[state_name]
    lat, lon = coords["lat"], coords["lon"]
    
    # Limit to 16 days (Open-Meteo free limit)
    num_days = min(num_days, 16)
    
    try:
        # Open-Meteo API endpoint for daily forecast
        url = "https://api.open-meteo.com/v1/forecast"
        
        # API parameters - requesting all weather variables needed by the model
        # Open-Meteo API expects daily as comma-separated string
        # Note: temperature_2m_mean is not available, we'll calculate it from max/min
        # Note: dewpoint_2m should be dewpoint_2m_mean
        daily_params = "temperature_2m_max,temperature_2m_min,dewpoint_2m_mean,windspeed_10m_max,winddirection_10m_dominant,shortwave_radiation_sum,cloudcover_mean"
        
        # Build query string manually to control encoding
        from urllib.parse import urlencode, quote
        base_params = {
            "latitude": lat,
            "longitude": lon,
            "forecast_days": num_days,
            "timezone": "Asia/Kolkata"
        }
        query_parts = [urlencode(base_params), f"daily={daily_params}"]
        full_url = f"{url}?{'&'.join(query_parts)}"
        
        # Make API request with timeout
        response = requests.get(full_url, timeout=10)
        response.raise_for_status()  # Raise exception for bad status codes
        data = response.json()
        
        if "daily" not in data:
            raise ValueError("Invalid API response: missing 'daily' key")
        
        daily_data = data["daily"]
        
        # Convert API response to DataFrame matching model's expected format
        # Calculate temperature mean from max and min (API doesn't provide mean directly)
        temp_mean = [(max + min) / 2 for max, min in zip(
            daily_data["temperature_2m_max"], 
            daily_data["temperature_2m_min"]
        )]
        
        forecast_df = pd.DataFrame({
            "date": pd.to_datetime(daily_data["time"]),
            "2m_temperature_max": daily_data["temperature_2m_max"],
            "2m_temperature_min": daily_data["temperature_2m_min"],
            "2m_temperature_mean": temp_mean,
        })
        
        # Convert temperatures from Celsius to Kelvin
        # (Check your model - if it uses Celsius, remove this conversion)
        forecast_df["2m_temperature_max"] = forecast_df["2m_temperature_max"] + 273.15
        forecast_df["2m_temperature_min"] = forecast_df["2m_temperature_min"] + 273.15
        forecast_df["2m_temperature_mean"] = forecast_df["2m_temperature_mean"] + 273.15
        
        # Add dewpoint (convert to Kelvin if needed)
        # API provides dewpoint_2m_mean, not dewpoint_2m
        if "dewpoint_2m_mean" in daily_data:
            forecast_df["2m_dewpoint_temperature_mean"] = [
                d + 273.15 if d is not None else None 
                for d in daily_data["dewpoint_2m_mean"]
            ]
            # For min/max, use mean as approximation (API doesn't provide min/max dewpoint)
            forecast_df["2m_dewpoint_temperature_min"] = forecast_df["2m_dewpoint_temperature_mean"]
            forecast_df["2m_dewpoint_temperature_max"] = forecast_df["2m_dewpoint_temperature_mean"]
        else:
            # Fallback: estimate dewpoint from temperature (rough approximation)
            forecast_df["2m_dewpoint_temperature_mean"] = forecast_df["2m_temperature_mean"] - 5
            forecast_df["2m_dewpoint_temperature_min"] = forecast_df["2m_dewpoint_temperature_mean"]
            forecast_df["2m_dewpoint_temperature_max"] = forecast_df["2m_dewpoint_temperature_mean"]
        
        # Wind components conversion
        # Open-Meteo gives speed and direction, we need to convert to u/v components
        if "windspeed_10m_max" in daily_data and "winddirection_10m_dominant" in daily_data:
            wind_speed = daily_data["windspeed_10m_max"]
            wind_dir_rad = np.radians(daily_data["winddirection_10m_dominant"])
            
            # Convert to u and v components (meteorological convention)
            # u = -speed * sin(direction), v = -speed * cos(direction)
            forecast_df["10m_u_component_of_wind_mean"] = [
                -speed * np.sin(rad) for speed, rad in zip(wind_speed, wind_dir_rad)
            ]
            forecast_df["10m_v_component_of_wind_mean"] = [
                -speed * np.cos(rad) for speed, rad in zip(wind_speed, wind_dir_rad)
            ]
            
            # For min/max, use mean as approximation
            forecast_df["10m_u_component_of_wind_min"] = forecast_df["10m_u_component_of_wind_mean"]
            forecast_df["10m_u_component_of_wind_max"] = forecast_df["10m_u_component_of_wind_mean"]
            forecast_df["10m_v_component_of_wind_min"] = forecast_df["10m_v_component_of_wind_mean"]
            forecast_df["10m_v_component_of_wind_max"] = forecast_df["10m_v_component_of_wind_mean"]
        else:
            # Fallback: set to 0 if data not available
            for col in ["10m_u_component_of_wind_min", "10m_u_component_of_wind_mean", 
                       "10m_u_component_of_wind_max", "10m_v_component_of_wind_min",
                       "10m_v_component_of_wind_mean", "10m_v_component_of_wind_max"]:
                forecast_df[col] = 0.0
        
        # Solar radiation conversion
        # Open-Meteo gives daily sum in J/m², convert to mean W/m²
        if "shortwave_radiation_sum" in daily_data:
            # Divide by seconds in a day (86400) to get average W/m²
            forecast_df["surface_solar_radiation_downwards_mean"] = [
                (val / 86400) if val is not None else 0 
                for val in daily_data["shortwave_radiation_sum"]
            ]
            forecast_df["surface_solar_radiation_downwards_min"] = 0.0
            forecast_df["surface_solar_radiation_downwards_max"] = forecast_df["surface_solar_radiation_downwards_mean"] * 2
        else:
            forecast_df["surface_solar_radiation_downwards_mean"] = 0.0
            forecast_df["surface_solar_radiation_downwards_min"] = 0.0
            forecast_df["surface_solar_radiation_downwards_max"] = 0.0
        
        # Cloud cover conversion
        # API provides 0-100 scale, convert to 0-1 if needed
        if "cloudcover_mean" in daily_data:
            forecast_df["total_cloud_cover_mean"] = [
                val / 100.0 if val is not None else 0 
                for val in daily_data["cloudcover_mean"]
            ]
            forecast_df["total_cloud_cover_min"] = 0.0
            forecast_df["total_cloud_cover_max"] = forecast_df["total_cloud_cover_mean"]
        else:
            forecast_df["total_cloud_cover_mean"] = 0.0
            forecast_df["total_cloud_cover_min"] = 0.0
            forecast_df["total_cloud_cover_max"] = 0.0
        
        # UTCI (Universal Thermal Climate Index) - not directly available
        # Estimate from temperature and humidity (simplified calculation)
        # For production, consider using a proper UTCI calculation library
        forecast_df["utci_mean"] = forecast_df["2m_temperature_mean"]
        forecast_df["utci_min"] = forecast_df["2m_temperature_min"]
        forecast_df["utci_max"] = forecast_df["2m_temperature_max"]
        
        return forecast_df
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {str(e)}")
    except KeyError as e:
        raise Exception(f"Missing data in API response: {str(e)}")
    except Exception as e:
        raise Exception(f"Error fetching weather data: {str(e)}")


def get_weather_forecast_climatology(state_name, start_date, num_days):
    """
    Get weather forecast using historical climatology (average of same dates in past years).
    
    This is a fallback method when weather API is unavailable. It uses historical
    averages for the same calendar dates from previous years. Less accurate than
    real forecasts but useful as a backup.
    
    Parameters:
        state_name (str): Name of the Indian state
        start_date (datetime.date): Start date for forecast
        num_days (int): Number of days to forecast
        
    Returns:
        pd.DataFrame: Weather data with historical averages
    """
    # Load historical data for the state
    historical_df = load_historical_data(state_name)
    if historical_df is None:
        raise ValueError(f"No historical data found for {state_name}")
    
    # Create date range for forecast
    forecast_dates = pd.date_range(start=start_date, periods=num_days, freq='D')
    
    # For each forecast date, get average weather from same month/day in past years
    forecast_data = []
    
    for date in forecast_dates:
        # Get same month and day from historical data (all years)
        month_day = (date.month, date.day)
        historical_same_date = historical_df[
            (historical_df['date'].dt.month == month_day[0]) & 
            (historical_df['date'].dt.day == month_day[1])
        ]
        
        if len(historical_same_date) > 0:
            # Calculate mean weather for this date across all years
            avg_weather = {
                'date': date,
                '2m_temperature_min': historical_same_date['2m_temperature_min'].mean(),
                '2m_temperature_mean': historical_same_date['2m_temperature_mean'].mean(),
                '2m_temperature_max': historical_same_date['2m_temperature_max'].mean(),
                '2m_dewpoint_temperature_min': historical_same_date['2m_dewpoint_temperature_min'].mean(),
                '2m_dewpoint_temperature_mean': historical_same_date['2m_dewpoint_temperature_mean'].mean(),
                '2m_dewpoint_temperature_max': historical_same_date['2m_dewpoint_temperature_max'].mean(),
                '10m_u_component_of_wind_mean': historical_same_date['10m_u_component_of_wind_mean'].mean(),
                '10m_v_component_of_wind_mean': historical_same_date['10m_v_component_of_wind_mean'].mean(),
                'surface_solar_radiation_downwards_mean': historical_same_date['surface_solar_radiation_downwards_mean'].mean(),
                'total_cloud_cover_mean': historical_same_date['total_cloud_cover_mean'].mean(),
                'utci_mean': historical_same_date['utci_mean'].mean(),
            }
            # Add min/max approximations
            avg_weather['10m_u_component_of_wind_min'] = avg_weather['10m_u_component_of_wind_mean']
            avg_weather['10m_u_component_of_wind_max'] = avg_weather['10m_u_component_of_wind_mean']
            avg_weather['10m_v_component_of_wind_min'] = avg_weather['10m_v_component_of_wind_mean']
            avg_weather['10m_v_component_of_wind_max'] = avg_weather['10m_v_component_of_wind_mean']
            avg_weather['surface_solar_radiation_downwards_min'] = 0.0
            avg_weather['surface_solar_radiation_downwards_max'] = avg_weather['surface_solar_radiation_downwards_mean'] * 2
            avg_weather['total_cloud_cover_min'] = 0.0
            avg_weather['total_cloud_cover_max'] = avg_weather['total_cloud_cover_mean']
            avg_weather['utci_min'] = avg_weather['2m_temperature_min']
            avg_weather['utci_max'] = avg_weather['2m_temperature_max']
            
            forecast_data.append(avg_weather)
        else:
            # Fallback: use monthly average if no exact date match
            monthly_avg = historical_df[historical_df['date'].dt.month == date.month].mean()
            avg_weather = {
                'date': date,
                '2m_temperature_min': monthly_avg.get('2m_temperature_min', 295.0),
                '2m_temperature_mean': monthly_avg.get('2m_temperature_mean', 300.0),
                '2m_temperature_max': monthly_avg.get('2m_temperature_max', 305.0),
                '2m_dewpoint_temperature_min': monthly_avg.get('2m_dewpoint_temperature_min', 290.0),
                '2m_dewpoint_temperature_mean': monthly_avg.get('2m_dewpoint_temperature_mean', 295.0),
                '2m_dewpoint_temperature_max': monthly_avg.get('2m_dewpoint_temperature_max', 300.0),
                '10m_u_component_of_wind_mean': monthly_avg.get('10m_u_component_of_wind_mean', 0.0),
                '10m_v_component_of_wind_mean': monthly_avg.get('10m_v_component_of_wind_mean', 0.0),
                'surface_solar_radiation_downwards_mean': monthly_avg.get('surface_solar_radiation_downwards_mean', 0.0),
                'total_cloud_cover_mean': monthly_avg.get('total_cloud_cover_mean', 0.0),
                'utci_mean': monthly_avg.get('utci_mean', 300.0),
            }
            # Add min/max approximations
            for key in ['10m_u_component_of_wind', '10m_v_component_of_wind']:
                avg_weather[f'{key}_min'] = avg_weather[f'{key}_mean']
                avg_weather[f'{key}_max'] = avg_weather[f'{key}_mean']
            avg_weather['surface_solar_radiation_downwards_min'] = 0.0
            avg_weather['surface_solar_radiation_downwards_max'] = avg_weather['surface_solar_radiation_downwards_mean'] * 2
            avg_weather['total_cloud_cover_min'] = 0.0
            avg_weather['total_cloud_cover_max'] = avg_weather['total_cloud_cover_mean']
            avg_weather['utci_min'] = avg_weather['2m_temperature_min']
            avg_weather['utci_max'] = avg_weather['2m_temperature_max']
            
            forecast_data.append(avg_weather)
    
    return pd.DataFrame(forecast_data)


def get_weather_forecast(state_name, start_date, num_days, use_api=True):
    """
    Main function to get weather forecast with automatic fallback.
    
    This function tries to get weather data from API first, and falls back
    to climatology if API fails. This ensures the app always has data to work with.
    
    Parameters:
        state_name (str): Name of the Indian state
        start_date (datetime.date): Start date for forecast
        num_days (int): Number of days to forecast
        use_api (bool): If True, try API first. If False, use climatology directly.
        
    Returns:
        pd.DataFrame: Weather forecast data
    """
    if use_api:
        try:
            # Try Open-Meteo API first (free, no key needed)
            return get_weather_forecast_openmeteo(state_name, start_date, num_days)
        except Exception as e:
            # If API fails, fall back to climatology
            st.warning(f"⚠️ Weather API failed: {str(e)}. Using historical averages...")
            return get_weather_forecast_climatology(state_name, start_date, num_days)
    else:
        # Use climatology directly
        return get_weather_forecast_climatology(state_name, start_date, num_days)

# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

def engineer_features(df, state_name):
    """
    Engineer features exactly as done in training.
    
    This function must match your notebook's feature engineering pipeline exactly.
    It creates calendar features, holiday flags, cooling degree days, state/region
    encodings, and other derived features that the model expects.
    
    Parameters:
        df (pd.DataFrame): Weather data with date column
        state_name (str): Name of the state (for state encoding)
        
    Returns:
        pd.DataFrame: Data with engineered features added
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Calendar features - extract time-based information
    df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['dayofweek'] = df['date'].dt.dayofweek  # Duplicate for model compatibility
    df['month'] = df['date'].dt.month  # 1-12
    df['year'] = df['date'].dt.year
    df['day_of_year'] = df['date'].dt.dayofyear  # 1-365
    
    # Weekend flags
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Season encoding (as done in notebook)
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Summer'
        elif month in [6, 7, 8, 9]:
            return 'Monsoon'
        else:
            return 'Autumn'
    
    df['season'] = df['month'].apply(get_season)
    
    # Holiday detection (India)
    india_holidays = holidays.India(years=range(2010, 2030))
    df['holiday_name'] = df['date'].apply(lambda x: india_holidays.get(x))
    df['holiday_name'] = df['holiday_name'].fillna('No Holiday')
    df['is_holiday'] = (df['holiday_name'] != 'No Holiday').astype(int)
    
    # State mapping for region assignment
    state_to_region = {
        'Arunachal Pradesh': 'ner', 'Assam': 'ner', 'Bihar': 'er', 'Chhattisgarh': 'er',
        'Goa': 'wr', 'Gujarat': 'wr', 'Haryana': 'nr', 'Himachal Pradesh': 'nr',
        'Jammu and Kashmir': 'nr', 'Jharkhand': 'er', 'Karnataka': 'sr', 'Kerala': 'sr',
        'Madhya Pradesh': 'wr', 'Maharashtra': 'wr', 'Manipur': 'ner', 'Meghalaya': 'ner',
        'Mizoram': 'ner', 'Nagaland': 'ner', 'Odisha': 'er', 'Punjab': 'nr',
        'Rajasthan': 'nr', 'Sikkim': 'ner', 'Tamil Nadu': 'sr', 'Telangana': 'sr',
        'Tripura': 'ner', 'Uttar Pradesh': 'nr', 'Uttarakhand': 'nr', 'West Bengal': 'er',
        'NCT of Delhi': 'nr', 'Puducherry': 'sr', 'Chandigarh': 'nr',
        'Dadra and Nagar Haveli': 'wr', 'Daman and Diu': 'wr', 'Lakshadweep': 'sr',
        'Andaman and Nicobar': 'sr'
    }
    
    # Normalize state name (handle variations)
    normalized_state = state_name
    if state_name == 'Jammu and Kashmir':
        normalized_state = 'J&K'
    
    df['states'] = normalized_state
    df['region'] = state_to_region.get(state_name, 'nr')
    
    # Cooling Degree Days (CDD) - using threshold from notebook
    threshold_temp = 297.15  # 24°C in Kelvin (from notebook: (temp_mean - 24).clip(lower=0))
    df['cdd'] = np.maximum(0, df['2m_temperature_mean'] - threshold_temp)
    df['CDD'] = df['cdd']  # Alternative name
    
    # Extreme heat flag (using quantile-based threshold approach from notebook)
    # For forecasting, use a fixed threshold based on typical 95th percentile
    heat_threshold = df['2m_temperature_mean'].quantile(0.95) if len(df) > 0 else 305.0
    df['extreme_heat'] = (df['2m_temperature_mean'] > heat_threshold).astype(int)
    df['extreme_heat_flag'] = df['extreme_heat'].astype(int)
    
    # Temperature interaction features
    df['temp_min_heat_interaction'] = df['2m_temperature_min'] * df['extreme_heat_flag']
    df['temp_mean_heat_interaction'] = df['2m_temperature_mean'] * df['extreme_heat_flag']
    df['temp_max_heat_interaction'] = df['2m_temperature_max'] * df['extreme_heat_flag']
    
    # Historical features (for forecasting, use climatology/defaults)
    # These would normally come from historical data, but for forecasting we use defaults
    df['generation_mu'] = 0.0  # Placeholder - would need historical data
    df['energymet_7d_avg'] = 0.0  # Placeholder
    df['generation_7d_avg'] = 0.0  # Placeholder
    df['monthly_generation_avg'] = 0.0  # Placeholder
    df['monthly_energy_avg'] = 0.0  # Placeholder
    df['monthly_rolling_mean'] = 0.0  # Placeholder
    df['demand_rolling'] = 0.0  # Placeholder
    df['gen_rolling'] = 0.0  # Placeholder
    df['max.demand met during the day(mw)'] = 0.0  # Placeholder
    df['state_share'] = 0.0  # Placeholder
    
    # Calculate state_30d_baseline - CRITICAL: Model expects this as a feature
    # Load historical demand to calculate baseline
    historical_demand = load_historical_demand(state_name)
    if historical_demand is not None and len(historical_demand) > 0:
        # Calculate 30-day baseline for each date
        df['state_30d_baseline'] = df['date'].apply(
            lambda d: calculate_state_30d_baseline(historical_demand, d)
        )
        # Fill NaN with mean baseline if any dates don't have enough history
        if df['state_30d_baseline'].isna().any():
            mean_baseline = df['state_30d_baseline'].mean() if df['state_30d_baseline'].notna().any() else historical_demand['actual_demand_MU'].mean()
            df['state_30d_baseline'] = df['state_30d_baseline'].fillna(mean_baseline)
    else:
        # If no historical demand available, use a default (but this is not ideal)
        df['state_30d_baseline'] = 100.0  # Default fallback
    
    # Temperature range
    df['temp_range'] = df['2m_temperature_max'] - df['2m_temperature_min']
    df['temp_range_heat'] = df['temp_range'] * df['extreme_heat_flag']
    
    # Lag features (for forecasting, these would be NaN, so we'll use recent values or defaults)
    # Since we're forecasting future dates, we can't have true lags
    # Use current values as approximation or set to 0
    for lag in [1, 3, 7]:
        df[f'temp_mean_lag{lag}'] = df['2m_temperature_mean']  # Use current as approximation
    
    # Rolling features (similar issue - use current values)
    df['temp_mean_roll3'] = df['2m_temperature_mean']
    df['temp_mean_roll7'] = df['2m_temperature_mean']
    df['temp_max_roll3'] = df['2m_temperature_max']
    df['temp_max_roll7'] = df['2m_temperature_max']
    df['temp_min_roll3'] = df['2m_temperature_min']
    df['temp_min_roll7'] = df['2m_temperature_min']
    
    return df


def prepare_features_for_prediction(weather_df, state_name, model=None):
    """
    Prepare the exact feature set the model expects.
    
    This function ensures that the features match exactly what the model was
    trained on. The feature names and order must match the training data.
    
    Parameters:
        weather_df (pd.DataFrame): Raw weather data
        state_name (str): Name of the state
        
    Returns:
        tuple: (X_features, df_features)
            - X_features: DataFrame with only model features in correct order
            - df_features: Full dataframe with all engineered features
    """
    # Engineer features
    df_features = engineer_features(weather_df, state_name)
    
    # Get all categorical columns for one-hot encoding
    categorical_cols = ['states', 'region', 'season', 'holiday_name']
    
    # Store original values before encoding (for manual setting)
    season_values = df_features['season'].copy() if 'season' in df_features.columns else None
    holiday_values = df_features['holiday_name'].copy() if 'holiday_name' in df_features.columns else None
    
    # Create one-hot encodings (matching notebook approach)
    for col in categorical_cols:
        if col in df_features.columns:
            dummies = pd.get_dummies(df_features[col], prefix=col)
            df_features = pd.concat([df_features, dummies], axis=1)
    
    # Drop original categorical columns (keep encoded versions)
    df_features = df_features.drop(columns=categorical_cols, errors='ignore')
    
    # Get the model's expected feature names in EXACT order from training
    # This order is critical - must match exactly what the model was trained on
    # Order from the error message shows the exact sequence
    expected_features_base = [
        '10m_u_component_of_wind_min', '10m_u_component_of_wind_mean', '10m_u_component_of_wind_max',
        '10m_v_component_of_wind_min', '10m_v_component_of_wind_mean', '10m_v_component_of_wind_max',
        '2m_temperature_min', '2m_temperature_mean', '2m_temperature_max',
        '2m_dewpoint_temperature_min', '2m_dewpoint_temperature_mean', '2m_dewpoint_temperature_max',
        'surface_solar_radiation_downwards_min', 'surface_solar_radiation_downwards_mean', 'surface_solar_radiation_downwards_max',
        'total_cloud_cover_min', 'total_cloud_cover_mean', 'total_cloud_cover_max',
        'utci_min', 'utci_mean', 'utci_max',
        'max.demand met during the day(mw)', 'generation_mu', 'state_share',
        'month', 'day_of_week', 'is_weekend', 'energymet_7d_avg', 'generation_7d_avg',
        'monthly_generation_avg', 'monthly_energy_avg', 'dayofweek', 'weekend',
        'monthly_rolling_mean', 'demand_rolling', 'gen_rolling', 'year',
        'extreme_heat', 'extreme_heat_flag', 'temp_min_heat_interaction',
        'temp_mean_heat_interaction', 'temp_max_heat_interaction'
    ]
    
    # States in exact order from error message
    all_states_ordered = [
        'Arunachal Pradesh', 'Assam', 'Bihar', 'Chandigarh', 'Chhattisgarh',
        'Goa', 'Gujarat', 'Haryana', 'J&K', 'Jharkhand', 'Karnataka', 'Kerala',
        'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha',
        'Puducherry', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana',
        'Tripura', 'Uttarakhand', 'West Bengal'
    ]
    
    # Regions in exact order
    regions_ordered = ['ner', 'nr', 'sr', 'wr']
    
    # Seasons in exact order
    seasons_ordered = ['Monsoon', 'Summer', 'Winter']
    
    # Holidays in exact order
    holidays_ordered = ['Dussehra', 'Guru Nanak Jayanti', 'No Holiday']
    
    # Build complete feature list in EXACT order expected by model
    # If model is provided, use its feature_names_in_ for exact order
    if model is not None and hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
    else:
        # Fallback to hardcoded order from error message
        expected_features = expected_features_base.copy()
        expected_features.extend([f'states_{s}' for s in all_states_ordered])
        expected_features.extend([f'region_{r}' for r in regions_ordered])
        expected_features.extend([f'season_{s}' for s in seasons_ordered])
        expected_features.extend([f'holiday_name_{h}' for h in holidays_ordered])
    
    # Add state one-hot columns (all possible states from training) in exact order
    all_states_ordered = [
        'Arunachal Pradesh', 'Assam', 'Bihar', 'Chandigarh', 'Chhattisgarh',
        'Goa', 'Gujarat', 'Haryana', 'J&K', 'Jharkhand', 'Karnataka', 'Kerala',
        'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha',
        'Puducherry', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana',
        'Tripura', 'Uttarakhand', 'West Bengal'
    ]
    for state in all_states_ordered:
        col_name = f'states_{state}'
        if col_name not in df_features.columns:
            df_features[col_name] = 0
        # Set the current state to 1
        if state_name == state or (state_name == 'Jammu and Kashmir' and state == 'J&K'):
            df_features[col_name] = 1
    
    # Add region one-hot columns in exact order (note: model expects only ner, nr, sr, wr - no er)
    # Get region before dropping categorical columns
    state_to_region = {
        'Arunachal Pradesh': 'ner', 'Assam': 'ner', 'Bihar': 'er', 'Chhattisgarh': 'er',
        'Goa': 'wr', 'Gujarat': 'wr', 'Haryana': 'nr', 'Himachal Pradesh': 'nr',
        'Jammu and Kashmir': 'nr', 'Jharkhand': 'er', 'Karnataka': 'sr', 'Kerala': 'sr',
        'Madhya Pradesh': 'wr', 'Maharashtra': 'wr', 'Manipur': 'ner', 'Meghalaya': 'ner',
        'Mizoram': 'ner', 'Nagaland': 'ner', 'Odisha': 'er', 'Punjab': 'nr',
        'Rajasthan': 'nr', 'Sikkim': 'ner', 'Tamil Nadu': 'sr', 'Telangana': 'sr',
        'Tripura': 'ner', 'Uttar Pradesh': 'nr', 'Uttarakhand': 'nr', 'West Bengal': 'er',
        'NCT of Delhi': 'nr', 'Puducherry': 'sr', 'Chandigarh': 'nr',
        'Dadra and Nagar Haveli': 'wr', 'Daman and Diu': 'wr', 'Lakshadweep': 'sr',
        'Andaman and Nicobar': 'sr'
    }
    current_region = state_to_region.get(state_name, 'nr')
    # Map 'er' to closest region (model doesn't have 'er', so map to 'nr')
    if current_region == 'er':
        current_region = 'nr'
    regions_ordered = ['ner', 'nr', 'sr', 'wr']  # Exact order from error message
    for region in regions_ordered:
        col_name = f'region_{region}'
        if col_name not in df_features.columns:
            df_features[col_name] = 0
        # Set current region to 1
        if region == current_region:
            df_features[col_name] = 1
    
    # Add season one-hot columns and set the correct one (exact order)
    seasons_ordered = ['Monsoon', 'Summer', 'Winter']
    for season in seasons_ordered:
        col_name = f'season_{season}'
        if col_name not in df_features.columns:
            df_features[col_name] = 0
        # Set current season to 1 using stored values
        if season_values is not None:
            df_features[col_name] = (season_values == season).astype(int)
    
    # Add holiday name one-hot columns and set the correct one (exact order)
    holidays_ordered = ['Dussehra', 'Guru Nanak Jayanti', 'No Holiday']
    for holiday in holidays_ordered:
        col_name = f'holiday_name_{holiday}'
        if col_name not in df_features.columns:
            df_features[col_name] = 0
        # Set current holiday to 1 using stored values
        if holiday_values is not None:
            # Check if holiday name contains the holiday string
            df_features[col_name] = holiday_values.apply(
                lambda x: 1 if holiday in str(x) else 0
            )
    
    # Select features in the EXACT order the model expects (already built in expected_features)
    all_possible_features = expected_features
    
    # Create feature matrix with only expected features, in correct order
    # Fill missing features with 0
    X = pd.DataFrame(index=df_features.index)
    for feat in all_possible_features:
        if feat in df_features.columns:
            X[feat] = df_features[feat]
        else:
            X[feat] = 0
    
    return X, df_features

# ============================================================================
# WEATHER IMPACT VISUALIZATION FUNCTIONS
# ============================================================================

def get_feature_importance(model, top_n=15, filter_weather_only=False):
    """
    Extract feature importance from XGBoost model.
    
    Parameters:
        model: Trained XGBoost model
        top_n: Number of top features to return
        filter_weather_only: If True, only return weather-related features
        
    Returns:
        dict: Dictionary with feature names as keys and importance scores as values
    """
    if model is None or not hasattr(model, 'feature_importances_'):
        return None
    
    try:
        feature_names = model.feature_names_in_
        importances = model.feature_importances_
        
        # Create dictionary of feature importance
        importance_dict = dict(zip(feature_names, importances))
        
        # Filter to weather-only features if requested
        if filter_weather_only:
            weather_categories = ['Temperature', 'Humidity', 'Weather']
            weather_features = {}
            for feat, imp in importance_dict.items():
                category = categorize_feature(feat)
                if category in weather_categories:
                    weather_features[feat] = imp
            importance_dict = weather_features
        
        # Sort by importance (descending)
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        # Return top N features
        top_features = dict(list(sorted_importance.items())[:top_n])
        
        return top_features
    except Exception as e:
        return None


def categorize_feature(feature_name):
    """
    Categorize a feature into a group for color coding.
    
    Parameters:
        feature_name: Name of the feature
        
    Returns:
        str: Category name
    """
    feature_lower = feature_name.lower()
    
    if any(x in feature_lower for x in ['temperature', 'temp', 'cdd']):
        return 'Temperature'
    elif any(x in feature_lower for x in ['dewpoint', 'utci', 'humidity']):
        return 'Humidity'
    elif any(x in feature_lower for x in ['month', 'day', 'weekend', 'holiday', 'season']):
        return 'Calendar'
    elif any(x in feature_lower for x in ['solar', 'radiation', 'cloud', 'wind']):
        return 'Weather'
    elif any(x in feature_lower for x in ['generation', 'energy', 'demand', 'rolling', 'avg']):
        return 'Historical'
    elif any(x in feature_lower for x in ['state', 'region']):
        return 'Location'
    else:
        return 'Other'


def plot_demand_temperature_overlay(results_df, weather_df, df_features=None):
    """
    Create dual-axis plot showing demand and temperature on the same timeline.
    Optionally includes baseline (7-day average) for comparison.
    
    Parameters:
        results_df: DataFrame with forecasted demand and dates
        weather_df: DataFrame with temperature data
        df_features: Optional DataFrame with features to calculate baseline
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Merge dataframes on date
    merged_df = pd.merge(results_df, weather_df, on='date', how='inner')
    
    # Convert temperature from Kelvin to Celsius
    temp_mean_c = merged_df['2m_temperature_mean'] - 273.15
    temp_max_c = merged_df['2m_temperature_max'] - 273.15
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add baseline (7-day rolling average) if features available
    if df_features is not None and 'energymet_7d_avg' in df_features.columns:
        # Merge to get baseline
        baseline_df = pd.merge(results_df, df_features[['date', 'energymet_7d_avg']], on='date', how='inner')
        if len(baseline_df) > 0 and 'energymet_7d_avg' in baseline_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=baseline_df['date'],
                    y=baseline_df['energymet_7d_avg'],
                    mode='lines',
                    name='Baseline (7-day avg, no weather)',
                    line=dict(color='gray', width=2, dash='dash'),
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Baseline: %{y:.2f}<extra></extra>'
                ),
                secondary_y=False,
            )
    
    # Add demand trace (primary axis)
    fig.add_trace(
        go.Scatter(
            x=merged_df['date'],
            y=merged_df['forecasted_demand_MU'],
            mode='lines+markers',
            name='Forecasted Demand (with weather)',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Demand: %{y:.2f} (normalized)<extra></extra>'
        ),
        secondary_y=False,
    )
    
    # Add temperature mean trace (secondary axis)
    fig.add_trace(
        go.Scatter(
            x=merged_df['date'],
            y=temp_mean_c,
            mode='lines+markers',
            name='Temperature (Mean)',
            line=dict(color='#ff7f0e', width=2, dash='solid'),
            marker=dict(size=5),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Temp: %{y:.1f}°C<extra></extra>'
        ),
        secondary_y=True,
    )
    
    # Add temperature max trace (secondary axis)
    fig.add_trace(
        go.Scatter(
            x=merged_df['date'],
            y=temp_max_c,
            mode='lines+markers',
            name='Temperature (Max)',
            line=dict(color='#d62728', width=2, dash='dash'),
            marker=dict(size=5),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Temp Max: %{y:.1f}°C<extra></extra>'
        ),
        secondary_y=True,
    )
    
    # Calculate correlation
    correlation = merged_df['forecasted_demand_MU'].corr(temp_mean_c)
    
    # Set x-axis title
    fig.update_xaxes(title_text="Date")
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Demand (normalized units)", secondary_y=False)
    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=True)
    
    # Update layout
    fig.update_layout(
        title=f"Demand vs Temperature Overlay (Correlation: {correlation:.3f})",
        hovermode='x unified',
        height=500,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def plot_cdd_timeline(results_df, df_features):
    """
    Plot CDD (Cooling Degree Days) as bars with demand overlay.
    
    Parameters:
        results_df: DataFrame with forecasted demand and dates
        df_features: DataFrame with engineered features including CDD
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Merge dataframes on date - handle duplicate columns with suffixes
    merged_df = pd.merge(results_df, df_features, on='date', how='inner', suffixes=('_results', '_features'))
    
    # Resolve CDD column - handle merge suffixes
    if 'cdd_features' in merged_df.columns:
        cdd_values = merged_df['cdd_features']
    elif 'cdd_results' in merged_df.columns:
        cdd_values = merged_df['cdd_results']
    elif 'cdd' in merged_df.columns:
        cdd_values = merged_df['cdd']
    else:
        # Fallback: use from results_df
        cdd_values = results_df['cdd'] if 'cdd' in results_df.columns else pd.Series([0.0] * len(merged_df))
    
    # Convert CDD from Kelvin to Celsius (if needed, CDD is already in correct units)
    # CDD is calculated as (temp_mean - 297.15), so it's already in Kelvin units
    # For display, we can show it as-is or convert
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Resolve extreme_heat column - handle merge suffixes
    if 'extreme_heat_features' in merged_df.columns:
        extreme_heat_col = 'extreme_heat_features'
    elif 'extreme_heat_results' in merged_df.columns:
        extreme_heat_col = 'extreme_heat_results'
    elif 'extreme_heat' in merged_df.columns:
        extreme_heat_col = 'extreme_heat'
    else:
        extreme_heat_col = None
        # Create dummy column if missing
        merged_df['extreme_heat'] = 0
    
    # Use resolved column or fallback
    heat_col_for_color = extreme_heat_col if extreme_heat_col else 'extreme_heat'
    
    # Add CDD bars (primary axis)
    fig.add_trace(
        go.Bar(
            x=merged_df['date'],
            y=cdd_values,
            name='CDD',
            marker=dict(
                color=merged_df[heat_col_for_color],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Extreme Heat", x=1.15),
                cmin=0,
                cmax=1
            ),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>CDD: %{y:.2f}°C<extra></extra>'
        ),
        secondary_y=False,
    )
    
    # Add demand line (secondary axis)
    fig.add_trace(
        go.Scatter(
            x=merged_df['date'],
            y=merged_df['forecasted_demand_MU'],
            mode='lines+markers',
            name='Forecasted Demand',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Demand: %{y:.2f} MU<extra></extra>'
        ),
        secondary_y=True,
    )
    
    # Highlight extreme heat days with markers
    if extreme_heat_col:
        extreme_heat_days = merged_df[merged_df[extreme_heat_col] == 1]
    else:
        extreme_heat_days = pd.DataFrame()  # Empty dataframe if no extreme_heat column
    if len(extreme_heat_days) > 0:
        fig.add_trace(
            go.Scatter(
                x=extreme_heat_days['date'],
                y=extreme_heat_days['forecasted_demand_MU'],
                mode='markers',
                name='Extreme Heat',
                marker=dict(
                    symbol='triangle-up',
                    size=12,
                    color='red',
                    line=dict(width=2, color='darkred')
                ),
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Extreme Heat Day<br>Demand: %{y:.2f} MU<extra></extra>'
            ),
            secondary_y=True,
        )
    
    # Set x-axis title
    fig.update_xaxes(title_text="Date")
    
    # Set y-axes titles
    fig.update_yaxes(title_text="CDD (°C)", secondary_y=False)
    fig.update_yaxes(title_text="Demand (normalized units)", secondary_y=True)
    
    # Update layout
    fig.update_layout(
        title="Cooling Degree Days (CDD) Timeline with Demand Overlay",
        hovermode='x unified',
        height=500,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        barmode='overlay'
    )
    
    return fig


def plot_feature_importance(importance_dict, chart_type='bar'):
    """
    Plot feature importance as horizontal bar chart with color coding by category.
    
    Parameters:
        importance_dict: Dictionary with feature names and importance scores
        chart_type: Type of chart ('bar' or 'horizontal_bar')
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    if importance_dict is None or len(importance_dict) == 0:
        return None
    
    # Prepare data
    features = list(importance_dict.keys())
    importances = list(importance_dict.values())
    categories = [categorize_feature(feat) for feat in features]
    
    # Create color map for categories
    category_colors = {
        'Temperature': '#ff7f0e',
        'Humidity': '#2ca02c',
        'Calendar': '#9467bd',
        'Weather': '#8c564b',
        'Historical': '#e377c2',
        'Location': '#7f7f7f',
        'Other': '#bcbd22'
    }
    
    colors = [category_colors.get(cat, '#bcbd22') for cat in categories]
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=importances,
        y=features,
        orientation='h',
        marker=dict(color=colors),
        text=[f'{imp:.4f}' for imp in importances],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title="Model Dependency (Relative Influence)",
        xaxis_title="Relative Influence Score",
        yaxis_title="Feature",
        height=max(400, len(features) * 25),
        template='plotly_white',
        yaxis=dict(autorange="reversed")
    )
    
    # Add legend for categories
    unique_categories = list(set(categories))
    legend_items = []
    for cat in unique_categories:
        legend_items.append(
            dict(
                label=cat,
                marker=dict(color=category_colors.get(cat, '#bcbd22'))
            )
        )
    
    return fig


def generate_weather_insight(results_df, df_features, state_name):
    """
    Generate natural language insight about weather impact on demand.
    
    Parameters:
        results_df: DataFrame with forecasted demand and dates
        df_features: DataFrame with engineered features
        state_name: Name of the state
        
    Returns:
        str: Natural language insight text
    """
    # Merge dataframes - use suffixes to handle duplicate columns
    # Since both results_df and df_features have 'cdd', we'll get 'cdd_results' and 'cdd_features'
    merged_df = pd.merge(results_df, df_features, on='date', how='inner', suffixes=('_results', '_features'))
    
    if len(merged_df) == 0:
        return "No data available for insight generation."
    
    # Resolve CDD column - check for merged suffixes first, then fallback
    if 'cdd_features' in merged_df.columns:
        merged_df['cdd'] = merged_df['cdd_features']  # Create unified 'cdd' column
    elif 'cdd_results' in merged_df.columns:
        merged_df['cdd'] = merged_df['cdd_results']  # Create unified 'cdd' column
    elif 'cdd' not in merged_df.columns:
        # Fallback: try to get from original dataframes
        if 'cdd' in results_df.columns:
            merged_df['cdd'] = results_df['cdd'].values
        elif 'cdd' in df_features.columns:
            merged_df['cdd'] = df_features['cdd'].values
    
    # Convert temperature to Celsius
    temp_mean_c = merged_df['2m_temperature_mean'] - 273.15
    temp_max_c = merged_df['2m_temperature_max'] - 273.15
    
    # Find peak demand day
    peak_idx = merged_df['forecasted_demand_MU'].idxmax()
    peak_demand = merged_df.loc[peak_idx, 'forecasted_demand_MU']
    peak_date = merged_df.loc[peak_idx, 'date']
    peak_temp = temp_mean_c.loc[peak_idx]
    peak_temp_max = temp_max_c.loc[peak_idx]
    peak_cdd = merged_df.loc[peak_idx, 'cdd'] if 'cdd' in merged_df.columns else 0.0
    
    # Calculate correlations
    temp_demand_corr = merged_df['forecasted_demand_MU'].corr(temp_mean_c)
    
    # Use resolved CDD column
    if 'cdd' in merged_df.columns:
        cdd_demand_corr = merged_df['forecasted_demand_MU'].corr(merged_df['cdd'])
        avg_cdd = merged_df['cdd'].mean()
        max_cdd = merged_df['cdd'].max()
    else:
        cdd_demand_corr = 0.0
        avg_cdd = 0.0
        max_cdd = 0.0
    
    # Count extreme heat days - handle merge suffixes
    if 'extreme_heat_features' in merged_df.columns:
        extreme_heat_count = merged_df['extreme_heat_features'].sum()
    elif 'extreme_heat_results' in merged_df.columns:
        extreme_heat_count = merged_df['extreme_heat_results'].sum()
    elif 'extreme_heat' in merged_df.columns:
        extreme_heat_count = merged_df['extreme_heat'].sum()
    else:
        extreme_heat_count = 0
    
    total_days = len(merged_df)
    extreme_heat_pct = (extreme_heat_count / total_days * 100) if total_days > 0 else 0
    
    # Calculate average temperature
    avg_temp = temp_mean_c.mean()
    max_temp = temp_max_c.max()
    
    # Generate insight
    insights = []
    
    # Peak demand insight
    insights.append(
        f"**Peak Demand**: Demand peaks on {peak_date.strftime('%B %d, %Y')} at {peak_temp:.1f}°C "
        f"(max {peak_temp_max:.1f}°C) with CDD of {peak_cdd:.1f}°C, coinciding with elevated demand of {peak_demand:.2f} (normalized units)."
    )
    
    # Correlation insight
    if abs(temp_demand_corr) > 0.5:
        corr_strength = "strong" if abs(temp_demand_corr) > 0.7 else "moderate"
        corr_direction = "positive" if temp_demand_corr > 0 else "negative"
        insights.append(
            f"**Temperature Impact**: {corr_strength.capitalize()} {corr_direction} correlation ({temp_demand_corr:.2f}) "
            f"between temperature and demand, suggesting temperature influences demand patterns."
        )
    
    # Extreme heat insight - with threshold explanation
    if extreme_heat_count > 0:
        insights.append(
            f"**Extreme Heat Risk**: {extreme_heat_count} extreme heat day(s) identified ({extreme_heat_pct:.0f}% of period), "
            f"with average CDD of {avg_cdd:.1f}°C (max {max_cdd:.1f}°C), potentially amplifying cooling load. "
            f"*Extreme heat defined as days where max temperature exceeds the 95th percentile for {state_name}.*"
        )
    else:
        insights.append(
            f"**Temperature Conditions**: No extreme heat days in this period. Average temperature is {avg_temp:.1f}°C "
            f"(max {max_temp:.1f}°C) with average CDD of {avg_cdd:.1f}°C. "
            f"*Extreme heat threshold: 95th percentile of max temperature for {state_name}.*"
        )
    
    # CDD insight
    if cdd_demand_corr > 0.3:
        insights.append(
            f"**Cooling Load**: CDD shows {cdd_demand_corr:.2f} correlation with demand, suggesting that "
            f"cooling degree days influence electricity consumption patterns in {state_name}."
        )
    
    return " ".join(insights)


def get_seasonal_comparison(state_name, current_season, historical_df=None):
    """
    Get seasonal comparison data for a state.
    
    Parameters:
        state_name: Name of the state
        current_season: Current season ('Summer', 'Monsoon', 'Winter', 'Autumn')
        historical_df: Optional historical dataframe (if None, will load)
        
    Returns:
        dict: Dictionary with seasonal statistics
    """
    if historical_df is None:
        historical_df = load_historical_data(state_name)
    
    if historical_df is None or len(historical_df) == 0:
        return None
    
    # Engineer features to get season
    df_features = engineer_features(historical_df, state_name)
    
    # Group by season
    seasonal_stats = {}
    for season in ['Summer', 'Monsoon', 'Winter', 'Autumn']:
        season_data = df_features[df_features['season'] == season]
        if len(season_data) > 0:
            temp_mean_c = (season_data['2m_temperature_mean'] - 273.15).mean()
            temp_max_c = (season_data['2m_temperature_max'] - 273.15).mean()
            cdd_mean = season_data['cdd'].mean()
            extreme_heat_pct = (season_data['extreme_heat'].sum() / len(season_data) * 100)
            
            seasonal_stats[season] = {
                'temp_mean': temp_mean_c,
                'temp_max': temp_max_c,
                'cdd_mean': cdd_mean,
                'extreme_heat_pct': extreme_heat_pct,
                'count': len(season_data)
            }
    
    return seasonal_stats


# ============================================================================
# RAG FUNCTIONS
# ============================================================================

def generate_forecast_summary(results_df, state, horizon_days, df_features=None, metadata=None):
    """
    Generate structured forecast summary for RAG system.
    Creates a single locked object per forecast with exact values - no recomputation allowed.
    
    Parameters:
        results_df: DataFrame with forecast results
        state: State name
        horizon_days: Forecast horizon in days
        df_features: Optional DataFrame with features
        metadata: Optional model metadata for RMSE and confidence_level
        
    Returns:
        Dictionary containing forecast summary with locked schema
    """
    if results_df is None or len(results_df) == 0:
        return None
    
    # Get state-specific RMSE from metadata
    state_rmse = None
    confidence_level = 0.9  # Default
    if metadata:
        state_rmse_dict = metadata.get('state_rmse', {})
        # Handle state name variations
        state_rmse = state_rmse_dict.get(state)
        if state_rmse is None and state == "Jammu and Kashmir":
            state_rmse = state_rmse_dict.get("J&K")
        confidence_level = metadata.get('confidence_level', 0.9)
    
    # Identify high-risk days (extreme heat or very high demand)
    high_risk_dates = []
    extreme_heat_dates = []
    
    if 'extreme_heat' in results_df.columns:
        # Get extreme heat dates
        extreme_heat_rows = results_df[results_df['extreme_heat'] == 1]
        extreme_heat_dates = [row['date'].strftime('%Y-%m-%d') for _, row in extreme_heat_rows.iterrows()]
        
        # High-risk days: extreme heat OR top 10% demand
        high_demand_threshold = results_df['forecasted_demand_MU'].quantile(0.9)
        high_risk_rows = results_df[
            (results_df['extreme_heat'] == 1) | 
            (results_df['forecasted_demand_MU'] >= high_demand_threshold)
        ]
        high_risk_dates = [row['date'].strftime('%Y-%m-%d') for _, row in high_risk_rows.iterrows()]
    
    # Create locked forecast summary object
    summary = {
        'state': state,
        'horizon_days': int(horizon_days),
        'start_date': results_df['date'].min().strftime('%Y-%m-%d'),
        'end_date': results_df['date'].max().strftime('%Y-%m-%d'),
        'avg_demand': round(float(results_df['forecasted_demand_MU'].mean()), 1),
        'peak_demand': round(float(results_df['forecasted_demand_MU'].max()), 1),
        'peak_date': results_df.loc[results_df['forecasted_demand_MU'].idxmax(), 'date'].strftime('%Y-%m-%d'),
        'min_demand': round(float(results_df['forecasted_demand_MU'].min()), 1),
        'high_risk_days_count': len(high_risk_dates),
        'high_risk_dates': high_risk_dates,
        'extreme_heat_dates': extreme_heat_dates,
        'rmse': round(float(state_rmse), 2) if state_rmse is not None else None,
        'confidence_level': confidence_level
    }
    
    return summary


@st.cache_resource
def initialize_rag_system():
    """
    Initialize RAG system with knowledge base.
    Creates vector store and loads initial knowledge documents.
    
    Returns:
        Tuple of (VectorStore, RAGEngine) or (None, None) if initialization fails
    """
    try:
        # Create data directory for RAG index if it doesn't exist
        rag_index_dir = "data/rag_index"
        os.makedirs(rag_index_dir, exist_ok=True)
        index_path = os.path.join(rag_index_dir, "knowledge_base.index")
        
        # Initialize vector store
        # Dimension 384 for sentence-transformers (all-MiniLM-L6-v2)
        # Will auto-adjust if using OpenAI embeddings (1536)
        vector_store = VectorStore(dimension=384, index_path=index_path)
        
        # Check if index already exists and has documents
        if vector_store.get_size() > 0:
            # Index already loaded
            rag_engine = RAGEngine(vector_store)
            return vector_store, rag_engine
        
        # Build knowledge base from model metadata
        try:
            metadata = load_model_metadata_rag("model_metadata.json")
            model_metrics_docs = build_model_metrics_documents(metadata)
            
            # Get model for feature importance
            model = load_model()
            if model:
                feature_importance = get_feature_importance(model, top_n=15)
                if feature_importance:
                    feature_docs = build_feature_importance_documents(feature_importance)
                    model_metrics_docs.extend(feature_docs)
            
            # Embed and add documents
            if model_metrics_docs:
                texts = [doc['text'] for doc in model_metrics_docs]
                metadata_list = [doc['metadata'] for doc in model_metrics_docs]
                
                try:
                    embeddings = get_embeddings(texts)
                    vector_store.add_documents(embeddings, texts, metadata_list)
                    vector_store.save_index()
                except Exception as e:
                    # If embedding fails (e.g., no API key), return None
                    st.warning(f"Could not initialize embeddings: {str(e)}")
                    return None, None
            
            # Initialize RAG engine
            rag_engine = RAGEngine(vector_store)
            
            return vector_store, rag_engine
            
        except FileNotFoundError as e:
            st.warning("Model metadata file not found. RAG system will work with forecast data only.")
            # Return empty vector store that can be populated with forecasts
            rag_engine = RAGEngine(vector_store)
            return vector_store, rag_engine
        except Exception as e:
            st.error(f"Error building knowledge base: {str(e)}")
            return None, None
            
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return None, None


# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================

# Title and header
st.title("⚡ Smart Energy Forecasting Platform")
st.markdown("**Stage 1.5: Forecast + Explain**")
st.markdown("Interactive dashboard for electricity demand forecasting using ML and real-time weather data")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("---")
    
    # Model status
    model = load_model()
    if model:
        st.success("✅ Model loaded successfully")
        st.caption("XGBoost model ready for predictions")
    else:
        st.error("❌ Model not found")
        st.caption("Please check model path")
    
    st.markdown("---")
    
    # Weather data source selection
    st.subheader("🌤️ Weather Data Source")
    use_weather_api = st.checkbox(
        "Use Weather API (Open-Meteo)",
        value=True,
        help="Use real-time weather API. If unchecked, uses historical averages."
    )
    
    st.markdown("---")
    st.caption("💡 Tip: Weather API provides more accurate forecasts but requires internet connection.")
    
    # RAG/AI Assistant status
    st.markdown("---")
    st.subheader("🤖 AI Assistant")
    
    # Check for sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        st.success("✅ Local embeddings available")
        st.caption("Using sentence-transformers (no API needed)")
    except ImportError:
        st.warning("⚠️ sentence-transformers not installed")
        st.caption("Install for free local embeddings: `pip install sentence-transformers`")
        with st.expander("Installation Instructions"):
            st.code("pip install sentence-transformers", language="bash")
            st.markdown("""
            **Why install this?**
            - Free local embeddings (no API costs)
            - Works offline
            - Faster than API calls
            - First download is ~90MB, then cached locally
            """)
    
    api_key = get_openrouter_api_key()
    if api_key:
        st.info("ℹ️ API key configured (optional - local embeddings preferred)")
    else:
        st.caption("💡 Tip: Local embeddings work without an API key")

# Main tabs
tab1, tab2, tab3 = st.tabs([
    "📊 Forecast", 
    "🌡️ Weather Impact", 
    "🤖 AI Energy Assistant"
])

# ============================================================================
# TAB 1: FORECAST
# ============================================================================
with tab1:
    st.header("Demand Forecasting")
    st.markdown("Generate electricity demand forecasts for any Indian state using ML predictions.")
    
    # Input controls
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        state = st.selectbox(
            "Select State",
            ALL_STATES,
            index=ALL_STATES.index("Kerala") if "Kerala" in ALL_STATES else 0,
            help="Choose the state for which you want to forecast electricity demand"
        )
    
    with col2:
        horizon_days = st.selectbox(
            "Forecast Horizon",
            [7, 14, 30],
            index=0,
            format_func=lambda x: f"{x} days",
            help="Number of days into the future to forecast"
        )
    
    with col3:
        st.write("")  # Spacing
        generate_btn = st.button("🚀 Generate Forecast", type="primary", use_container_width=True)
    
    # Forecast generation
    if generate_btn:
        if model is None:
            st.error("❌ Model not loaded. Cannot generate forecast.")
            st.stop()
        
        with st.spinner("🔄 Generating forecast... This may take a few seconds."):
            try:
                # Get start date (today)
                start_date = datetime.now().date()
                
                # Get weather forecast
                weather_forecast = get_weather_forecast(
                    state, 
                    start_date, 
                    horizon_days,
                    use_api=use_weather_api
                )
                
                if weather_forecast is None or len(weather_forecast) == 0:
                    st.error("❌ Failed to get weather data. Please try again.")
                    st.stop()
                
                # Prepare features for model (pass model to get exact feature order)
                X_forecast, df_features = prepare_features_for_prediction(
                    weather_forecast, state, model=model
                )
                
                # Make predictions using the model (model outputs log-space predictions)
                pred_log = model.predict(X_forecast)
                
                # #region agent log
                import json
                import os
                log_path = os.path.join(os.path.dirname(__file__), ".cursor", "debug.log")
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "deployment-debug", "hypothesisId": "A", "location": "app.py:1885", "message": "Model predictions (log-space)", "data": {"pred_log_mean": float(np.mean(pred_log)), "pred_log_shape": list(pred_log.shape), "state": state}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
                # #endregion
                
                # Convert from log space to absolute MU using denormalize_predictions
                historical_demand = load_historical_demand(state)
                
                # #region agent log
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "deployment-debug", "hypothesisId": "D", "location": "app.py:1889", "message": "Historical demand load result", "data": {"historical_demand_is_none": historical_demand is None, "historical_demand_len": len(historical_demand) if historical_demand is not None else 0, "state": state}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
                # #endregion
                
                # #region agent log
                first_date = pd.Timestamp(weather_forecast['date'].iloc[0]) if len(weather_forecast) > 0 else None
                current_date = pd.Timestamp.now()
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "deployment-debug", "hypothesisId": "B", "location": "app.py:1890", "message": "Date comparison check", "data": {"first_date": str(first_date) if first_date is not None else None, "current_date": str(current_date), "first_date_gt_current": bool(first_date > current_date) if first_date is not None else None, "dates_len": len(weather_forecast['date'])}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
                # #endregion
                
                denorm_result = denormalize_predictions(pred_log, weather_forecast['date'], state, historical_demand)
                
                # #region agent log
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "deployment-debug", "hypothesisId": "C", "location": "app.py:1891", "message": "denormalize_predictions result", "data": {"denorm_mean": float(np.mean(denorm_result)), "denorm_min": float(np.min(denorm_result)), "denorm_max": float(np.max(denorm_result)), "denorm_shape": list(denorm_result.shape)}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
                # #endregion
                
                # Create results dataframe with absolute MU values
                results_df = pd.DataFrame({
                    'date': weather_forecast['date'],
                    'forecasted_demand_MU': denorm_result,  # Absolute MU values after baseline multiplication
                    'temperature_mean': weather_forecast['2m_temperature_mean'],
                    'temperature_max': weather_forecast['2m_temperature_max'],
                    'cdd': df_features['cdd'],
                    'extreme_heat': df_features['extreme_heat'],
                    'is_holiday': df_features['is_holiday']
                })
                
                # Generate forecast summary with locked schema
                metadata = load_model_metadata()
                forecast_summary = generate_forecast_summary(
                    results_df,
                    state,
                    horizon_days,
                    df_features,
                    metadata=metadata
                )
                
                # Store forecast data in session state for Tab 2 and Tab 3
                st.session_state['last_forecast'] = {
                    'results_df': results_df,
                    'weather_forecast': weather_forecast,
                    'df_features': df_features,
                    'state': state,
                    'horizon_days': horizon_days,
                    'forecast_summary': forecast_summary  # Include locked summary
                }
                
                # #region agent log
                import json
                import os
                log_path = os.path.join(os.path.dirname(__file__), ".cursor", "debug.log")
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "deployment-debug", "hypothesisId": "A", "location": "app.py:1992", "message": "Final forecasted_demand_MU before assertion", "data": {"mean": float(results_df['forecasted_demand_MU'].mean()), "min": float(results_df['forecasted_demand_MU'].min()), "max": float(results_df['forecasted_demand_MU'].max()), "values_sample": [float(x) for x in results_df['forecasted_demand_MU'].head(3).tolist()]}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
                # #endregion
                
                # Fix 5: Defensive assertion to catch model-space leaks
                assert results_df['forecasted_demand_MU'].mean() > 10, \
                    f"Forecast values look like model-space, not MU. Mean: {results_df['forecasted_demand_MU'].mean():.2f}"
                
                # Sanity check: Print forecast range
                st.write("🔍 **Forecast range (MU):**", 
                        f"{results_df['forecasted_demand_MU'].min():.2f} - {results_df['forecasted_demand_MU'].max():.2f} MU")
                
                # Display success message
                st.success(f"✅ Forecast generated successfully for **{state}** ({horizon_days} days)")
                
                # Fix 3: Summary metrics must aggregate raw MU only
                st.subheader("📈 Forecast Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_demand = results_df['forecasted_demand_MU'].mean()
                    st.metric("Avg Daily Demand", f"{avg_demand:.1f} MU")
                
                with col2:
                    peak_demand = results_df['forecasted_demand_MU'].max()
                    st.metric("Peak Demand", f"{peak_demand:.1f} MU")
                
                with col3:
                    min_demand = results_df['forecasted_demand_MU'].min()
                    st.metric("Min Demand", f"{min_demand:.1f} MU")
                
                with col4:
                    high_risk_days = results_df['extreme_heat'].sum()
                    st.metric("High Risk Days", f"{high_risk_days}", 
                             delta=f"{high_risk_days/horizon_days*100:.0f}% of period")
                
                # Main forecast plot
                st.subheader("📊 Forecast Visualization")
                
                # Load metadata and get state-specific RMSE
                metadata = load_model_metadata()
                state_rmse = get_state_rmse(state, metadata)
                confidence_level = metadata.get('confidence_level', 0.9) if metadata else 0.9
                
                # Fix 2: Calculate confidence intervals in raw MU space (not log space)
                if state_rmse is not None:
                    # state_rmse is already in raw MU space from metadata
                    # Calculate z-score for confidence level (e.g., 1.645 for 90%, 1.96 for 95%)
                    z_score = stats.norm.ppf((1 + confidence_level) / 2)
                    # RMSE from metadata is already in MU space, apply directly
                    rmse_mu = state_rmse
                    # Optional widening factor for forecast uncertainty (can be 1.0 for no widening)
                    widening_factor = 1.0
                    margin = z_score * rmse_mu * widening_factor
                    
                    # Confidence intervals in raw MU space
                    upper_bound = results_df['forecasted_demand_MU'] + margin
                    lower_bound = results_df['forecasted_demand_MU'] - margin
                    
                    # Display RMSE info
                    st.caption(f"📊 Using state-specific RMSE: {state_rmse:.2f} MU (Confidence: {confidence_level*100:.0f}%)")
                else:
                    # If state RMSE not found, show warning but don't add intervals
                    st.warning(f"⚠️ State-specific RMSE not found for {state} in metadata. Confidence intervals not displayed.")
                    upper_bound = None
                    lower_bound = None
                
                fig = go.Figure()
                
                # Add confidence interval bands if RMSE is available
                if state_rmse is not None and upper_bound is not None:
                    # Upper bound
                    fig.add_trace(go.Scatter(
                        x=results_df['date'],
                        y=upper_bound,
                        mode='lines',
                        name=f'Upper Bound ({confidence_level*100:.0f}% CI)',
                        line=dict(width=0),
                        showlegend=True,
                        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Upper: %{y:.2f} MU<extra></extra>'
                    ))
                    
                    # Lower bound (filled area)
                    fig.add_trace(go.Scatter(
                        x=results_df['date'],
                        y=lower_bound,
                        mode='lines',
                        name=f'Lower Bound ({confidence_level*100:.0f}% CI)',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(31, 119, 180, 0.2)',
                        showlegend=True,
                        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Lower: %{y:.2f} MU<extra></extra>'
                    ))
                
                # Forecast line
                fig.add_trace(go.Scatter(
                    x=results_df['date'],
                    y=results_df['forecasted_demand_MU'],
                    mode='lines+markers',
                    name='Forecasted Demand',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=8),
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Demand: %{y:.2f} MU<extra></extra>'
                ))
                
                # Add peak marker trace
                peak_idx = results_df['forecasted_demand_MU'].idxmax()
                peak_date = results_df.loc[peak_idx, 'date']
                peak_value = results_df.loc[peak_idx, 'forecasted_demand_MU']
                fig.add_trace(
                    go.Scatter(
                        x=[peak_date],
                        y=[peak_value],
                        mode="markers",
                        marker=dict(size=12, color="cyan"),
                        name="Peak Demand",
                        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Peak Demand: %{y:.2f} MU<extra></extra>'
                    )
                )
                
                # Highlight extreme heat days
                heat_days = results_df[results_df['extreme_heat'] == 1]
                if len(heat_days) > 0:
                    fig.add_trace(go.Scatter(
                        x=heat_days['date'],
                        y=heat_days['forecasted_demand_MU'],
                        mode='markers',
                        name='High Risk (Heat / Uncertainty)',
                        marker=dict(
                            symbol='triangle-up',
                            size=15,
                            color='red',
                            line=dict(width=2, color='darkred')
                        ),
                        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>High Risk Day<br>Demand: %{y:.2f} MU<extra></extra>'
                    ))
                
                # Highlight holidays
                holiday_days = results_df[results_df['is_holiday'] == 1]
                if len(holiday_days) > 0:
                    fig.add_trace(go.Scatter(
                        x=holiday_days['date'],
                        y=holiday_days['forecasted_demand_MU'],
                        mode='markers',
                        name='Holiday',
                        marker=dict(
                            symbol='star',
                            size=12,
                            color='orange',
                            line=dict(width=1, color='darkorange')
                        ),
                        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Holiday<br>Demand: %{y:.2f} MU<extra></extra>'
                    ))
                
                # Update layout
                fig.update_layout(
                    title=f"Electricity Demand Forecast - {state}",
                    xaxis_title="Date",
                    yaxis_title="Demand (MU - Million Units)",
                    hovermode='x unified',
                    height=500,
                    template='plotly_white',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed data table
                with st.expander("📋 View Detailed Forecast Data"):
                    display_df = results_df.copy()
                    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                    display_df['temperature_mean'] = display_df['temperature_mean'] - 273.15  # Convert to Celsius for display
                    display_df['temperature_max'] = display_df['temperature_max'] - 273.15
                    
                    st.dataframe(
                        display_df.style.format({
                            'forecasted_demand_MU': '{:.2f}',
                            'temperature_mean': '{:.1f}°C',
                            'temperature_max': '{:.1f}°C',
                            'cdd': '{:.2f}'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                
            except Exception as e:
                st.error(f"❌ Error generating forecast: {str(e)}")
                st.exception(e)

# ============================================================================
# TAB 2: WEATHER IMPACT
# ============================================================================
with tab2:
    st.header("🌡️ Weather Impact Analysis")
    st.markdown("**Weather amplifies demand patterns rather than solely driving them.** Explore how temperature, CDD, and weather variables influence deviations from baseline demand patterns.")
    
    # Critical disclaimer about normalized values
    st.info("📊 **Important**: Demand values shown are normalized relative to each state's historical baseline to enable cross-state comparison. These are not absolute MU values.")
    
    # Check if model is loaded
    if model is None:
        st.error("❌ Model not loaded. Please check model path in settings.")
        st.stop()
    
    # Controls Section
    st.subheader("📊 Analysis Controls")
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        state_weather = st.selectbox(
            "Select State",
            ALL_STATES,
            index=ALL_STATES.index("Kerala") if "Kerala" in ALL_STATES else 0,
            help="Choose the state for weather impact analysis",
            key="weather_state"
        )
    
    with col2:
        analysis_mode = st.selectbox(
            "Analysis Mode",
            ["Forecast Analysis", "Historical Analysis"],
            index=0,
            help="Analyze forecast period or historical data",
            key="analysis_mode"
        )
    
    with col3:
        st.write("")  # Spacing
        analyze_btn = st.button("🔍 Analyze Weather Impact", type="primary", use_container_width=True)
    
    # Analysis execution
    if analyze_btn:
        with st.spinner("🔄 Analyzing weather impact... This may take a few seconds."):
            try:
                if analysis_mode == "Forecast Analysis":
                    # Use forecast data from Tab 1 if available, otherwise generate new forecast
                    if 'last_forecast' in st.session_state:
                        results_df = st.session_state['last_forecast']['results_df']
                        weather_forecast = st.session_state['last_forecast']['weather_forecast']
                        df_features = st.session_state['last_forecast']['df_features']
                        state_used = st.session_state['last_forecast']['state']
                        
                        if state_used != state_weather:
                            st.warning(f"⚠️ Last forecast was for {state_used}. Generating new forecast for {state_weather}...")
                            raise ValueError("State mismatch")
                    else:
                        # Generate new forecast
                        start_date = datetime.now().date()
                        horizon_days = 14  # Default to 14 days for analysis
                        
                        weather_forecast = get_weather_forecast(
                            state_weather,
                            start_date,
                            horizon_days,
                            use_api=use_weather_api
                        )
                        
                        if weather_forecast is None or len(weather_forecast) == 0:
                            st.error("❌ Failed to get weather data. Please try again.")
                            st.stop()
                        
                        # Prepare features and make predictions
                        X_forecast, df_features = prepare_features_for_prediction(
                            weather_forecast, state_weather, model=model
                        )
                        
                        # Convert from log space to absolute MU using denormalize_predictions
                        pred_log = model.predict(X_forecast)
                        historical_demand = load_historical_demand(state_weather)
                        pred_mu = denormalize_predictions(pred_log, weather_forecast['date'], state_weather, historical_demand)
                        
                        results_df = pd.DataFrame({
                            'date': weather_forecast['date'],
                            'forecasted_demand_MU': pred_mu,  # Absolute MU values after baseline multiplication
                            'temperature_mean': weather_forecast['2m_temperature_mean'],
                            'temperature_max': weather_forecast['2m_temperature_max'],
                            'cdd': df_features['cdd'],
                            'extreme_heat': df_features['extreme_heat'],
                            'is_holiday': df_features['is_holiday']
                        })
                
                else:  # Historical Analysis
                    # Load historical data
                    historical_df = load_historical_data(state_weather)
                    if historical_df is None:
                        st.error(f"❌ No historical data found for {state_weather}.")
                        st.stop()
                    
                    # Use last 90 days of historical data
                    historical_df = historical_df.sort_values('date').tail(90)
                    
                    # Prepare features and make predictions
                    X_historical, df_features = prepare_features_for_prediction(
                        historical_df, state_weather, model=model
                    )
                    
                    # Convert from log space to absolute MU using denormalize_predictions
                    pred_log = model.predict(X_historical)
                    historical_demand = load_historical_demand(state_weather)
                    pred_mu = denormalize_predictions(pred_log, historical_df['date'], state_weather, historical_demand)
                    
                    results_df = pd.DataFrame({
                        'date': historical_df['date'],
                        'forecasted_demand_MU': pred_mu,  # Absolute MU values after baseline multiplication
                        'temperature_mean': historical_df['2m_temperature_mean'],
                        'temperature_max': historical_df['2m_temperature_max'],
                        'cdd': df_features['cdd'],
                        'extreme_heat': df_features['extreme_heat'],
                        'is_holiday': df_features['is_holiday']
                    })
                    
                    weather_forecast = historical_df[['date', '2m_temperature_mean', '2m_temperature_max']].copy()
                
                # Store in session state for future use
                st.session_state['weather_analysis'] = {
                    'results_df': results_df,
                    'weather_forecast': weather_forecast,
                    'df_features': df_features,
                    'state': state_weather,
                    'mode': analysis_mode
                }
                
                # Display success message
                st.success(f"✅ Weather impact analysis completed for **{state_weather}** ({analysis_mode})")
                
                # Key Metrics
                st.subheader("📈 Key Weather Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_temp = (results_df['temperature_mean'] - 273.15).mean()
                    st.metric("Avg Temperature", f"{avg_temp:.1f}°C")
                
                with col2:
                    max_temp = (results_df['temperature_max'] - 273.15).max()
                    st.metric("Max Temperature", f"{max_temp:.1f}°C")
                
                with col3:
                    avg_cdd = results_df['cdd'].mean()
                    st.metric("Avg CDD", f"{avg_cdd:.2f}°C")
                
                with col4:
                    extreme_heat_days = results_df['extreme_heat'].sum()
                    st.metric("Extreme Heat Days", f"{extreme_heat_days}", 
                             delta=f"{extreme_heat_days/len(results_df)*100:.0f}% of period")
                
                # Natural Language Insight
                st.subheader("💡 Weather Impact Insight")
                insight_text = generate_weather_insight(results_df, df_features, state_weather)
                st.info(insight_text)
                
                # Visualizations
                st.subheader("📊 Weather Impact Visualizations")
                
                # Temperature Overlay
                st.markdown("### Temperature vs Demand Overlay")
                st.caption("💡 The baseline (dashed gray line) shows demand without weather signal. The difference demonstrates weather impact.")
                temp_overlay_fig = plot_demand_temperature_overlay(results_df, weather_forecast, df_features)
                if temp_overlay_fig:
                    st.plotly_chart(temp_overlay_fig, use_container_width=True)
                
                # CDD Timeline
                st.markdown("### Cooling Degree Days (CDD) Timeline")
                cdd_fig = plot_cdd_timeline(results_df, df_features)
                if cdd_fig:
                    st.plotly_chart(cdd_fig, use_container_width=True)
                
                # Feature Importance with toggle
                st.markdown("### Model Dependency (Relative Influence)")
                
                # Add toggle for All Features vs Weather-Only
                col1, col2 = st.columns([1, 3])
                with col1:
                    show_weather_only = st.checkbox(
                        "Weather-Only Features",
                        value=False,
                        help="Show only weather-related features (Temperature, Humidity, Weather categories)"
                    )
                
                feature_importance = get_feature_importance(model, top_n=15, filter_weather_only=show_weather_only)
                if feature_importance:
                    importance_fig = plot_feature_importance(feature_importance)
                    if importance_fig:
                        st.plotly_chart(importance_fig, use_container_width=True)
                    
                    # Add important note about autoregressive nature
                    st.caption(
                        "💡 **Note**: Short-term demand history explains most variance; weather variables primarily influence deviations during extreme conditions. "
                        "This model is primarily autoregressive (momentum-based) with weather as a secondary signal."
                    )
                    
                    # Show top 5 features in expander
                    with st.expander("📋 Top 5 Most Influential Features"):
                        top_5 = dict(list(feature_importance.items())[:5])
                        for i, (feat, imp) in enumerate(top_5.items(), 1):
                            category = categorize_feature(feat)
                            st.markdown(f"{i}. **{feat}** ({category}) - Relative Influence: {imp:.4f}")
                else:
                    st.warning("⚠️ Could not extract feature importance from model.")
                
                # Seasonal Comparison (if historical data available)
                if analysis_mode == "Historical Analysis" or 'last_forecast' in st.session_state:
                    st.markdown("### Seasonal Comparison")
                    
                    # Get current season from data
                    if len(df_features) > 0:
                        current_season = df_features['season'].iloc[0] if 'season' in df_features.columns else None
                        
                        if current_season:
                            seasonal_stats = get_seasonal_comparison(state_weather, current_season)
                            
                            if seasonal_stats:
                                # Create seasonal comparison chart
                                seasons = list(seasonal_stats.keys())
                                temp_means = [seasonal_stats[s]['temp_mean'] for s in seasons]
                                cdd_means = [seasonal_stats[s]['cdd_mean'] for s in seasons]
                                
                                fig_seasonal = make_subplots(specs=[[{"secondary_y": True}]])
                                
                                fig_seasonal.add_trace(
                                    go.Bar(
                                        x=seasons,
                                        y=temp_means,
                                        name='Avg Temperature',
                                        marker_color='#ff7f0e',
                                        hovertemplate='<b>%{x}</b><br>Temp: %{y:.1f}°C<extra></extra>'
                                    ),
                                    secondary_y=False,
                                )
                                
                                fig_seasonal.add_trace(
                                    go.Bar(
                                        x=seasons,
                                        y=cdd_means,
                                        name='Avg CDD',
                                        marker_color='#d62728',
                                        hovertemplate='<b>%{x}</b><br>CDD: %{y:.2f}°C<extra></extra>'
                                    ),
                                    secondary_y=True,
                                )
                                
                                fig_seasonal.update_xaxes(title_text="Season")
                                fig_seasonal.update_yaxes(title_text="Temperature (°C)", secondary_y=False)
                                fig_seasonal.update_yaxes(title_text="CDD (°C)", secondary_y=True)
                                
                                fig_seasonal.update_layout(
                                    title="Seasonal Temperature and CDD Comparison",
                                    hovermode='x unified',
                                    height=400,
                                    template='plotly_white',
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1
                                    )
                                )
                                
                                st.plotly_chart(fig_seasonal, use_container_width=True)
                                
                                # Show seasonal statistics table
                                with st.expander("📋 Detailed Seasonal Statistics"):
                                    seasonal_df = pd.DataFrame(seasonal_stats).T
                                    seasonal_df.index.name = 'Season'
                                    st.dataframe(seasonal_df.style.format({
                                        'temp_mean': '{:.1f}°C',
                                        'temp_max': '{:.1f}°C',
                                        'cdd_mean': '{:.2f}°C',
                                        'extreme_heat_pct': '{:.1f}%',
                                        'count': '{:.0f}'
                                    }), use_container_width=True)
                            else:
                                st.info("ℹ️ Seasonal comparison data not available for this state.")
                        else:
                            st.info("ℹ️ Could not determine current season for comparison.")
                    else:
                        st.info("ℹ️ Insufficient data for seasonal comparison.")
                
            except ValueError as e:
                if "State mismatch" in str(e):
                    # Regenerate forecast for correct state
                    start_date = datetime.now().date()
                    horizon_days = 14
                    
                    weather_forecast = get_weather_forecast(
                        state_weather,
                        start_date,
                        horizon_days,
                        use_api=use_weather_api
                    )
                    
                    X_forecast, df_features = prepare_features_for_prediction(
                        weather_forecast, state_weather, model=model
                    )
                    
                    # Convert from log space to absolute MU using denormalize_predictions
                    pred_log = model.predict(X_forecast)
                    historical_demand = load_historical_demand(state_weather)
                    pred_mu = denormalize_predictions(pred_log, weather_forecast['date'], state_weather, historical_demand)
                    
                    results_df = pd.DataFrame({
                        'date': weather_forecast['date'],
                        'forecasted_demand_MU': pred_mu,  # Absolute MU values after baseline multiplication
                        'temperature_mean': weather_forecast['2m_temperature_mean'],
                        'temperature_max': weather_forecast['2m_temperature_max'],
                        'cdd': df_features['cdd'],
                        'extreme_heat': df_features['extreme_heat'],
                        'is_holiday': df_features['is_holiday']
                    })
                    
                    # Continue with analysis (code would continue here, but for brevity, we'll show error)
                    st.rerun()
                else:
                    st.error(f"❌ Error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error analyzing weather impact: {str(e)}")
                st.exception(e)
    
    # Show instructions if no analysis has been run
    if 'weather_analysis' not in st.session_state:
        st.info("👆 Select a state and analysis mode, then click 'Analyze Weather Impact' to see how weather drives electricity demand.")
        
        st.markdown("""
        **What you'll see:**
        - **Temperature Overlay**: Dual-axis plot showing demand and temperature correlation
        - **CDD Timeline**: Cooling Degree Days with extreme heat day highlighting
        - **Feature Importance**: Top 15 features that drive demand predictions
        - **Seasonal Comparison**: Historical seasonal patterns (when available)
        - **Natural Language Insights**: AI-generated explanations of weather impact
        """)

# ============================================================================
# TAB 3: AI ASSISTANT
# ============================================================================
with tab3:
    st.header("🤖 AI Energy Assistant")
    st.markdown("Ask natural language questions about forecasts, weather impacts, model performance, and risk factors.")
    
    # Fail-safe RAG initialization
    rag_enabled = True
    vector_store = None
    rag_engine = None
    
    try:
        vector_store, rag_engine = initialize_rag_system()
        if vector_store is None or rag_engine is None:
            rag_enabled = False
    except FileNotFoundError as e:
        rag_enabled = False
    except ImportError as e:
        rag_enabled = False
    except Exception as e:
        rag_enabled = False
    
    # Show status and handle limited mode
    if not rag_enabled or vector_store is None or rag_engine is None:
        st.warning("⚠️ AI Assistant is running in limited mode (knowledge base unavailable).")
        st.info("""
        **What this means:**
        - The knowledge base could not be loaded (missing files, path issues, or dependencies)
        - You can still ask questions, but responses will be limited
        - To enable full functionality:
          1. Ensure sentence-transformers is installed: `pip install sentence-transformers`
          2. Check that `model_metadata.json` exists in the project root
          3. Restart the application
        """)
        # Continue in limited mode - don't stop the app
        rag_enabled = False
    
    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Get context from last forecast (if available)
    forecast_context = None
    forecast_state = None
    forecast_horizon = None
    
    if 'last_forecast' in st.session_state:
        forecast_data = st.session_state['last_forecast']
        forecast_state = forecast_data.get('state')
        forecast_horizon = forecast_data.get('horizon_days')
        
        # Get forecast summary (use stored one if available, otherwise generate)
        forecast_summary = forecast_data.get('forecast_summary')
        results_df = forecast_data.get('results_df')
        
        if forecast_summary is None and results_df is not None and len(results_df) > 0:
            # Generate if not already stored (shouldn't happen if forecast was just created)
            metadata = load_model_metadata()
            forecast_summary = generate_forecast_summary(
                results_df,
                forecast_state,
                forecast_horizon,
                forecast_data.get('df_features'),
                metadata=metadata
            )
        
        # Only try to add to vector store if RAG is enabled
        if rag_enabled and forecast_summary and vector_store is not None:
            # Check if this forecast is already in the vector store
            forecast_id = f"forecast_{forecast_state}_{forecast_summary['start_date']}"
            if 'last_forecast_id' not in st.session_state or st.session_state['last_forecast_id'] != forecast_id:
                # Add forecast summary to vector store
                forecast_docs = build_forecast_summary_documents(forecast_summary)
                if forecast_docs:
                    try:
                        texts = [doc['text'] for doc in forecast_docs]
                        metadata_list = [doc['metadata'] for doc in forecast_docs]
                        embeddings = get_embeddings(texts)
                        vector_store.add_documents(embeddings, texts, metadata_list)
                        vector_store.save_index()  # Save after adding
                        st.session_state['last_forecast_id'] = forecast_id
                    except Exception as e:
                        st.warning(f"Could not add forecast to knowledge base: {str(e)}")
    
    # Display example questions
    with st.expander("💡 Example Questions", expanded=False):
        st.markdown("""
        **Forecast Questions:**
        - "What is the forecast for [state]?"
        - "When is peak demand expected?"
        - "Are there any high-risk days in the forecast?"
        
        **Weather Impact:**
        - "How does temperature affect demand?"
        - "What is the impact of extreme heat on electricity demand?"
        - "Which weather variables are most important?"
        
        **Model Performance:**
        - "What is the model's RMSE for [state]?"
        - "How accurate is the model?"
        - "What are the top features for forecasting?"
        
        **Risk & Diagnostics:**
        - "What are the confidence intervals?"
        - "Why might the forecast be uncertain?"
        """)
    
    # Context display
    if forecast_state:
        st.info(f"📊 Current context: {forecast_state} ({forecast_horizon} days forecast available)")
    
    # Chat interface
    st.markdown("### 💬 Chat")
    
    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources", expanded=False):
                    for j, source in enumerate(message["sources"][:3], 1):
                        st.caption(f"Source {j}: {source['text'][:200]}...")
                        if 'metadata' in source:
                            meta = source['metadata']
                            if 'state' in meta:
                                st.caption(f"  State: {meta['state']}")
                            if 'type' in meta:
                                st.caption(f"  Type: {meta['type']}")
            
            # Show confidence if available (with clarified wording)
            if "confidence" in message:
                confidence_pct = message["confidence"] * 100
                confidence_text = f"Response confidence: {confidence_pct:.0f}% (based on data availability and forecast horizon)"
                if confidence_pct >= 80:
                    st.success(confidence_text)
                elif confidence_pct >= 60:
                    st.warning(confidence_text)
                else:
                    st.info(confidence_text)
    
    # User input
    user_query = st.chat_input("Ask a question about forecasts, weather, or model performance...")
    
    if user_query:
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if not rag_enabled or rag_engine is None:
                    # Limited mode - provide basic response
                    st.info("""
                    **Limited Mode Active**
                    
                    The AI Assistant is running in limited mode because the knowledge base is unavailable.
                    
                    To answer your question properly, please:
                    1. Ensure sentence-transformers is installed: `pip install sentence-transformers`
                    2. Check that model_metadata.json exists
                    3. Restart the application
                    
                    **Your question:** "{}"
                    """.format(user_query))
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "AI Assistant is in limited mode. Please check the setup instructions above."
                    })
                else:
                    try:
                        response, contexts, avg_similarity = rag_engine.query(
                            user_query,
                            current_state=forecast_state,
                            forecast_horizon=forecast_horizon,
                            top_k=5,
                            min_similarity=0.5  # Lower threshold to get more results
                        )
                        
                        st.markdown(response)
                        
                        # Add assistant response to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response,
                            "sources": contexts,
                            "confidence": avg_similarity
                        })
                        
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": error_msg
                        })
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption("⚡ Smart Energy Forecasting Platform | Stage 1.5: Forecast + Explain")
