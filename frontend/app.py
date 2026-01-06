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
        '../xgb1_model.pkl',  # From frontend directory
        'xgb1_model.pkl',      # From root directory
        '../models/xgb1_model.pkl'  # If models folder exists
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
        # Open-Meteo API expects daily as a comma-separated string, not a list
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,dewpoint_2m,windspeed_10m_max,winddirection_10m_dominant,shortwave_radiation_sum,cloudcover_mean",
            "forecast_days": num_days,
            "timezone": "Asia/Kolkata"      # Indian Standard Time
        }
        
        # Make API request with timeout
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # Raise exception for bad status codes
        data = response.json()
        
        if "daily" not in data:
            raise ValueError("Invalid API response: missing 'daily' key")
        
        daily_data = data["daily"]
        
        # Convert API response to DataFrame matching model's expected format
        forecast_df = pd.DataFrame({
            "date": pd.to_datetime(daily_data["time"]),
            "2m_temperature_max": daily_data["temperature_2m_max"],
            "2m_temperature_min": daily_data["temperature_2m_min"],
            "2m_temperature_mean": daily_data.get("temperature_2m_mean", 
                # Calculate mean if not provided
                [(max + min) / 2 for max, min in zip(
                    daily_data["temperature_2m_max"], 
                    daily_data["temperature_2m_min"]
                )]
            ),
        })
        
        # Convert temperatures from Celsius to Kelvin
        # (Check your model - if it uses Celsius, remove this conversion)
        forecast_df["2m_temperature_max"] = forecast_df["2m_temperature_max"] + 273.15
        forecast_df["2m_temperature_min"] = forecast_df["2m_temperature_min"] + 273.15
        forecast_df["2m_temperature_mean"] = forecast_df["2m_temperature_mean"] + 273.15
        
        # Add dewpoint (convert to Kelvin if needed)
        if "dewpoint_2m" in daily_data:
            forecast_df["2m_dewpoint_temperature_mean"] = [
                d + 273.15 if d is not None else None 
                for d in daily_data["dewpoint_2m"]
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

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Forecast", 
    "🌡️ Weather Impact", 
    "⚠️ Risk & Diagnostics", 
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
                
                # Make predictions using the model
                predictions = model.predict(X_forecast)
                
                # Create results dataframe
                results_df = pd.DataFrame({
                    'date': weather_forecast['date'],
                    'forecasted_demand_MU': predictions,
                    'temperature_mean': weather_forecast['2m_temperature_mean'],
                    'temperature_max': weather_forecast['2m_temperature_max'],
                    'cdd': df_features['cdd'],
                    'extreme_heat': df_features['extreme_heat'],
                    'is_holiday': df_features['is_holiday']
                })
                
                # Display success message
                st.success(f"✅ Forecast generated successfully for **{state}** ({horizon_days} days)")
                
                # Key metrics
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
                
                fig = go.Figure()
                
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
                
                # Highlight extreme heat days
                heat_days = results_df[results_df['extreme_heat'] == 1]
                if len(heat_days) > 0:
                    fig.add_trace(go.Scatter(
                        x=heat_days['date'],
                        y=heat_days['forecasted_demand_MU'],
                        mode='markers',
                        name='Extreme Heat Risk',
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
# TAB 2: WEATHER IMPACT (Placeholder)
# ============================================================================
with tab2:
    st.header("Weather Impact & Risk Indicators")
    st.info("🚧 This tab will show temperature overlays, CDD analysis, and extreme heat indicators.")
    st.markdown("""
    **Planned Features:**
    - Temperature & CDD overlays on demand forecast
    - Extreme heat indicators with risk scores
    - Seasonal comparison charts
    - Weather-driven demand patterns
    """)

# ============================================================================
# TAB 3: RISK & DIAGNOSTICS (Placeholder)
# ============================================================================
with tab3:
    st.header("Risk & Diagnostics")
    st.info("🚧 This tab will show residual plots, anomaly detection, and error analysis.")
    st.markdown("""
    **Planned Features:**
    - Residual plots over time
    - Highlighted anomaly days
    - Error vs weather visualization
    - Model performance metrics
    """)

# ============================================================================
# TAB 4: AI ASSISTANT (Placeholder)
# ============================================================================
with tab4:
    st.header("AI Energy Assistant")
    st.info("🚧 This tab will provide AI-driven explanations using LLM + RAG.")
    st.markdown("""
    **Planned Features:**
    - Natural language explanations of forecasts
    - Answer questions like:
        - "Why is demand expected to increase next week?"
        - "Which weather variables matter most for this state?"
        - "Why did the model fail on these dates?"
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption("⚡ Smart Energy Forecasting Platform | Stage 1.5: Forecast + Explain")
