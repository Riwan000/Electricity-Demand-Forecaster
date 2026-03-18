# ============================================================================
# WEATHER API FUNCTIONS
# ============================================================================
import requests
import pandas as pd
import numpy as np
import streamlit as st

from config import STATE_COORDINATES
from data_loading import load_historical_data


def get_weather_forecast_openmeteo(state_name, start_date, num_days):
    """
    Get weather forecast using Open-Meteo API (FREE, no API key required).

    Open-Meteo is a free weather API that provides up to 16 days of forecast
    data without requiring registration or API keys.

    Parameters:
        state_name (str): Name of the Indian state
        start_date (datetime.date): Start date for forecast
        num_days (int): Number of days to forecast (max 16 for free tier)

    Returns:
        pd.DataFrame: Weather data matching model's expected format
    """
    if state_name not in STATE_COORDINATES:
        raise ValueError(f"Coordinates not found for state: {state_name}")

    coords = STATE_COORDINATES[state_name]
    lat, lon = coords["lat"], coords["lon"]

    num_days = min(num_days, 16)

    try:
        url = "https://api.open-meteo.com/v1/forecast"
        daily_params = "temperature_2m_max,temperature_2m_min,dewpoint_2m_mean,windspeed_10m_max,winddirection_10m_dominant,shortwave_radiation_sum,cloudcover_mean"

        from urllib.parse import urlencode
        base_params = {
            "latitude": lat,
            "longitude": lon,
            "forecast_days": num_days,
            "timezone": "Asia/Kolkata"
        }
        query_parts = [urlencode(base_params), f"daily={daily_params}"]
        full_url = f"{url}?{'&'.join(query_parts)}"

        response = requests.get(full_url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "daily" not in data:
            raise ValueError("Invalid API response: missing 'daily' key")

        daily_data = data["daily"]

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
        forecast_df["2m_temperature_max"] = forecast_df["2m_temperature_max"] + 273.15
        forecast_df["2m_temperature_min"] = forecast_df["2m_temperature_min"] + 273.15
        forecast_df["2m_temperature_mean"] = forecast_df["2m_temperature_mean"] + 273.15

        if "dewpoint_2m_mean" in daily_data:
            forecast_df["2m_dewpoint_temperature_mean"] = [
                d + 273.15 if d is not None else None
                for d in daily_data["dewpoint_2m_mean"]
            ]
            forecast_df["2m_dewpoint_temperature_min"] = forecast_df["2m_dewpoint_temperature_mean"]
            forecast_df["2m_dewpoint_temperature_max"] = forecast_df["2m_dewpoint_temperature_mean"]
        else:
            forecast_df["2m_dewpoint_temperature_mean"] = forecast_df["2m_temperature_mean"] - 5
            forecast_df["2m_dewpoint_temperature_min"] = forecast_df["2m_dewpoint_temperature_mean"]
            forecast_df["2m_dewpoint_temperature_max"] = forecast_df["2m_dewpoint_temperature_mean"]

        if "windspeed_10m_max" in daily_data and "winddirection_10m_dominant" in daily_data:
            wind_speed = daily_data["windspeed_10m_max"]
            wind_dir_rad = np.radians(daily_data["winddirection_10m_dominant"])

            forecast_df["10m_u_component_of_wind_mean"] = [
                -speed * np.sin(rad) for speed, rad in zip(wind_speed, wind_dir_rad)
            ]
            forecast_df["10m_v_component_of_wind_mean"] = [
                -speed * np.cos(rad) for speed, rad in zip(wind_speed, wind_dir_rad)
            ]
            forecast_df["10m_u_component_of_wind_min"] = forecast_df["10m_u_component_of_wind_mean"]
            forecast_df["10m_u_component_of_wind_max"] = forecast_df["10m_u_component_of_wind_mean"]
            forecast_df["10m_v_component_of_wind_min"] = forecast_df["10m_v_component_of_wind_mean"]
            forecast_df["10m_v_component_of_wind_max"] = forecast_df["10m_v_component_of_wind_mean"]
        else:
            for col in ["10m_u_component_of_wind_min", "10m_u_component_of_wind_mean",
                        "10m_u_component_of_wind_max", "10m_v_component_of_wind_min",
                        "10m_v_component_of_wind_mean", "10m_v_component_of_wind_max"]:
                forecast_df[col] = 0.0

        if "shortwave_radiation_sum" in daily_data:
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

    This is a fallback method when weather API is unavailable.

    Parameters:
        state_name (str): Name of the Indian state
        start_date (datetime.date): Start date for forecast
        num_days (int): Number of days to forecast

    Returns:
        pd.DataFrame: Weather data with historical averages
    """
    historical_df = load_historical_data(state_name)
    if historical_df is None:
        raise ValueError(f"No historical data found for {state_name}")

    forecast_dates = pd.date_range(start=start_date, periods=num_days, freq='D')
    forecast_data = []

    for date in forecast_dates:
        month_day = (date.month, date.day)
        historical_same_date = historical_df[
            (historical_df['date'].dt.month == month_day[0]) &
            (historical_df['date'].dt.day == month_day[1])
        ]

        if len(historical_same_date) > 0:
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

    Tries API first; falls back to climatology if API fails.

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
            return get_weather_forecast_openmeteo(state_name, start_date, num_days)
        except Exception as e:
            st.warning(f"⚠️ Weather API failed: {str(e)}. Using historical averages...")
            return get_weather_forecast_climatology(state_name, start_date, num_days)
    else:
        return get_weather_forecast_climatology(state_name, start_date, num_days)
