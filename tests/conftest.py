# ============================================================================
# PYTEST FIXTURES & CONFIGURATION
# ============================================================================
import pytest
import pandas as pd
import numpy as np
from models import load_model, load_model_metadata


@pytest.fixture
def sample_weather_df():
    """Weather data for 10 days (realistic range: 290-320 K)."""
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    temps = np.linspace(290, 310, 10)  # 17°C to 37°C

    return pd.DataFrame({
        'date': dates,
        '2m_temperature_max': temps + 5,
        '2m_temperature_min': temps - 5,
        '2m_temperature_mean': temps,
        '2m_dewpoint_temperature_min': temps - 10,
        '2m_dewpoint_temperature_mean': temps - 5,
        '2m_dewpoint_temperature_max': temps,
        '10m_u_component_of_wind_min': np.zeros(10),
        '10m_u_component_of_wind_mean': np.ones(10),
        '10m_u_component_of_wind_max': 2 * np.ones(10),
        '10m_v_component_of_wind_min': np.zeros(10),
        '10m_v_component_of_wind_mean': np.ones(10),
        '10m_v_component_of_wind_max': 2 * np.ones(10),
        'surface_solar_radiation_downwards_min': np.zeros(10),
        'surface_solar_radiation_downwards_mean': 100 * np.ones(10),
        'surface_solar_radiation_downwards_max': 200 * np.ones(10),
        'total_cloud_cover_min': np.zeros(10),
        'total_cloud_cover_mean': 50 * np.ones(10),
        'total_cloud_cover_max': 100 * np.ones(10),
        'utci_min': temps - 5,
        'utci_mean': temps,
        'utci_max': temps + 5,
    })


@pytest.fixture
def single_day_weather_df():
    """Single-day weather data (for lag edge cases)."""
    return pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01']),
        '2m_temperature_max': [300.0],
        '2m_temperature_min': [290.0],
        '2m_temperature_mean': [295.0],
        '2m_dewpoint_temperature_min': [285.0],
        '2m_dewpoint_temperature_mean': [290.0],
        '2m_dewpoint_temperature_max': [295.0],
        '10m_u_component_of_wind_min': [0.0],
        '10m_u_component_of_wind_mean': [1.0],
        '10m_u_component_of_wind_max': [2.0],
        '10m_v_component_of_wind_min': [0.0],
        '10m_v_component_of_wind_mean': [1.0],
        '10m_v_component_of_wind_max': [2.0],
        'surface_solar_radiation_downwards_min': [0.0],
        'surface_solar_radiation_downwards_mean': [100.0],
        'surface_solar_radiation_downwards_max': [200.0],
        'total_cloud_cover_min': [0.0],
        'total_cloud_cover_mean': [50.0],
        'total_cloud_cover_max': [100.0],
        'utci_min': [290.0],
        'utci_mean': [295.0],
        'utci_max': [300.0],
    })


@pytest.fixture
def metadata():
    """Model metadata with CDD and extreme heat thresholds."""
    return {
        'cdd_threshold_kelvin': 297.15,
        'extreme_heat_threshold_kelvin': 305.0,
        'state_rmse': {'Maharashtra': 4.2, 'Tamil Nadu': 3.8},
    }


@pytest.fixture
def all_states():
    """All 36 Indian states/UTs for mapping tests."""
    return [
        'Arunachal Pradesh', 'Assam', 'Bihar', 'Chandigarh', 'Chhattisgarh',
        'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu and Kashmir',
        'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra',
        'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha',
        'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana',
        'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal',
        'NCT of Delhi', 'Puducherry', 'Daman and Diu', 'Dadra and Nagar Haveli',
        'Lakshadweep', 'Andaman and Nicobar'
    ]


@pytest.fixture
def loaded_model():
    """Load the actual trained XGBoost model."""
    try:
        return load_model()
    except:
        pytest.skip("Model file not found - skipping model-dependent tests")


@pytest.fixture
def loaded_metadata():
    """Load the actual model metadata."""
    try:
        return load_model_metadata()
    except:
        return {
            'cdd_threshold_kelvin': 297.15,
            'extreme_heat_threshold_kelvin': 305.0,
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_sample_data_with_temps(state, temps):
    """
    Create weather data with specific temperatures.

    Args:
        state: State name
        temps: List of temperatures (mean values in Kelvin)

    Returns:
        pd.DataFrame with weather data
    """
    dates = pd.date_range('2023-01-01', periods=len(temps))
    return pd.DataFrame({
        'date': dates,
        '2m_temperature_max': [t + 5 for t in temps],
        '2m_temperature_min': [t - 5 for t in temps],
        '2m_temperature_mean': temps,
        '2m_dewpoint_temperature_min': [t - 10 for t in temps],
        '2m_dewpoint_temperature_mean': [t - 5 for t in temps],
        '2m_dewpoint_temperature_max': temps,
        '10m_u_component_of_wind_min': np.zeros(len(temps)),
        '10m_u_component_of_wind_mean': np.ones(len(temps)),
        '10m_u_component_of_wind_max': 2 * np.ones(len(temps)),
        '10m_v_component_of_wind_min': np.zeros(len(temps)),
        '10m_v_component_of_wind_mean': np.ones(len(temps)),
        '10m_v_component_of_wind_max': 2 * np.ones(len(temps)),
        'surface_solar_radiation_downwards_min': np.zeros(len(temps)),
        'surface_solar_radiation_downwards_mean': 100 * np.ones(len(temps)),
        'surface_solar_radiation_downwards_max': 200 * np.ones(len(temps)),
        'total_cloud_cover_min': np.zeros(len(temps)),
        'total_cloud_cover_mean': 50 * np.ones(len(temps)),
        'total_cloud_cover_max': 100 * np.ones(len(temps)),
        'utci_min': [t - 5 for t in temps],
        'utci_mean': temps,
        'utci_max': [t + 5 for t in temps],
    })
