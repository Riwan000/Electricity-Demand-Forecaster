# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================
import pandas as pd
import numpy as np
import holidays
from data_loading import get_state_historical_averages



def engineer_features(df, state_name, metadata=None):
    """
    Engineer features exactly as done in training.

    This function must match the notebook's feature engineering pipeline exactly.
    It creates calendar features, holiday flags, cooling degree days, state/region
    encodings, and other derived features that the model expects.

    Parameters:
        df (pd.DataFrame): Weather data with date column
        state_name (str): Name of the state (for state encoding)
        metadata (dict, optional): model_metadata.json contents. Used to read
            train-derived thresholds (cdd_threshold_kelvin,
            extreme_heat_threshold_kelvin) so inference matches training exactly.

    Returns:
        pd.DataFrame: Data with engineered features added
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Calendar features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_year'] = df['date'].dt.dayofyear

    # Weekend flags
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Season encoding
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

    normalized_state = state_name
    if state_name == 'Jammu and Kashmir':
        normalized_state = 'J&K'

    df['states'] = normalized_state
    df['region'] = state_to_region.get(state_name, 'nr')

    # Cooling Degree Days (CDD)
    threshold_temp = (metadata.get('cdd_threshold_kelvin', 297.15)
                      if metadata else 297.15)
    df['cdd'] = np.maximum(0, df['2m_temperature_mean'] - threshold_temp)
    df['CDD'] = df['cdd']

    # Extreme heat flag — use train-derived threshold from metadata
    if metadata and 'extreme_heat_threshold_kelvin' in metadata:
        heat_threshold = metadata['extreme_heat_threshold_kelvin']
    else:
        heat_threshold = 305.0  # fixed historical threshold (train-derived)
    df['extreme_heat'] = (df['2m_temperature_mean'] > heat_threshold).astype(int)
    df['extreme_heat_flag'] = df['extreme_heat'].astype(int)

    # Temperature interaction features
    df['temp_min_heat_interaction'] = df['2m_temperature_min'] * df['extreme_heat_flag']
    df['temp_mean_heat_interaction'] = df['2m_temperature_mean'] * df['extreme_heat_flag']
    df['temp_max_heat_interaction'] = df['2m_temperature_max'] * df['extreme_heat_flag']

    # Historical features: use per-state averages where available; generation data
    # is unavailable at inference time and stays at 0.0.
    hist = get_state_historical_averages(state_name)
    df['energymet_7d_avg'] = hist.get('energymet_7d_avg', 0.0)
    df['monthly_energy_avg'] = hist.get('monthly_energy_avg', 0.0)
    df['monthly_rolling_mean'] = hist.get('monthly_rolling_mean', 0.0)
    df['demand_rolling'] = hist.get('demand_rolling', 0.0)
    df['state_share'] = hist.get('state_share', 0.0)
    df['max.demand met during the day(mw)'] = hist.get('max_demand_mw', 0.0)
    # Generation features have no inference-time source — kept at 0.0
    df['generation_mu'] = 0.0
    df['generation_7d_avg'] = 0.0
    df['monthly_generation_avg'] = 0.0
    df['gen_rolling'] = 0.0

    # Temperature range
    df['temp_range'] = df['2m_temperature_max'] - df['2m_temperature_min']
    df['temp_range_heat'] = df['temp_range'] * df['extreme_heat_flag']

    # Lag and rolling features (use current values as approximation for forecasting)
    for lag in [1, 3, 7]:
        df[f'temp_mean_lag{lag}'] = df['2m_temperature_mean']

    df['temp_mean_roll3'] = df['2m_temperature_mean']
    df['temp_mean_roll7'] = df['2m_temperature_mean']
    df['temp_max_roll3'] = df['2m_temperature_max']
    df['temp_max_roll7'] = df['2m_temperature_max']
    df['temp_min_roll3'] = df['2m_temperature_min']
    df['temp_min_roll7'] = df['2m_temperature_min']

    return df


def prepare_features_for_prediction(weather_df, state_name, model=None, metadata=None):
    """
    Prepare the exact feature set the model expects.

    Ensures that the features match exactly what the model was trained on.
    Feature names and order must match the training data.

    Parameters:
        weather_df (pd.DataFrame): Raw weather data
        state_name (str): Name of the state
        model: Optional loaded XGBoost model (used to extract exact feature order)
        metadata (dict): Optional model metadata

    Returns:
        tuple: (X_features, df_features)
            - X_features: DataFrame with only model features in correct order
            - df_features: Full dataframe with all engineered features
    """
    df_features = engineer_features(weather_df, state_name, metadata=metadata)

    categorical_cols = ['states', 'region', 'season', 'holiday_name']

    season_values = df_features['season'].copy() if 'season' in df_features.columns else None
    holiday_values = df_features['holiday_name'].copy() if 'holiday_name' in df_features.columns else None

    for col in categorical_cols:
        if col in df_features.columns:
            dummies = pd.get_dummies(df_features[col], prefix=col)
            df_features = pd.concat([df_features, dummies], axis=1)

    df_features = df_features.drop(columns=categorical_cols, errors='ignore')

    # Restore season as a plain column so callers (e.g. weather_tab) can read it
    if season_values is not None:
        df_features['season'] = season_values.values

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

    # Use model's exact feature order if available
    if model is not None and hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
    else:
        all_states_ordered = [
            'Arunachal Pradesh', 'Assam', 'Bihar', 'Chandigarh', 'Chhattisgarh',
            'Goa', 'Gujarat', 'Haryana', 'J&K', 'Jharkhand', 'Karnataka', 'Kerala',
            'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha',
            'Puducherry', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana',
            'Tripura', 'Uttarakhand', 'West Bengal'
        ]
        regions_ordered = ['ner', 'nr', 'sr', 'wr']
        seasons_ordered = ['Monsoon', 'Summer', 'Winter']
        holidays_ordered = ['Dussehra', 'Guru Nanak Jayanti', 'No Holiday']

        expected_features = expected_features_base.copy()
        expected_features.extend([f'states_{s}' for s in all_states_ordered])
        expected_features.extend([f'region_{r}' for r in regions_ordered])
        expected_features.extend([f'season_{s}' for s in seasons_ordered])
        expected_features.extend([f'holiday_name_{h}' for h in holidays_ordered])

    # Add state one-hot columns in exact order
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
        if state_name == state or (state_name == 'Jammu and Kashmir' and state == 'J&K'):
            df_features[col_name] = 1

    # Add region one-hot columns (note: model doesn't have 'er', maps to 'nr')
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
    if current_region == 'er':
        current_region = 'nr'
    regions_ordered = ['ner', 'nr', 'sr', 'wr']
    for region in regions_ordered:
        col_name = f'region_{region}'
        if col_name not in df_features.columns:
            df_features[col_name] = 0
        if region == current_region:
            df_features[col_name] = 1

    # Add season one-hot columns
    seasons_ordered = ['Monsoon', 'Summer', 'Winter']
    for season in seasons_ordered:
        col_name = f'season_{season}'
        if col_name not in df_features.columns:
            df_features[col_name] = 0
        if season_values is not None:
            df_features[col_name] = (season_values == season).astype(int)

    # Add holiday name one-hot columns
    holidays_ordered = ['Dussehra', 'Guru Nanak Jayanti', 'No Holiday']
    for holiday in holidays_ordered:
        col_name = f'holiday_name_{holiday}'
        if col_name not in df_features.columns:
            df_features[col_name] = 0
        if holiday_values is not None:
            df_features[col_name] = holiday_values.apply(
                lambda x: 1 if holiday in str(x) else 0
            )

    # Build feature matrix in exact expected order, filling missing with 0
    X = pd.DataFrame(index=df_features.index)
    for feat in expected_features:
        if feat in df_features.columns:
            X[feat] = df_features[feat]
        else:
            X[feat] = 0

    return X, df_features
