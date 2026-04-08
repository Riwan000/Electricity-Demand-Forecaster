# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================
import os
import pandas as pd
import numpy as np
import streamlit as st


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

    column_name = state_column_map.get(state_name)
    if column_name is None:
        return None

    possible_paths = [
        'data/daily_energy_met_MU.csv',
        '../data/daily_energy_met_MU.csv',
        'data/14983362/daily_energy_met_MU.csv',
        '../data/14983362/daily_energy_met_MU.csv'
    ]

    for file_path in possible_paths:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)

                if column_name not in df.columns:
                    if state_name == 'NCT of Delhi' and 'Delhi' in df.columns:
                        column_name = 'Delhi'
                    elif state_name == 'Andaman and Nicobar' and 'Andaman and Nicobar Islands' in df.columns:
                        column_name = 'Andaman and Nicobar Islands'
                    else:
                        continue

                date_col = 'Date' if 'Date' in df.columns else 'date'
                if date_col not in df.columns:
                    continue

                demand_df = pd.DataFrame({
                    'date': pd.to_datetime(df[date_col]),
                    'actual_demand_MU': pd.to_numeric(df[column_name], errors='coerce')
                })

                demand_df = demand_df.dropna()

                if len(demand_df) > 0:
                    return demand_df

            except Exception as e:
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
    baseline = fallback_baselines.get(state_name)
    if baseline is None:
        st.warning(f"⚠️ No fallback baseline defined for '{state_name}'. Using 100 MU — forecasts will be inaccurate.")
        return 100.0
    return baseline


@st.cache_data
def get_state_historical_averages(state_name: str) -> dict:
    """
    Compute per-state historical averages for inference-time feature imputation.
    Uses daily_energy_met_MU.csv. Returns a dict of feature_name -> float.
    Falls back to empty dict (callers use 0.0) if data is unavailable.
    """
    column_map = {
        'Andaman and Nicobar': 'Andaman and Nicobar Islands',
        'NCT of Delhi': 'Delhi',
    }

    possible_paths = [
        'data/daily_energy_met_MU.csv',
        '../data/daily_energy_met_MU.csv',
        'data/14983362/daily_energy_met_MU.csv',
        '../data/14983362/daily_energy_met_MU.csv',
    ]

    for file_path in possible_paths:
        if not os.path.exists(file_path):
            continue
        try:
            df = pd.read_csv(file_path)
            date_col = 'Date' if 'Date' in df.columns else 'date'
            df['_date'] = pd.to_datetime(df[date_col])

            col = column_map.get(state_name, state_name)
            if col not in df.columns or 'Total' not in df.columns:
                continue

            df['_state'] = pd.to_numeric(df[col], errors='coerce')
            df['_total'] = pd.to_numeric(df['Total'], errors='coerce')
            df = df.dropna(subset=['_state', '_total'])

            if len(df) == 0:
                continue

            mean_daily_mu = df['_state'].mean()
            energymet_7d_avg = df['_state'].rolling(7, min_periods=1).mean().mean()
            monthly_energy_avg = df.groupby(df['_date'].dt.month)['_state'].mean().mean()
            monthly_rolling_mean = df['_state'].rolling(30, min_periods=1).mean().mean()
            demand_rolling = energymet_7d_avg
            state_share = (df['_state'] / df['_total'].replace(0, np.nan)).mean()
            max_demand_mw = mean_daily_mu * 1000 / (0.60 * 24)

            return {
                'energymet_7d_avg': float(energymet_7d_avg),
                'monthly_energy_avg': float(monthly_energy_avg),
                'monthly_rolling_mean': float(monthly_rolling_mean),
                'demand_rolling': float(demand_rolling),
                'state_share': float(state_share),
                'max_demand_mw': float(max_demand_mw),
            }
        except Exception:
            continue

    return {}


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

    historical_demand_sorted = historical_demand.sort_values('date')
    before_date = historical_demand_sorted[historical_demand_sorted['date'] < date]

    if len(before_date) < 30:
        if len(before_date) > 0:
            return before_date['actual_demand_MU'].tail(len(before_date)).mean()
        return None

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
    predictions_processed = np.expm1(predictions)

    if historical_demand is None:
        historical_demand = load_historical_demand(state_name)

    if historical_demand is None or len(historical_demand) == 0:
        fallback_baseline = get_fallback_baseline(state_name)
        st.caption(f"📌 Historical demand data unavailable for **{state_name}** — using reference baseline of {fallback_baseline:.0f} MU.")
        return predictions_processed * fallback_baseline

    if len(dates) > 0:
        first_date = pd.Timestamp(dates.iloc[0])
        current_date = pd.Timestamp.now()

        if first_date > current_date:
            if len(historical_demand) >= 30:
                baseline = historical_demand['actual_demand_MU'].tail(30).mean()
            elif len(historical_demand) > 0:
                baseline = historical_demand['actual_demand_MU'].mean()
            else:
                fallback_baseline = get_fallback_baseline(state_name)
                st.caption(f"📌 Historical demand data unavailable for **{state_name}** — using reference baseline of {fallback_baseline:.0f} MU.")
                return predictions_processed * fallback_baseline

            return predictions_processed * baseline
        else:
            baselines = dates.apply(lambda d: calculate_state_30d_baseline(historical_demand, d))
            if baselines.isna().any():
                mean_baseline = baselines.mean() if baselines.notna().any() else historical_demand['actual_demand_MU'].mean()
                baselines = baselines.fillna(mean_baseline)

            return predictions_processed * baselines.values
    else:
        fallback_baseline = get_fallback_baseline(state_name)
        st.caption(f"📌 Historical demand data unavailable for **{state_name}** — using reference baseline of {fallback_baseline:.0f} MU.")
        return predictions_processed * fallback_baseline
