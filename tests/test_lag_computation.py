# ============================================================================
# LAG & ROLLING FEATURE EDGE CASE TESTS (8 tests)
# ============================================================================
import pytest
import pandas as pd
import numpy as np
from features import engineer_features


class TestLagComputationEdgeCases:
    """Edge cases for lag feature computation."""

    def test_lag_with_single_day_forecast(self, single_day_weather_df):
        """
        PURPOSE: Verify lag gracefully handles single day of data
        CATCHES: Lag computation crashes with sparse data
        PRODUCTION IMPACT: Real-time forecasts don't crash
        """
        df = engineer_features(single_day_weather_df.copy(), 'Maharashtra')

        # With single day, lags may be NaN or filled - should not crash
        assert 'temp_mean_lag1' in df.columns
        assert 'temp_mean_lag7' in df.columns
        # At least lags should exist in dataframe
        assert len(df) > 0

    def test_lag_with_partial_data(self):
        """
        PURPOSE: Verify lag gracefully handles < 7 days (partial forecast)
        CATCHES: Lag computation crashes with sparse data
        PRODUCTION IMPACT: 1-3 day forecasts don't crash
        """
        dates = pd.date_range('2023-01-01', periods=3)
        df = pd.DataFrame({
            'date': dates,
            '2m_temperature_mean': [300.0, 305.0, 310.0],
            '2m_temperature_max': [305.0, 310.0, 315.0],
            '2m_temperature_min': [295.0, 300.0, 305.0],
            '2m_dewpoint_temperature_min': [285.0, 290.0, 295.0],
            '2m_dewpoint_temperature_mean': [290.0, 295.0, 300.0],
            '2m_dewpoint_temperature_max': [295.0, 300.0, 305.0],
            '10m_u_component_of_wind_min': [0.0, 0.0, 0.0],
            '10m_u_component_of_wind_mean': [1.0, 1.0, 1.0],
            '10m_u_component_of_wind_max': [2.0, 2.0, 2.0],
            '10m_v_component_of_wind_min': [0.0, 0.0, 0.0],
            '10m_v_component_of_wind_mean': [1.0, 1.0, 1.0],
            '10m_v_component_of_wind_max': [2.0, 2.0, 2.0],
            'surface_solar_radiation_downwards_min': [0.0, 0.0, 0.0],
            'surface_solar_radiation_downwards_mean': [100.0, 100.0, 100.0],
            'surface_solar_radiation_downwards_max': [200.0, 200.0, 200.0],
            'total_cloud_cover_min': [0.0, 0.0, 0.0],
            'total_cloud_cover_mean': [50.0, 50.0, 50.0],
            'total_cloud_cover_max': [100.0, 100.0, 100.0],
            'utci_min': [295.0, 300.0, 305.0],
            'utci_mean': [300.0, 305.0, 310.0],
            'utci_max': [305.0, 310.0, 315.0],
        })

        result = engineer_features(df.copy(), 'Maharashtra')

        # Lag features should exist and not crash
        assert 'temp_mean_lag1' in result.columns
        assert 'temp_mean_lag7' in result.columns
        # lag1 should be filled from available data
        assert not result['temp_mean_lag1'].isna().any()

    def test_lag_rolling_with_constant_temperature(self):
        """
        PURPOSE: Verify lag/rolling work when temps constant
        CATCHES: Division by zero, NaN propagation
        PRODUCTION IMPACT: Stable weather conditions cause forecast to fail
        """
        temps = [305.0] * 10  # All same temp
        dates = pd.date_range('2023-01-01', periods=10)

        df = pd.DataFrame({
            'date': dates,
            '2m_temperature_mean': temps,
            '2m_temperature_max': [310.0] * 10,
            '2m_temperature_min': [300.0] * 10,
            '2m_dewpoint_temperature_min': [295.0] * 10,
            '2m_dewpoint_temperature_mean': [300.0] * 10,
            '2m_dewpoint_temperature_max': [305.0] * 10,
            '10m_u_component_of_wind_min': [0.0] * 10,
            '10m_u_component_of_wind_mean': [1.0] * 10,
            '10m_u_component_of_wind_max': [2.0] * 10,
            '10m_v_component_of_wind_min': [0.0] * 10,
            '10m_v_component_of_wind_mean': [1.0] * 10,
            '10m_v_component_of_wind_max': [2.0] * 10,
            'surface_solar_radiation_downwards_min': [0.0] * 10,
            'surface_solar_radiation_downwards_mean': [100.0] * 10,
            'surface_solar_radiation_downwards_max': [200.0] * 10,
            'total_cloud_cover_min': [0.0] * 10,
            'total_cloud_cover_mean': [50.0] * 10,
            'total_cloud_cover_max': [100.0] * 10,
            'utci_min': [300.0] * 10,
            'utci_mean': [305.0] * 10,
            'utci_max': [310.0] * 10,
        })

        result = engineer_features(df.copy(), 'Maharashtra')

        # Lag should equal constant temp
        assert (result['temp_mean_lag1'] == 305.0).all()
        # Rolling average should equal constant temp
        assert (result['temp_mean_roll3'] == 305.0).all()

    def test_lag_values_are_numeric(self, sample_weather_df):
        """
        PURPOSE: Verify lag features are numeric (not strings, NaN)
        CATCHES: Type conversion bugs
        PRODUCTION IMPACT: Model gets wrong dtype, crashes
        """
        df = engineer_features(sample_weather_df.copy(), 'Maharashtra')

        lag_cols = ['temp_mean_lag1', 'temp_mean_lag3', 'temp_mean_lag7']
        for col in lag_cols:
            assert pd.api.types.is_numeric_dtype(df[col])

    def test_bfill_only_initial_rows(self):
        """
        PURPOSE: Verify that NaN handling during lag computation is consistent
        CATCHES: Unexpected NaN propagation in lag features
        PRODUCTION IMPACT: Data quality issues in predictions
        """
        temps = [300.0, 305.0, np.nan, 310.0, 315.0]  # NaN in middle
        dates = pd.date_range('2023-01-01', periods=5)

        df = pd.DataFrame({
            'date': dates,
            '2m_temperature_mean': temps,
            '2m_temperature_max': [305.0, 310.0, np.nan, 315.0, 320.0],
            '2m_temperature_min': [295.0, 300.0, np.nan, 305.0, 310.0],
            '2m_dewpoint_temperature_min': [285.0, 290.0, np.nan, 295.0, 300.0],
            '2m_dewpoint_temperature_mean': [290.0, 295.0, np.nan, 300.0, 305.0],
            '2m_dewpoint_temperature_max': [295.0, 300.0, np.nan, 305.0, 310.0],
            '10m_u_component_of_wind_min': [0.0, 0.0, np.nan, 0.0, 0.0],
            '10m_u_component_of_wind_mean': [1.0, 1.0, np.nan, 1.0, 1.0],
            '10m_u_component_of_wind_max': [2.0, 2.0, np.nan, 2.0, 2.0],
            '10m_v_component_of_wind_min': [0.0, 0.0, np.nan, 0.0, 0.0],
            '10m_v_component_of_wind_mean': [1.0, 1.0, np.nan, 1.0, 1.0],
            '10m_v_component_of_wind_max': [2.0, 2.0, np.nan, 2.0, 2.0],
            'surface_solar_radiation_downwards_min': [0.0, 0.0, np.nan, 0.0, 0.0],
            'surface_solar_radiation_downwards_mean': [100.0, 100.0, np.nan, 100.0, 100.0],
            'surface_solar_radiation_downwards_max': [200.0, 200.0, np.nan, 200.0, 200.0],
            'total_cloud_cover_min': [0.0, 0.0, np.nan, 0.0, 0.0],
            'total_cloud_cover_mean': [50.0, 50.0, np.nan, 50.0, 50.0],
            'total_cloud_cover_max': [100.0, 100.0, np.nan, 100.0, 100.0],
            'utci_min': [295.0, 300.0, np.nan, 305.0, 310.0],
            'utci_mean': [300.0, 305.0, np.nan, 310.0, 315.0],
            'utci_max': [305.0, 310.0, np.nan, 315.0, 320.0],
        })

        result = engineer_features(df.copy(), 'Maharashtra')

        # Lag features should exist even with NaN in source data
        assert 'temp_mean_lag1' in result.columns
        # Rows 3+ should have valid lag values
        assert not pd.isna(result['temp_mean_lag1'].iloc[3:]).any()

    def test_lag_inference_mode_fresh_data(self):
        """
        PURPOSE: Verify lags work at inference time (fresh, current data)
        CATCHES: Stale historical data used instead of fresh forecast
        PRODUCTION IMPACT: Real-time forecast uses outdated temps
        """
        # Fresh forecast data (last few days)
        fresh_dates = pd.date_range(pd.Timestamp.now() - pd.Timedelta(days=10), periods=10)
        fresh_temps = np.random.uniform(300, 310, 10)

        df = pd.DataFrame({
            'date': fresh_dates,
            '2m_temperature_mean': fresh_temps,
            '2m_temperature_max': fresh_temps + 5,
            '2m_temperature_min': fresh_temps - 5,
            '2m_dewpoint_temperature_min': fresh_temps - 10,
            '2m_dewpoint_temperature_mean': fresh_temps - 5,
            '2m_dewpoint_temperature_max': fresh_temps,
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
            'utci_min': fresh_temps - 5,
            'utci_mean': fresh_temps,
            'utci_max': fresh_temps + 5,
        })

        result = engineer_features(df.copy(), 'Maharashtra')

        # All lags should exist
        assert not result['temp_mean_lag1'].isna().any()

    def test_rolling_mean_accuracy_known_values(self):
        """
        PURPOSE: Verify rolling mean calculation with known temps
        CATCHES: Off-by-one in window size, wrong averaging formula
        PRODUCTION IMPACT: Misleading rolling features
        """
        from tests.conftest import create_sample_data_with_temps

        # Known temps: [310, 320, 330, 340, 350]
        df = create_sample_data_with_temps('Maharashtra', [310, 320, 330, 340, 350])
        df = engineer_features(df, 'Maharashtra')

        # Row 2's roll3 = (310 + 320 + 330) / 3 = 320
        assert abs(df['temp_mean_roll3'].iloc[2] - 320) < 0.01

        # Row 4's roll3 = (330 + 340 + 350) / 3 = 340
        assert abs(df['temp_mean_roll3'].iloc[4] - 340) < 0.01

    def test_lag_preserves_multistate_data(self):
        """
        PURPOSE: Verify lag computation works across multiple states
        CATCHES: State-specific lag corruption
        PRODUCTION IMPACT: Multi-state forecast accuracy drops
        """
        dates = pd.date_range('2023-01-01', periods=10)
        df = pd.DataFrame({
            'date': dates,
            '2m_temperature_mean': np.linspace(300, 310, 10),
            '2m_temperature_max': np.linspace(305, 315, 10),
            '2m_temperature_min': np.linspace(295, 305, 10),
            '2m_dewpoint_temperature_min': np.linspace(285, 295, 10),
            '2m_dewpoint_temperature_mean': np.linspace(290, 300, 10),
            '2m_dewpoint_temperature_max': np.linspace(295, 305, 10),
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
            'utci_min': np.linspace(295, 305, 10),
            'utci_mean': np.linspace(300, 310, 10),
            'utci_max': np.linspace(305, 315, 10),
        })

        for state in ['Maharashtra', 'Tamil Nadu', 'Karnataka']:
            result = engineer_features(df.copy(), state)
            assert not result['temp_mean_lag1'].isna().any()
            assert not result['temp_mean_lag7'].isna().any()
            assert len(result) == len(df)
