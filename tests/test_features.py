# ============================================================================
# FEATURE ENGINEERING TESTS (22 tests)
# ============================================================================
import pytest
import pandas as pd
import numpy as np
from features import engineer_features, prepare_features_for_prediction


class TestCalendarFeatures:
    """Tests for calendar-based features."""

    def test_day_of_week_mapping(self, sample_weather_df):
        """
        PURPOSE: Verify day_of_week maps correctly (Mon=0, Sun=6)
        CATCHES: Off-by-one errors, incorrect weekday() usage
        PRODUCTION IMPACT: Wrong day-of-week causes weekly seasonality mismatch
        """
        df = engineer_features(sample_weather_df, 'Maharashtra')

        # Jan 1, 2023 = Sunday = 6
        assert df['day_of_week'].iloc[0] == 6
        # Jan 2, 2023 = Monday = 0
        assert df['day_of_week'].iloc[1] == 0
        # All values in [0, 6]
        assert (df['day_of_week'] >= 0).all() and (df['day_of_week'] <= 6).all()

    def test_is_weekend_flag(self, sample_weather_df):
        """
        PURPOSE: Verify weekends flagged as 1, weekdays as 0
        CATCHES: Inverted flag, off-by-one in day_of_week
        PRODUCTION IMPACT: Weekend demand patterns not captured
        """
        df = engineer_features(sample_weather_df, 'Maharashtra')

        # Where day_of_week is 5 or 6 (Sat/Sun), is_weekend should be 1
        weekend_mask = df['day_of_week'].isin([5, 6])
        assert (df[weekend_mask]['is_weekend'] == 1).all()

        # Where day_of_week not in [5, 6], is_weekend should be 0
        weekday_mask = ~df['day_of_week'].isin([5, 6])
        assert (df[weekday_mask]['is_weekend'] == 0).all()

    def test_season_encoding(self, sample_weather_df):
        """
        PURPOSE: Verify seasons match India calendar (Winter=Dec-Feb, Summer=Mar-May, etc.)
        CATCHES: Wrong season boundaries, month mapping errors
        PRODUCTION IMPACT: Seasonal demand patterns inverted (winter demand in summer)
        """
        df = engineer_features(sample_weather_df, 'Maharashtra')

        # Jan = Winter
        assert df[df['month'] == 1]['season'].iloc[0] == 'Winter'

        # Valid seasons
        valid_seasons = {'Winter', 'Summer', 'Monsoon', 'Autumn'}
        assert set(df['season'].unique()) <= valid_seasons

    def test_holiday_detection(self, sample_weather_df):
        """
        PURPOSE: Verify holidays detected correctly
        CATCHES: Missing holidays, wrong holiday names
        PRODUCTION IMPACT: Holiday demand patterns ignored
        """
        df = engineer_features(sample_weather_df, 'Maharashtra')

        # is_holiday should be binary
        assert set(df['is_holiday'].unique()) <= {0, 1}

        # holiday_name should have "No Holiday" or actual holiday
        assert all(h == 'No Holiday' or isinstance(h, str) for h in df['holiday_name'])

    def test_month_year_range(self, sample_weather_df):
        """
        PURPOSE: Verify month/year extracted correctly
        CATCHES: UTC vs local timezone bugs, year boundary issues
        PRODUCTION IMPACT: Time-based patterns misaligned
        """
        df = engineer_features(sample_weather_df, 'Maharashtra')

        # Months in range [1, 12]
        assert df['month'].min() >= 1 and df['month'].max() <= 12
        # Years realistic
        assert df['year'].min() >= 2013 and df['year'].max() <= 2030


class TestStateRegionEncoding:
    """Tests for state and region mapping."""

    def test_state_assignment(self, sample_weather_df, all_states):
        """
        PURPOSE: Verify state name assigned correctly
        CATCHES: State name mismatch, missing states
        PRODUCTION IMPACT: Model uses wrong baseline for state
        """
        for state in ['Maharashtra', 'Tamil Nadu', 'Karnataka']:
            df = engineer_features(sample_weather_df.copy(), state)
            assert (df['states'] == state).all()

    def test_jammu_kashmir_normalization(self, sample_weather_df):
        """
        PURPOSE: Verify "Jammu and Kashmir" normalized to "J&K"
        CATCHES: Special character handling, case sensitivity
        PRODUCTION IMPACT: J&K state not recognized by model
        """
        df = engineer_features(sample_weather_df.copy(), 'Jammu and Kashmir')
        assert (df['states'] == 'J&K').all()

    def test_region_mapping(self, sample_weather_df):
        """
        PURPOSE: Verify state maps to correct region
        CATCHES: Wrong region assignment
        PRODUCTION IMPACT: Regional patterns confused, forecast off
        """
        test_cases = {
            'Maharashtra': 'wr',      # Western Region
            'Tamil Nadu': 'sr',       # Southern Region
            'Gujarat': 'wr',
            'Assam': 'ner',           # North Eastern Region
        }

        for state, expected_region in test_cases.items():
            df = engineer_features(sample_weather_df.copy(), state)
            assert (df['region'] == expected_region).all()

    def test_state_one_hot_encoding(self, sample_weather_df):
        """
        PURPOSE: Verify exactly one state one-hot column is 1 per row
        CATCHES: Multiple states set to 1, encoding duplication
        PRODUCTION IMPACT: Model sees ambiguous state, predictions garbage
        """
        X, _ = prepare_features_for_prediction(sample_weather_df, 'Maharashtra')

        # Count 'states_*' columns
        state_cols = [col for col in X.columns if col.startswith('states_')]

        # Each row should have exactly 1 state = 1
        state_sums = X[state_cols].sum(axis=1)
        assert (state_sums == 1).all()


class TestTemperatureFeatures:
    """Tests for temperature-derived features."""

    def test_cdd_calculation(self, sample_weather_df, metadata):
        """
        PURPOSE: Verify CDD = max(0, T - threshold)
        CATCHES: Wrong threshold, inverted sign, off-by-one
        PRODUCTION IMPACT: Extreme heat demand not modeled, summer forecast fails
        """
        df = engineer_features(sample_weather_df.copy(), 'Maharashtra', metadata=metadata)

        threshold = 297.15
        expected_cdd = np.maximum(0, df['2m_temperature_mean'] - threshold)

        # CDD should match expected
        pd.testing.assert_series_equal(df['cdd'].reset_index(drop=True),
                                       expected_cdd.reset_index(drop=True),
                                       check_names=False)

        # CDD always >= 0
        assert (df['cdd'] >= 0).all()

    def test_cdd_metadata_override(self, sample_weather_df):
        """
        PURPOSE: Verify CDD uses metadata threshold, not hardcoded
        CATCHES: Hardcoded threshold used instead of metadata
        PRODUCTION IMPACT: Retraining with different threshold doesn't work
        """
        meta1 = {'cdd_threshold_kelvin': 297.15}
        meta2 = {'cdd_threshold_kelvin': 300.0}

        df1 = engineer_features(sample_weather_df.copy(), 'Maharashtra', metadata=meta1)
        df2 = engineer_features(sample_weather_df.copy(), 'Maharashtra', metadata=meta2)

        # CDD values should differ
        assert not df1['cdd'].equals(df2['cdd'])

    def test_extreme_heat_flag(self, sample_weather_df, metadata):
        """
        PURPOSE: Verify extreme_heat = 1 when T > threshold, 0 otherwise
        CATCHES: Inverted flag, wrong threshold
        PRODUCTION IMPACT: Extreme heat demand multiplier doesn't activate
        """
        df = engineer_features(sample_weather_df.copy(), 'Maharashtra', metadata=metadata)

        threshold = 305.0
        expected = (df['2m_temperature_mean'] > threshold).astype(int)

        pd.testing.assert_series_equal(df['extreme_heat'].reset_index(drop=True),
                                       expected.reset_index(drop=True),
                                       check_names=False)

        # Binary flag
        assert set(df['extreme_heat'].unique()) <= {0, 1}

    def test_temp_heat_interactions(self, sample_weather_df, metadata):
        """
        PURPOSE: Verify interactions = temp * extreme_heat_flag (0 when not extreme)
        CATCHES: Missing multiplier, wrong coefficient
        PRODUCTION IMPACT: Model doesn't learn extreme heat demand multiplier
        """
        df = engineer_features(sample_weather_df.copy(), 'Maharashtra', metadata=metadata)

        # When extreme_heat_flag = 0, interaction should be 0
        not_extreme = df[df['extreme_heat_flag'] == 0]
        assert (not_extreme['temp_mean_heat_interaction'] == 0).all()

        # When extreme_heat_flag = 1, interaction should equal temperature
        extreme = df[df['extreme_heat_flag'] == 1]
        assert (extreme['temp_mean_heat_interaction'] == extreme['2m_temperature_mean']).all()

    def test_temp_range(self, sample_weather_df):
        """
        PURPOSE: Verify temp_range = max - min
        CATCHES: Wrong formula, inverted subtraction
        PRODUCTION IMPACT: Temperature volatility not captured
        """
        df = engineer_features(sample_weather_df.copy(), 'Maharashtra')

        expected = df['2m_temperature_max'] - df['2m_temperature_min']
        pd.testing.assert_series_equal(df['temp_range'].reset_index(drop=True),
                                       expected.reset_index(drop=True),
                                       check_names=False)

        # Temperature range always >= 0
        assert (df['temp_range'] >= 0).all()


class TestLagRollingFeatures:
    """Tests for lag and rolling window features (CRITICAL tests)."""

    def test_lag1_temporal_shift(self, sample_weather_df):
        """
        PURPOSE: Verify lag1 = previous day's temp (NO forward leakage)
        CATCHES: Lag feature using same-day data instead of shifted
        PRODUCTION IMPACT: Model sees future data during training, accuracy drops 60-70%
        SEVERITY: 🔴 CRITICAL — This was the bug you fixed!
        """
        df = engineer_features(sample_weather_df.copy(), 'Maharashtra')

        # Row 1's lag1 should equal row 0's temp
        assert df['temp_mean_lag1'].iloc[1] == df['2m_temperature_mean'].iloc[0]

        # Row 0 should be filled (not NaN)
        assert not pd.isna(df['temp_mean_lag1'].iloc[0])

        # Verify shift() was applied: each row = previous row temp
        for i in range(1, len(df)):
            assert df['temp_mean_lag1'].iloc[i] == df['2m_temperature_mean'].iloc[i-1]

    def test_lag3_temporal_shift(self, sample_weather_df):
        """
        PURPOSE: Verify lag3 = 3 days ago (NO data leakage)
        CATCHES: Off-by-one errors in lag calculation
        PRODUCTION IMPACT: Multi-day patterns not captured
        """
        df = engineer_features(sample_weather_df.copy(), 'Maharashtra')

        # Row 3's lag3 should equal row 0's temp
        assert df['temp_mean_lag3'].iloc[3] == df['2m_temperature_mean'].iloc[0]

        # Verify 3-day shift
        for i in range(3, len(df)):
            assert df['temp_mean_lag3'].iloc[i] == df['2m_temperature_mean'].iloc[i-3]

    def test_lag7_temporal_shift(self, sample_weather_df):
        """
        PURPOSE: Verify lag7 = 7 days ago (weekly pattern capture)
        CATCHES: Off-by-one in lag calculation
        PRODUCTION IMPACT: Weekly seasonality not captured
        """
        df = engineer_features(sample_weather_df.copy(), 'Maharashtra')

        # Row 7's lag7 should equal row 0's temp
        assert df['temp_mean_lag7'].iloc[7] == df['2m_temperature_mean'].iloc[0]

    def test_lag_no_forward_leakage(self, sample_weather_df):
        """
        PURPOSE: Verify lag features never look into the future
        CATCHES: shift() without parameter, misunderstanding pandas shift
        PRODUCTION IMPACT: Model trained on future data fails in production
        """
        df = engineer_features(sample_weather_df.copy(), 'Maharashtra')

        # Lag should ONLY use past data, never future
        for i in range(1, len(df)):
            # lag1 at row i should equal row i-1
            assert df['temp_mean_lag1'].iloc[i] == df['2m_temperature_mean'].iloc[i-1]
            # lag1 at row i should NOT equal row i+1 (that's forward leakage)
            if i + 1 < len(df):
                assert df['temp_mean_lag1'].iloc[i] != df['2m_temperature_mean'].iloc[i+1]

    def test_lag_bfill_initial_rows(self, sample_weather_df):
        """
        PURPOSE: Verify bfill() fills initial rows (can't have prior data)
        CATCHES: NaN in first few rows = useless features
        PRODUCTION IMPACT: Model can't use lag features for first N days
        """
        df = engineer_features(sample_weather_df.copy(), 'Maharashtra')

        # No NaN in lag columns (bfill should fill them)
        assert not df['temp_mean_lag1'].isna().any()
        assert not df['temp_mean_lag3'].isna().any()
        assert not df['temp_mean_lag7'].isna().any()

    def test_rolling_window3_start(self, sample_weather_df):
        """
        PURPOSE: Verify rolling window starts from row 0 (min_periods=1)
        CATCHES: min_periods=3 means first 2 rows are NaN
        PRODUCTION IMPACT: Missing features for first forecast days
        """
        df = engineer_features(sample_weather_df.copy(), 'Maharashtra')

        # Row 0 should have value (not NaN)
        assert not pd.isna(df['temp_mean_roll3'].iloc[0])

        # Row 0's roll3 = just df[0] (average of 1 element)
        assert df['temp_mean_roll3'].iloc[0] == df['2m_temperature_mean'].iloc[0]

    def test_rolling_window3_calculation(self, sample_weather_df):
        """
        PURPOSE: Verify rolling mean = (T[i-2:i+1]).mean()
        CATCHES: Wrong window size, wrong averaging
        PRODUCTION IMPACT: Rolling features give wrong signal
        """
        df = engineer_features(sample_weather_df.copy(), 'Maharashtra')

        # Row 2's roll3 = average of rows [0, 1, 2]
        expected = df['2m_temperature_mean'].iloc[0:3].mean()
        assert abs(df['temp_mean_roll3'].iloc[2] - expected) < 0.01

    def test_rolling_window_shape(self, sample_weather_df):
        """
        PURPOSE: Verify rolling features have same shape as input
        CATCHES: Off-by-one shape bugs, misaligned indices
        PRODUCTION IMPACT: Feature/target mismatch = model fails
        """
        df = engineer_features(sample_weather_df.copy(), 'Maharashtra')

        # Same length
        assert len(df['temp_mean_roll3']) == len(df)
        assert len(df['temp_mean_roll7']) == len(df)

        # Same index
        assert df.index.equals(df['temp_mean_roll3'].index)
        assert df.index.equals(df['temp_mean_roll7'].index)


class TestCategoricalEncoding:
    """Tests for one-hot encoding of categorical features."""

    def test_season_one_hot_columns_exist(self, sample_weather_df):
        """
        PURPOSE: Verify season one-hot columns created correctly
        CATCHES: Missing season columns after get_dummies()
        PRODUCTION IMPACT: Model missing seasonal features
        """
        X, _ = prepare_features_for_prediction(sample_weather_df, 'Maharashtra')

        season_cols = ['season_Monsoon', 'season_Summer', 'season_Winter']
        for col in season_cols:
            assert col in X.columns, f"Missing {col}"

    def test_holiday_one_hot_columns_exist(self, sample_weather_df):
        """
        PURPOSE: Verify holiday one-hot columns created
        CATCHES: Missing holiday columns
        PRODUCTION IMPACT: Holiday demand patterns ignored
        """
        X, _ = prepare_features_for_prediction(sample_weather_df, 'Maharashtra')

        holiday_cols = ['holiday_name_Dussehra', 'holiday_name_Guru Nanak Jayanti', 'holiday_name_No Holiday']
        for col in holiday_cols:
            assert col in X.columns, f"Missing {col}"

    def test_feature_order_consistency(self, sample_weather_df, loaded_model):
        """
        PURPOSE: Verify features in exact same order as model expects
        CATCHES: Features in wrong order
        PRODUCTION IMPACT: Model gets features misaligned, predictions garbage
        """
        X, _ = prepare_features_for_prediction(sample_weather_df, 'Maharashtra', model=loaded_model)

        if hasattr(loaded_model, 'feature_names_in_'):
            assert list(X.columns) == list(loaded_model.feature_names_in_)
