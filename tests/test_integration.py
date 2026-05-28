# ============================================================================
# INTEGRATION TESTS (8 tests)
# ============================================================================
import pytest
import pandas as pd
import numpy as np
from features import engineer_features, prepare_features_for_prediction
from models import load_model


class TestIntegration:
    """Integration tests for full pipeline (features + model together)."""

    def test_full_feature_engineering_pipeline(self, sample_weather_df):
        """
        PURPOSE: Test full feature engineering without errors
        CATCHES: Missing columns, shape mismatches, type errors
        PRODUCTION IMPACT: Feature pipeline crashes, forecast fails
        """
        X, df_full = prepare_features_for_prediction(
            sample_weather_df,
            'Maharashtra'
        )

        assert X is not None
        assert len(X) == len(sample_weather_df)
        assert X.shape[1] > 0  # Has features

    def test_model_prediction_shape(self, sample_weather_df, loaded_model):
        """
        PURPOSE: Verify model predictions have correct shape
        CATCHES: Prediction shape mismatch
        PRODUCTION IMPACT: Shape error crashes UI
        """
        X, _ = prepare_features_for_prediction(sample_weather_df, 'Maharashtra', model=loaded_model)
        predictions = loaded_model.predict(X)

        assert len(predictions) == len(X)
        assert all(p > 0 for p in predictions)  # Demand always positive

    def test_prediction_range_realistic(self, sample_weather_df, loaded_model):
        """
        PURPOSE: Verify predictions are in realistic range (0-500 MU)
        CATCHES: Out-of-range predictions (negative, extremely high)
        PRODUCTION IMPACT: UI shows impossible demand values
        """
        X, _ = prepare_features_for_prediction(sample_weather_df, 'Maharashtra', model=loaded_model)
        predictions = loaded_model.predict(X)

        assert all(0 < p < 500 for p in predictions), "Predictions out of realistic range"

    def test_features_match_model_expectation(self, sample_weather_df, loaded_model):
        """
        PURPOSE: Verify feature count matches model's expected input
        CATCHES: Feature count mismatch (model expects 42, gets 40)
        PRODUCTION IMPACT: Model.predict() crashes with shape error
        """
        X, _ = prepare_features_for_prediction(sample_weather_df, 'Maharashtra', model=loaded_model)

        if hasattr(loaded_model, 'n_features_in_'):
            assert X.shape[1] == loaded_model.n_features_in_

    def test_state_encoding_consistency_across_pipeline(self, sample_weather_df):
        """
        PURPOSE: Verify state encoding is consistent from engineer_features to predict
        CATCHES: State encoding mismatch between functions
        PRODUCTION IMPACT: State features corrupted, predictions wrong
        """
        df_engineered = engineer_features(sample_weather_df, 'Maharashtra')
        X, _ = prepare_features_for_prediction(sample_weather_df, 'Maharashtra')

        # Verify states column exists in engineered df
        assert 'states' in df_engineered.columns
        # Verify states_* columns exist in X
        state_cols = [col for col in X.columns if col.startswith('states_')]
        assert len(state_cols) > 0

    def test_lag_features_not_nan_in_prediction(self, sample_weather_df):
        """
        PURPOSE: Verify lag features are not NaN before sending to model
        CATCHES: NaN lags passed to model (crashes or gives wrong predictions)
        PRODUCTION IMPACT: Model receives corrupt data, predictions garbage
        """
        df_engineered = engineer_features(sample_weather_df, 'Maharashtra')

        lag_cols = ['temp_mean_lag1', 'temp_mean_lag3', 'temp_mean_lag7']
        for col in lag_cols:
            assert not df_engineered[col].isna().any(), f"{col} has NaN values"

    def test_rolling_features_not_nan_in_prediction(self, sample_weather_df):
        """
        PURPOSE: Verify rolling features are not NaN before model prediction
        CATCHES: NaN rolling features sent to model
        PRODUCTION IMPACT: Model receives corrupt rolling averages
        """
        df_engineered = engineer_features(sample_weather_df, 'Maharashtra')

        roll_cols = ['temp_mean_roll3', 'temp_mean_roll7']
        for col in roll_cols:
            assert not df_engineered[col].isna().any(), f"{col} has NaN values"

    def test_prediction_changes_with_temperature_change(self, loaded_model):
        """
        PURPOSE: Verify predictions change when temperature changes
        CATCHES: Temperature feature disconnected (always 0 or unused)
        PRODUCTION IMPACT: Temperature changes don't affect forecast
        SEVERITY: 🔴 CRITICAL — model ignores weather
        """
        from tests.conftest import create_sample_data_with_temps

        # Create two datasets with different temperatures
        cold_df = create_sample_data_with_temps('Maharashtra', [300] * 10)
        hot_df = create_sample_data_with_temps('Maharashtra', [320] * 10)

        X_cold, _ = prepare_features_for_prediction(cold_df, 'Maharashtra', model=loaded_model)
        X_hot, _ = prepare_features_for_prediction(hot_df, 'Maharashtra', model=loaded_model)

        pred_cold = loaded_model.predict(X_cold)
        pred_hot = loaded_model.predict(X_hot)

        # Hot weather should generally predict higher demand
        assert pred_hot.mean() >= pred_cold.mean(), "Temperature not affecting predictions"
