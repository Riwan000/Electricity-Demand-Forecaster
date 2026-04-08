# ============================================================================
# TAB 1: FORECAST
# ============================================================================
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from scipy import stats

from config import ALL_STATES
from data_loading import load_historical_demand, denormalize_predictions
from weather import get_weather_forecast
from features import prepare_features_for_prediction
from models import load_model_metadata, get_state_rmse
from rag import generate_forecast_summary


def render_forecast_tab(model, metadata, use_weather_api):
    """
    Render the Forecast tab UI and handle forecast generation.

    Parameters:
        model: Loaded XGBoost model
        metadata (dict): Model metadata
        use_weather_api (bool): Whether to use the live weather API
    """
    st.header("Demand Forecasting")
    st.markdown("Generate electricity demand forecasts for any Indian state using ML predictions.")

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
        st.write("")
        generate_btn = st.button("🚀 Generate Forecast", type="primary", use_container_width=True)

    _EASTERN_REGION_STATES = {"Bihar", "Chhattisgarh", "Jharkhand", "Odisha", "West Bengal"}
    if state in _EASTERN_REGION_STATES:
        st.warning(
            f"⚠️ **{state}** is in the Eastern Region (ER), which was not included as a "
            "separate region in the model's training data. Forecasts for this state are "
            "approximate. Accuracy will improve in a future model retrain."
        )

    if generate_btn:
        if model is None:
            st.error("❌ Model not loaded. Cannot generate forecast.")
            st.stop()

        with st.spinner("🔄 Generating forecast... This may take a few seconds."):
            try:
                start_date = datetime.now().date()

                weather_forecast = get_weather_forecast(
                    state,
                    start_date,
                    horizon_days,
                    use_api=use_weather_api
                )

                if weather_forecast is None or len(weather_forecast) == 0:
                    st.error("❌ Failed to get weather data. Please try again.")
                    st.stop()

                X_forecast, df_features = prepare_features_for_prediction(
                    weather_forecast, state, model=model, metadata=metadata
                )

                pred_log = model.predict(X_forecast)

                historical_demand = load_historical_demand(state)
                denorm_result = denormalize_predictions(pred_log, weather_forecast['date'], state, historical_demand)

                results_df = pd.DataFrame({
                    'date': weather_forecast['date'],
                    'forecasted_demand_MU': denorm_result,
                    'temperature_mean': weather_forecast['2m_temperature_mean'],
                    'temperature_max': weather_forecast['2m_temperature_max'],
                    'cdd': df_features['cdd'],
                    'extreme_heat': df_features['extreme_heat'],
                    'is_holiday': df_features['is_holiday']
                })

                metadata = load_model_metadata()
                forecast_summary = generate_forecast_summary(
                    results_df,
                    state,
                    horizon_days,
                    df_features,
                    metadata=metadata
                )

                st.session_state['last_forecast'] = {
                    'results_df': results_df,
                    'weather_forecast': weather_forecast,
                    'df_features': df_features,
                    'state': state,
                    'horizon_days': horizon_days,
                    'forecast_summary': forecast_summary
                }

                # Sanity check
                assert results_df['forecasted_demand_MU'].mean() > 10, \
                    f"Forecast values look like model-space, not MU. Mean: {results_df['forecasted_demand_MU'].mean():.2f}"

                st.write("🔍 **Forecast range (MU):**",
                         f"{results_df['forecasted_demand_MU'].min():.2f} - {results_df['forecasted_demand_MU'].max():.2f} MU")

                st.success(f"✅ Forecast generated successfully for **{state}** ({horizon_days} days)")

                if not st.session_state.get('generation_warning_shown'):
                    st.info(
                        "ℹ️ **Note on generation features**: Generation data (generation_mu, gen_rolling, etc.) "
                        "is unavailable at inference time and is set to 0.0. Demand-based features use "
                        "per-state historical averages. Forecast accuracy may be slightly reduced."
                    )
                    st.session_state['generation_warning_shown'] = True

                # Summary metrics
                st.subheader("📈 Forecast Summary")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Avg Daily Demand", f"{results_df['forecasted_demand_MU'].mean():.1f} MU")
                with col2:
                    st.metric("Peak Demand", f"{results_df['forecasted_demand_MU'].max():.1f} MU")
                with col3:
                    st.metric("Min Demand", f"{results_df['forecasted_demand_MU'].min():.1f} MU")
                with col4:
                    high_risk_days = results_df['extreme_heat'].sum()
                    st.metric("High Risk Days", f"{high_risk_days}",
                              delta=f"{high_risk_days/horizon_days*100:.0f}% of period")

                # Main forecast plot
                st.subheader("📊 Forecast Visualization")

                metadata = load_model_metadata()
                state_rmse = get_state_rmse(state, metadata)
                confidence_level = metadata.get('confidence_level', 0.9) if metadata else 0.9

                if state_rmse is not None:
                    z_score = stats.norm.ppf((1 + confidence_level) / 2)
                    margin = z_score * state_rmse
                    upper_bound = results_df['forecasted_demand_MU'] + margin
                    lower_bound = results_df['forecasted_demand_MU'] - margin
                    st.caption(f"📊 Using state-specific RMSE: {state_rmse:.2f} MU (Confidence: {confidence_level*100:.0f}%)")
                else:
                    st.warning(f"⚠️ State-specific RMSE not found for {state} in metadata. Confidence intervals not displayed.")
                    upper_bound = None
                    lower_bound = None

                fig = go.Figure()

                if state_rmse is not None and upper_bound is not None:
                    fig.add_trace(go.Scatter(
                        x=results_df['date'], y=upper_bound,
                        mode='lines', name=f'Upper Bound ({confidence_level*100:.0f}% CI)',
                        line=dict(width=0), showlegend=True,
                        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Upper: %{y:.2f} MU<extra></extra>'
                    ))
                    fig.add_trace(go.Scatter(
                        x=results_df['date'], y=lower_bound,
                        mode='lines', name=f'Lower Bound ({confidence_level*100:.0f}% CI)',
                        line=dict(width=0), fill='tonexty',
                        fillcolor='rgba(31, 119, 180, 0.2)', showlegend=True,
                        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Lower: %{y:.2f} MU<extra></extra>'
                    ))

                fig.add_trace(go.Scatter(
                    x=results_df['date'], y=results_df['forecasted_demand_MU'],
                    mode='lines+markers', name='Forecasted Demand',
                    line=dict(color='#1f77b4', width=3), marker=dict(size=8),
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Demand: %{y:.2f} MU<extra></extra>'
                ))

                peak_idx = results_df['forecasted_demand_MU'].idxmax()
                fig.add_trace(go.Scatter(
                    x=[results_df.loc[peak_idx, 'date']],
                    y=[results_df.loc[peak_idx, 'forecasted_demand_MU']],
                    mode="markers", marker=dict(size=12, color="cyan"),
                    name="Peak Demand",
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Peak Demand: %{y:.2f} MU<extra></extra>'
                ))

                heat_days = results_df[results_df['extreme_heat'] == 1]
                if len(heat_days) > 0:
                    fig.add_trace(go.Scatter(
                        x=heat_days['date'], y=heat_days['forecasted_demand_MU'],
                        mode='markers', name='High Risk (Heat / Uncertainty)',
                        marker=dict(symbol='triangle-up', size=15, color='red', line=dict(width=2, color='darkred')),
                        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>High Risk Day<br>Demand: %{y:.2f} MU<extra></extra>'
                    ))

                holiday_days = results_df[results_df['is_holiday'] == 1]
                if len(holiday_days) > 0:
                    fig.add_trace(go.Scatter(
                        x=holiday_days['date'], y=holiday_days['forecasted_demand_MU'],
                        mode='markers', name='Holiday',
                        marker=dict(symbol='star', size=12, color='orange', line=dict(width=1, color='darkorange')),
                        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Holiday<br>Demand: %{y:.2f} MU<extra></extra>'
                    ))

                fig.update_layout(
                    title=f"Electricity Demand Forecast - {state}",
                    xaxis_title="Date",
                    yaxis_title="Demand (MU - Million Units)",
                    hovermode='x unified',
                    height=500,
                    template='plotly_white',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

                st.plotly_chart(fig, use_container_width=True)

                with st.expander("📋 View Detailed Forecast Data"):
                    display_df = results_df.copy()
                    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                    display_df['temperature_mean'] = display_df['temperature_mean'] - 273.15
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
