# ============================================================================
# TAB 2: WEATHER IMPACT
# ============================================================================
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from config import ALL_STATES
from data_loading import load_historical_data, load_historical_demand, denormalize_predictions
from weather import get_weather_forecast
from features import prepare_features_for_prediction
from visualization import (
    plot_demand_temperature_overlay,
    plot_cdd_timeline,
    plot_feature_importance,
    get_feature_importance,
    categorize_feature,
    generate_weather_insight,
    get_seasonal_comparison
)


def render_weather_tab(model, metadata, use_weather_api):
    """
    Render the Weather Impact Analysis tab UI.

    Parameters:
        model: Loaded XGBoost model
        metadata (dict): Model metadata
        use_weather_api (bool): Whether to use the live weather API
    """
    st.header("🌡️ Weather Impact Analysis")
    st.markdown("**Weather amplifies demand patterns rather than solely driving them.** Explore how temperature, CDD, and weather variables influence deviations from baseline demand patterns.")

    st.info("📊 **Important**: Demand values shown are normalized relative to each state's historical baseline to enable cross-state comparison. These are not absolute MU values.")

    if model is None:
        st.error("❌ Model not loaded. Please check model path in settings.")
        st.stop()

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
        st.write("")
        analyze_btn = st.button("🔍 Analyze Weather Impact", type="primary", use_container_width=True)

    if analyze_btn:
        with st.spinner("🔄 Analyzing weather impact... This may take a few seconds."):
            try:
                if analysis_mode == "Forecast Analysis":
                    if 'last_forecast' in st.session_state:
                        results_df = st.session_state['last_forecast']['results_df']
                        weather_forecast = st.session_state['last_forecast']['weather_forecast']
                        df_features = st.session_state['last_forecast']['df_features']
                        state_used = st.session_state['last_forecast']['state']

                        if state_used != state_weather:
                            st.warning(f"⚠️ Last forecast was for {state_used}. Generating new forecast for {state_weather}...")
                            raise ValueError("State mismatch")
                    else:
                        start_date = datetime.now().date()
                        horizon_days = 14

                        weather_forecast = get_weather_forecast(state_weather, start_date, horizon_days, use_api=use_weather_api)

                        if weather_forecast is None or len(weather_forecast) == 0:
                            st.error("❌ Failed to get weather data. Please try again.")
                            st.stop()

                        X_forecast, df_features = prepare_features_for_prediction(weather_forecast, state_weather, model=model, metadata=metadata)
                        pred_log = model.predict(X_forecast)
                        historical_demand = load_historical_demand(state_weather)
                        pred_mu = denormalize_predictions(pred_log, weather_forecast['date'], state_weather, historical_demand)

                        results_df = pd.DataFrame({
                            'date': weather_forecast['date'],
                            'forecasted_demand_MU': pred_mu,
                            'temperature_mean': weather_forecast['2m_temperature_mean'],
                            'temperature_max': weather_forecast['2m_temperature_max'],
                            'cdd': df_features['cdd'],
                            'extreme_heat': df_features['extreme_heat'],
                            'is_holiday': df_features['is_holiday']
                        })

                else:  # Historical Analysis
                    historical_df = load_historical_data(state_weather)
                    if historical_df is None:
                        st.error(f"❌ No historical data found for {state_weather}.")
                        st.stop()

                    historical_df = historical_df.sort_values('date').tail(90)

                    X_historical, df_features = prepare_features_for_prediction(historical_df, state_weather, model=model, metadata=metadata)
                    pred_log = model.predict(X_historical)
                    historical_demand = load_historical_demand(state_weather)
                    pred_mu = denormalize_predictions(pred_log, historical_df['date'], state_weather, historical_demand)

                    results_df = pd.DataFrame({
                        'date': historical_df['date'],
                        'forecasted_demand_MU': pred_mu,
                        'temperature_mean': historical_df['2m_temperature_mean'],
                        'temperature_max': historical_df['2m_temperature_max'],
                        'cdd': df_features['cdd'],
                        'extreme_heat': df_features['extreme_heat'],
                        'is_holiday': df_features['is_holiday']
                    })

                    weather_forecast = historical_df[['date', '2m_temperature_mean', '2m_temperature_max']].copy()

                st.session_state['weather_analysis'] = {
                    'results_df': results_df,
                    'weather_forecast': weather_forecast,
                    'df_features': df_features,
                    'state': state_weather,
                    'mode': analysis_mode
                }

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

                st.markdown("### Temperature vs Demand Overlay")
                st.caption("💡 The baseline (dashed gray line) shows demand without weather signal. The difference demonstrates weather impact.")
                temp_overlay_fig = plot_demand_temperature_overlay(results_df, weather_forecast, df_features)
                if temp_overlay_fig:
                    st.plotly_chart(temp_overlay_fig, use_container_width=True)

                st.markdown("### Cooling Degree Days (CDD) Timeline")
                cdd_fig = plot_cdd_timeline(results_df, df_features)
                if cdd_fig:
                    st.plotly_chart(cdd_fig, use_container_width=True)

                st.markdown("### Model Dependency (Relative Influence)")
                col1, col2 = st.columns([1, 3])
                with col1:
                    show_weather_only = st.checkbox(
                        "Weather-Only Features",
                        value=False,
                        help="Show only weather-related features"
                    )

                feature_importance = get_feature_importance(model, top_n=15, filter_weather_only=show_weather_only)
                if feature_importance:
                    importance_fig = plot_feature_importance(feature_importance)
                    if importance_fig:
                        st.plotly_chart(importance_fig, use_container_width=True)

                    st.caption(
                        "💡 **Note**: Short-term demand history explains most variance; weather variables primarily influence deviations during extreme conditions. "
                        "This model is primarily autoregressive (momentum-based) with weather as a secondary signal."
                    )

                    with st.expander("📋 Top 5 Most Influential Features"):
                        top_5 = dict(list(feature_importance.items())[:5])
                        for i, (feat, imp) in enumerate(top_5.items(), 1):
                            category = categorize_feature(feat)
                            st.markdown(f"{i}. **{feat}** ({category}) - Relative Influence: {imp:.4f}")
                else:
                    st.warning("⚠️ Could not extract feature importance from model.")

                # Seasonal Comparison
                if analysis_mode == "Historical Analysis" or 'last_forecast' in st.session_state:
                    st.markdown("### Seasonal Comparison")

                    if len(df_features) > 0:
                        current_season = df_features['season'].iloc[0] if 'season' in df_features.columns else None

                        if current_season:
                            seasonal_stats = get_seasonal_comparison(state_weather, current_season)

                            if seasonal_stats:
                                seasons = list(seasonal_stats.keys())
                                temp_means = [seasonal_stats[s]['temp_mean'] for s in seasons]
                                cdd_means = [seasonal_stats[s]['cdd_mean'] for s in seasons]

                                fig_seasonal = make_subplots(specs=[[{"secondary_y": True}]])
                                fig_seasonal.add_trace(
                                    go.Bar(x=seasons, y=temp_means, name='Avg Temperature',
                                           marker_color='#ff7f0e',
                                           hovertemplate='<b>%{x}</b><br>Temp: %{y:.1f}°C<extra></extra>'),
                                    secondary_y=False,
                                )
                                fig_seasonal.add_trace(
                                    go.Bar(x=seasons, y=cdd_means, name='Avg CDD',
                                           marker_color='#d62728',
                                           hovertemplate='<b>%{x}</b><br>CDD: %{y:.2f}°C<extra></extra>'),
                                    secondary_y=True,
                                )
                                fig_seasonal.update_xaxes(title_text="Season")
                                fig_seasonal.update_yaxes(title_text="Temperature (°C)", secondary_y=False)
                                fig_seasonal.update_yaxes(title_text="CDD (°C)", secondary_y=True)
                                fig_seasonal.update_layout(
                                    title="Seasonal Temperature and CDD Comparison",
                                    hovermode='x unified', height=400, template='plotly_white',
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                )
                                st.plotly_chart(fig_seasonal, use_container_width=True)

                                with st.expander("📋 Detailed Seasonal Statistics"):
                                    seasonal_df = pd.DataFrame(seasonal_stats).T
                                    seasonal_df.index.name = 'Season'
                                    st.dataframe(seasonal_df.style.format({
                                        'temp_mean': '{:.1f}°C', 'temp_max': '{:.1f}°C',
                                        'cdd_mean': '{:.2f}°C', 'extreme_heat_pct': '{:.1f}%',
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
                    start_date = datetime.now().date()
                    horizon_days = 14
                    weather_forecast = get_weather_forecast(state_weather, start_date, horizon_days, use_api=use_weather_api)
                    X_forecast, df_features = prepare_features_for_prediction(weather_forecast, state_weather, model=model, metadata=metadata)
                    pred_log = model.predict(X_forecast)
                    historical_demand = load_historical_demand(state_weather)
                    pred_mu = denormalize_predictions(pred_log, weather_forecast['date'], state_weather, historical_demand)
                    results_df = pd.DataFrame({
                        'date': weather_forecast['date'],
                        'forecasted_demand_MU': pred_mu,
                        'temperature_mean': weather_forecast['2m_temperature_mean'],
                        'temperature_max': weather_forecast['2m_temperature_max'],
                        'cdd': df_features['cdd'],
                        'extreme_heat': df_features['extreme_heat'],
                        'is_holiday': df_features['is_holiday']
                    })
                    st.rerun()
                else:
                    st.error(f"❌ Error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error analyzing weather impact: {str(e)}")
                st.exception(e)

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
