# ============================================================================
# WEATHER IMPACT VISUALIZATION FUNCTIONS
# ============================================================================
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_loading import load_historical_data
from features import engineer_features


def get_feature_importance(model, top_n=15, filter_weather_only=False):
    """
    Extract feature importance from XGBoost model.

    Parameters:
        model: Trained XGBoost model
        top_n (int): Number of top features to return
        filter_weather_only (bool): If True, only return weather-related features

    Returns:
        dict: Dictionary with feature names as keys and importance scores as values
    """
    if model is None or not hasattr(model, 'feature_importances_'):
        return None

    try:
        feature_names = model.feature_names_in_
        importances = model.feature_importances_

        importance_dict = dict(zip(feature_names, importances))

        if filter_weather_only:
            weather_categories = ['Temperature', 'Humidity', 'Weather']
            weather_features = {}
            for feat, imp in importance_dict.items():
                category = categorize_feature(feat)
                if category in weather_categories:
                    weather_features[feat] = imp
            importance_dict = weather_features

        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        return dict(list(sorted_importance.items())[:top_n])
    except Exception:
        return None


def categorize_feature(feature_name):
    """
    Categorize a feature into a group for color coding.

    Parameters:
        feature_name (str): Name of the feature

    Returns:
        str: Category name
    """
    feature_lower = feature_name.lower()

    if any(x in feature_lower for x in ['temperature', 'temp', 'cdd']):
        return 'Temperature'
    elif any(x in feature_lower for x in ['dewpoint', 'utci', 'humidity']):
        return 'Humidity'
    elif any(x in feature_lower for x in ['month', 'day', 'weekend', 'holiday', 'season']):
        return 'Calendar'
    elif any(x in feature_lower for x in ['solar', 'radiation', 'cloud', 'wind']):
        return 'Weather'
    elif any(x in feature_lower for x in ['generation', 'energy', 'demand', 'rolling', 'avg']):
        return 'Historical'
    elif any(x in feature_lower for x in ['state', 'region']):
        return 'Location'
    else:
        return 'Other'


def plot_demand_temperature_overlay(results_df, weather_df, df_features=None):
    """
    Create dual-axis plot showing demand and temperature on the same timeline.

    Parameters:
        results_df: DataFrame with forecasted demand and dates
        weather_df: DataFrame with temperature data
        df_features: Optional DataFrame with features to calculate baseline

    Returns:
        plotly.graph_objects.Figure
    """
    merged_df = pd.merge(results_df, weather_df, on='date', how='inner')

    temp_mean_c = merged_df['2m_temperature_mean'] - 273.15
    temp_max_c = merged_df['2m_temperature_max'] - 273.15

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if df_features is not None and 'energymet_7d_avg' in df_features.columns:
        baseline_df = pd.merge(results_df, df_features[['date', 'energymet_7d_avg']], on='date', how='inner')
        if len(baseline_df) > 0:
            fig.add_trace(
                go.Scatter(
                    x=baseline_df['date'],
                    y=baseline_df['energymet_7d_avg'],
                    mode='lines',
                    name='Baseline (7-day avg, no weather)',
                    line=dict(color='gray', width=2, dash='dash'),
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Baseline: %{y:.2f}<extra></extra>'
                ),
                secondary_y=False,
            )

    fig.add_trace(
        go.Scatter(
            x=merged_df['date'],
            y=merged_df['forecasted_demand_MU'],
            mode='lines+markers',
            name='Forecasted Demand (with weather)',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Demand: %{y:.2f} (normalized)<extra></extra>'
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=merged_df['date'],
            y=temp_mean_c,
            mode='lines+markers',
            name='Temperature (Mean)',
            line=dict(color='#ff7f0e', width=2),
            marker=dict(size=5),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Temp: %{y:.1f}°C<extra></extra>'
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=merged_df['date'],
            y=temp_max_c,
            mode='lines+markers',
            name='Temperature (Max)',
            line=dict(color='#d62728', width=2, dash='dash'),
            marker=dict(size=5),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Temp Max: %{y:.1f}°C<extra></extra>'
        ),
        secondary_y=True,
    )

    correlation = merged_df['forecasted_demand_MU'].corr(temp_mean_c)

    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Demand (normalized units)", secondary_y=False)
    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=True)
    fig.update_layout(
        title=f"Demand vs Temperature Overlay (Correlation: {correlation:.3f})",
        hovermode='x unified',
        height=500,
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def plot_cdd_timeline(results_df, df_features):
    """
    Plot CDD (Cooling Degree Days) as bars with demand overlay.

    Parameters:
        results_df: DataFrame with forecasted demand and dates
        df_features: DataFrame with engineered features including CDD

    Returns:
        plotly.graph_objects.Figure
    """
    merged_df = pd.merge(results_df, df_features, on='date', how='inner', suffixes=('_results', '_features'))

    if 'cdd_features' in merged_df.columns:
        cdd_values = merged_df['cdd_features']
    elif 'cdd_results' in merged_df.columns:
        cdd_values = merged_df['cdd_results']
    elif 'cdd' in merged_df.columns:
        cdd_values = merged_df['cdd']
    else:
        cdd_values = results_df['cdd'] if 'cdd' in results_df.columns else pd.Series([0.0] * len(merged_df))

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if 'extreme_heat_features' in merged_df.columns:
        extreme_heat_col = 'extreme_heat_features'
    elif 'extreme_heat_results' in merged_df.columns:
        extreme_heat_col = 'extreme_heat_results'
    elif 'extreme_heat' in merged_df.columns:
        extreme_heat_col = 'extreme_heat'
    else:
        extreme_heat_col = None
        merged_df['extreme_heat'] = 0

    heat_col_for_color = extreme_heat_col if extreme_heat_col else 'extreme_heat'

    fig.add_trace(
        go.Bar(
            x=merged_df['date'],
            y=cdd_values,
            name='CDD',
            marker=dict(
                color=merged_df[heat_col_for_color],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Extreme Heat", x=1.15),
                cmin=0,
                cmax=1
            ),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>CDD: %{y:.2f}°C<extra></extra>'
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=merged_df['date'],
            y=merged_df['forecasted_demand_MU'],
            mode='lines+markers',
            name='Forecasted Demand',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Demand: %{y:.2f} MU<extra></extra>'
        ),
        secondary_y=True,
    )

    if extreme_heat_col:
        extreme_heat_days = merged_df[merged_df[extreme_heat_col] == 1]
    else:
        extreme_heat_days = pd.DataFrame()

    if len(extreme_heat_days) > 0:
        fig.add_trace(
            go.Scatter(
                x=extreme_heat_days['date'],
                y=extreme_heat_days['forecasted_demand_MU'],
                mode='markers',
                name='Extreme Heat',
                marker=dict(symbol='triangle-up', size=12, color='red', line=dict(width=2, color='darkred')),
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Extreme Heat Day<br>Demand: %{y:.2f} MU<extra></extra>'
            ),
            secondary_y=True,
        )

    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="CDD (°C)", secondary_y=False)
    fig.update_yaxes(title_text="Demand (normalized units)", secondary_y=True)
    fig.update_layout(
        title="Cooling Degree Days (CDD) Timeline with Demand Overlay",
        hovermode='x unified',
        height=500,
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        barmode='overlay'
    )

    return fig


def plot_feature_importance(importance_dict, chart_type='bar'):
    """
    Plot feature importance as horizontal bar chart with color coding by category.

    Parameters:
        importance_dict (dict): Feature names and importance scores
        chart_type (str): Type of chart ('bar' or 'horizontal_bar')

    Returns:
        plotly.graph_objects.Figure
    """
    if importance_dict is None or len(importance_dict) == 0:
        return None

    features = list(importance_dict.keys())
    importances = list(importance_dict.values())
    categories = [categorize_feature(feat) for feat in features]

    category_colors = {
        'Temperature': '#ff7f0e',
        'Humidity': '#2ca02c',
        'Calendar': '#9467bd',
        'Weather': '#8c564b',
        'Historical': '#e377c2',
        'Location': '#7f7f7f',
        'Other': '#bcbd22'
    }

    colors = [category_colors.get(cat, '#bcbd22') for cat in categories]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=importances,
        y=features,
        orientation='h',
        marker=dict(color=colors),
        text=[f'{imp:.4f}' for imp in importances],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title="Model Dependency (Relative Influence)",
        xaxis_title="Relative Influence Score",
        yaxis_title="Feature",
        height=max(400, len(features) * 25),
        template='plotly_white',
        yaxis=dict(autorange="reversed")
    )

    return fig


def generate_weather_insight(results_df, df_features, state_name):
    """
    Generate natural language insight about weather impact on demand.

    Parameters:
        results_df: DataFrame with forecasted demand and dates
        df_features: DataFrame with engineered features
        state_name (str): Name of the state

    Returns:
        str: Natural language insight text
    """
    merged_df = pd.merge(results_df, df_features, on='date', how='inner', suffixes=('_results', '_features'))

    if len(merged_df) == 0:
        return "No data available for insight generation."

    if 'cdd_features' in merged_df.columns:
        merged_df['cdd'] = merged_df['cdd_features']
    elif 'cdd_results' in merged_df.columns:
        merged_df['cdd'] = merged_df['cdd_results']
    elif 'cdd' not in merged_df.columns:
        if 'cdd' in results_df.columns:
            merged_df['cdd'] = results_df['cdd'].values
        elif 'cdd' in df_features.columns:
            merged_df['cdd'] = df_features['cdd'].values

    temp_mean_c = merged_df['2m_temperature_mean'] - 273.15
    temp_max_c = merged_df['2m_temperature_max'] - 273.15

    peak_idx = merged_df['forecasted_demand_MU'].idxmax()
    peak_demand = merged_df.loc[peak_idx, 'forecasted_demand_MU']
    peak_date = merged_df.loc[peak_idx, 'date']
    peak_temp = temp_mean_c.loc[peak_idx]
    peak_temp_max = temp_max_c.loc[peak_idx]
    peak_cdd = merged_df.loc[peak_idx, 'cdd'] if 'cdd' in merged_df.columns else 0.0

    temp_demand_corr = merged_df['forecasted_demand_MU'].corr(temp_mean_c)

    if 'cdd' in merged_df.columns:
        cdd_demand_corr = merged_df['forecasted_demand_MU'].corr(merged_df['cdd'])
        avg_cdd = merged_df['cdd'].mean()
        max_cdd = merged_df['cdd'].max()
    else:
        cdd_demand_corr = 0.0
        avg_cdd = 0.0
        max_cdd = 0.0

    if 'extreme_heat_features' in merged_df.columns:
        extreme_heat_count = merged_df['extreme_heat_features'].sum()
    elif 'extreme_heat_results' in merged_df.columns:
        extreme_heat_count = merged_df['extreme_heat_results'].sum()
    elif 'extreme_heat' in merged_df.columns:
        extreme_heat_count = merged_df['extreme_heat'].sum()
    else:
        extreme_heat_count = 0

    total_days = len(merged_df)
    extreme_heat_pct = (extreme_heat_count / total_days * 100) if total_days > 0 else 0
    avg_temp = temp_mean_c.mean()
    max_temp = temp_max_c.max()

    insights = []

    insights.append(
        f"**Peak Demand**: Demand peaks on {peak_date.strftime('%B %d, %Y')} at {peak_temp:.1f}°C "
        f"(max {peak_temp_max:.1f}°C) with CDD of {peak_cdd:.1f}°C, coinciding with elevated demand of {peak_demand:.2f} (normalized units)."
    )

    if abs(temp_demand_corr) > 0.5:
        corr_strength = "strong" if abs(temp_demand_corr) > 0.7 else "moderate"
        corr_direction = "positive" if temp_demand_corr > 0 else "negative"
        insights.append(
            f"**Temperature Impact**: {corr_strength.capitalize()} {corr_direction} correlation ({temp_demand_corr:.2f}) "
            f"between temperature and demand, suggesting temperature influences demand patterns."
        )

    if extreme_heat_count > 0:
        insights.append(
            f"**Extreme Heat Risk**: {extreme_heat_count} extreme heat day(s) identified ({extreme_heat_pct:.0f}% of period), "
            f"with average CDD of {avg_cdd:.1f}°C (max {max_cdd:.1f}°C), potentially amplifying cooling load. "
            f"*Extreme heat defined as days where max temperature exceeds the 95th percentile for {state_name}.*"
        )
    else:
        insights.append(
            f"**Temperature Conditions**: No extreme heat days in this period. Average temperature is {avg_temp:.1f}°C "
            f"(max {max_temp:.1f}°C) with average CDD of {avg_cdd:.1f}°C. "
            f"*Extreme heat threshold: 95th percentile of max temperature for {state_name}.*"
        )

    if cdd_demand_corr > 0.3:
        insights.append(
            f"**Cooling Load**: CDD shows {cdd_demand_corr:.2f} correlation with demand, suggesting that "
            f"cooling degree days influence electricity consumption patterns in {state_name}."
        )

    return " ".join(insights)


def get_seasonal_comparison(state_name, current_season, historical_df=None):
    """
    Get seasonal comparison data for a state.

    Parameters:
        state_name (str): Name of the state
        current_season (str): Current season ('Summer', 'Monsoon', 'Winter', 'Autumn')
        historical_df: Optional historical dataframe (if None, will load)

    Returns:
        dict: Dictionary with seasonal statistics
    """
    if historical_df is None:
        historical_df = load_historical_data(state_name)

    if historical_df is None or len(historical_df) == 0:
        return None

    df_features = engineer_features(historical_df, state_name)

    seasonal_stats = {}
    for season in ['Summer', 'Monsoon', 'Winter', 'Autumn']:
        season_data = df_features[df_features['season'] == season]
        if len(season_data) > 0:
            temp_mean_c = (season_data['2m_temperature_mean'] - 273.15).mean()
            temp_max_c = (season_data['2m_temperature_max'] - 273.15).mean()
            cdd_mean = season_data['cdd'].mean()
            extreme_heat_pct = (season_data['extreme_heat'].sum() / len(season_data) * 100)

            seasonal_stats[season] = {
                'temp_mean': temp_mean_c,
                'temp_max': temp_max_c,
                'cdd_mean': cdd_mean,
                'extreme_heat_pct': extreme_heat_pct,
                'count': len(season_data)
            }

    return seasonal_stats
