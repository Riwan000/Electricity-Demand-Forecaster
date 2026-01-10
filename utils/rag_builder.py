"""
Knowledge base builder for RAG system.
Creates structured knowledge documents from model metadata, feature importance, and diagnostics.
"""

import json
import os
from typing import List, Dict, Optional
from pathlib import Path


def load_model_metadata(metadata_path: str = "model_metadata.json") -> Dict:
    """
    Load model metadata from JSON file.
    
    Parameters:
        metadata_path: Path to model_metadata.json
        
    Returns:
        Dictionary containing model metadata
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Model metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        return json.load(f)


def build_model_metrics_documents(metadata: Dict) -> List[Dict]:
    """
    Build knowledge documents from model performance metrics.
    
    Parameters:
        metadata: Model metadata dictionary
        
    Returns:
        List of document dictionaries with 'text' and 'metadata' keys
    """
    documents = []
    
    # Overall model metrics
    overall_text = (
        f"Model Performance: The electricity demand forecasting model (XGB2_Log) "
        f"has overall RMSE of {metadata.get('rmse', 'N/A'):.2f} MU, "
        f"MAE of {metadata.get('mae', 'N/A'):.2f} MU, and R² of {metadata.get('r2', 'N/A'):.4f}. "
        f"The model was trained on data from {metadata.get('trained_on', 'N/A')}. "
        f"Confidence level is set to {metadata.get('confidence_level', 0.9) * 100:.0f}%."
    )
    documents.append({
        'text': overall_text,
        'metadata': {
            'type': 'model_metrics',
            'scope': 'overall',
            'rmse': metadata.get('rmse'),
            'mae': metadata.get('mae'),
            'r2': metadata.get('r2')
        }
    })
    
    # State-specific metrics
    state_rmse = metadata.get('state_rmse', {})
    for state, rmse in state_rmse.items():
        state_text = (
            f"State Performance - {state}: The model has RMSE of {rmse:.2f} MU for {state}. "
            f"This indicates the typical prediction error for electricity demand forecasts in {state}. "
            f"Lower RMSE values indicate better model performance for this state."
        )
        documents.append({
            'text': state_text,
            'metadata': {
                'type': 'model_metrics',
                'scope': 'state',
                'state': state,
                'rmse': rmse
            }
        })
    
    # Summary statistics
    summary = metadata.get('state_rmse_summary', {})
    if summary:
        summary_text = (
            f"Model Performance Summary: The model was successfully trained for {summary.get('total_states', 0)} states. "
            f"Average RMSE across all states is {summary.get('mean_rmse', 0):.2f} MU. "
            f"Best performing state has RMSE of {summary.get('min_rmse', 0):.2f} MU, "
            f"while the most challenging state has RMSE of {summary.get('max_rmse', 0):.2f} MU."
        )
        documents.append({
            'text': summary_text,
            'metadata': {
                'type': 'model_metrics',
                'scope': 'summary',
                'mean_rmse': summary.get('mean_rmse'),
                'min_rmse': summary.get('min_rmse'),
                'max_rmse': summary.get('max_rmse')
            }
        })
    
    return documents


def build_feature_importance_documents(feature_importance: Dict, top_n: int = 10) -> List[Dict]:
    """
    Build knowledge documents from feature importance data.
    
    Parameters:
        feature_importance: Dictionary mapping feature names to importance scores
        top_n: Number of top features to include in documents
        
    Returns:
        List of document dictionaries
    """
    documents = []
    
    if not feature_importance:
        return documents
    
    # Sort features by importance
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    
    # Create document for top features
    top_features_text = "Top Features for Electricity Demand Forecasting: "
    feature_descriptions = []
    
    for i, (feature, importance) in enumerate(sorted_features, 1):
        # Categorize feature
        category = categorize_feature_simple(feature)
        feature_descriptions.append(
            f"{i}. {feature} ({category}) with relative importance of {importance:.4f}"
        )
    
    top_features_text += "; ".join(feature_descriptions) + ". "
    top_features_text += (
        "These features are the most influential in predicting electricity demand. "
        "Higher importance values indicate stronger influence on forecast predictions."
    )
    
    documents.append({
        'text': top_features_text,
        'metadata': {
            'type': 'feature_importance',
            'top_n': top_n,
            'features': [f[0] for f in sorted_features]
        }
    })
    
    # Create individual documents for top 5 features
    for i, (feature, importance) in enumerate(sorted_features[:5], 1):
        category = categorize_feature_simple(feature)
        feature_text = (
            f"Feature: {feature} is a {category} feature with importance score of {importance:.4f}, "
            f"making it one of the top {i} most important features for electricity demand forecasting. "
            f"This feature significantly influences the model's predictions."
        )
        documents.append({
            'text': feature_text,
            'metadata': {
                'type': 'feature_importance',
                'feature': feature,
                'importance': importance,
                'category': category,
                'rank': i
            }
        })
    
    return documents


def categorize_feature_simple(feature_name: str) -> str:
    """
    Categorize a feature name into a simple category.
    
    Parameters:
        feature_name: Name of the feature
        
    Returns:
        Category string
    """
    feature_lower = feature_name.lower()
    
    if 'temperature' in feature_lower or 'temp' in feature_lower:
        return 'Temperature'
    elif 'humidity' in feature_lower or 'dewpoint' in feature_lower or 'utci' in feature_lower:
        return 'Humidity'
    elif 'wind' in feature_lower:
        return 'Wind'
    elif 'solar' in feature_lower or 'radiation' in feature_lower:
        return 'Solar'
    elif 'cloud' in feature_lower:
        return 'Cloud Cover'
    elif 'holiday' in feature_lower or 'weekend' in feature_lower or 'month' in feature_lower or 'day' in feature_lower:
        return 'Calendar'
    elif 'demand' in feature_lower or 'energy' in feature_lower or 'generation' in feature_lower:
        return 'Historical Demand'
    elif 'heat' in feature_lower or 'cdd' in feature_lower:
        return 'Heat/Cooling'
    else:
        return 'Other'


def build_diagnostic_documents(diagnostic_summary: str, state: Optional[str] = None) -> List[Dict]:
    """
    Build knowledge documents from diagnostic summaries.
    
    Parameters:
        diagnostic_summary: Text summary of model diagnostics
        state: Optional state name for context
        
    Returns:
        List of document dictionaries
    """
    documents = []
    
    if not diagnostic_summary:
        return documents
    
    # Split summary into sentences and create structured documents
    # This is a simple approach - could be enhanced with NLP parsing
    diagnostic_text = (
        f"Model Diagnostics{f' for {state}' if state else ''}: {diagnostic_summary} "
        f"These diagnostics help understand model performance, identify risky periods, "
        f"and explain forecast confidence levels."
    )
    
    documents.append({
        'text': diagnostic_text,
        'metadata': {
            'type': 'diagnostics',
            'state': state,
            'summary': diagnostic_summary
        }
    })
    
    return documents


def build_forecast_summary_documents(forecast_summary: Dict) -> List[Dict]:
    """
    Build knowledge documents from forecast summary.
    Creates a single locked document per forecast with exact schema - no recomputation allowed.
    
    Parameters:
        forecast_summary: Dictionary containing forecast summary data with locked schema
        
    Returns:
        List containing a single document dictionary with the complete forecast object
    """
    documents = []
    
    if not forecast_summary:
        return documents
    
    # Create a single comprehensive document with the complete forecast object
    # Format as JSON-like structure for the LLM to parse exactly
    state = forecast_summary.get('state', 'Unknown')
    horizon = forecast_summary.get('horizon_days', 0)
    start_date = forecast_summary.get('start_date', '')
    end_date = forecast_summary.get('end_date', '')
    avg_demand = forecast_summary.get('avg_demand', 0)
    peak_demand = forecast_summary.get('peak_demand', 0)
    peak_date = forecast_summary.get('peak_date', '')
    min_demand = forecast_summary.get('min_demand', 0)
    high_risk_days_count = forecast_summary.get('high_risk_days_count', 0)
    high_risk_dates = forecast_summary.get('high_risk_dates', [])
    extreme_heat_dates = forecast_summary.get('extreme_heat_dates', [])
    rmse = forecast_summary.get('rmse')
    confidence_level = forecast_summary.get('confidence_level', 0.9)
    
    # Build comprehensive forecast text with exact values
    forecast_text = (
        f"FORECAST SUMMARY FOR {state.upper()}:\n"
        f"Forecast period: {start_date} to {end_date} ({horizon} days)\n"
        f"Average daily demand: {avg_demand} MU\n"
        f"Peak demand: {peak_demand} MU on {peak_date}\n"
        f"Minimum demand: {min_demand} MU\n"
    )
    
    if rmse is not None:
        forecast_text += f"Model RMSE for {state}: {rmse} MU\n"
    
    forecast_text += f"Confidence level: {confidence_level}\n"
    
    if high_risk_days_count > 0:
        forecast_text += f"High-risk days count: {high_risk_days_count}\n"
        if high_risk_dates:
            forecast_text += f"High-risk dates: {', '.join(high_risk_dates)}\n"
    else:
        forecast_text += "High-risk days count: 0\n"
        forecast_text += "High-risk dates: []\n"
    
    if extreme_heat_dates:
        forecast_text += f"Extreme heat dates: {', '.join(extreme_heat_dates)}\n"
    else:
        forecast_text += "Extreme heat dates: []\n"
    
    # Add structured JSON representation for precise parsing
    import json
    forecast_json = json.dumps(forecast_summary, indent=2)
    forecast_text += f"\nSTRUCTURED FORECAST DATA:\n{forecast_json}"
    
    documents.append({
        'text': forecast_text,
        'metadata': {
            'type': 'forecast',
            'state': state,
            'horizon_days': horizon,
            'start_date': start_date,
            'end_date': end_date,
            'forecast_summary': forecast_summary  # Include full object in metadata
        }
    })
    
    return documents