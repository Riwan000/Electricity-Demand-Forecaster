# ============================================================================
# RAG FUNCTIONS
# ============================================================================
import os
import streamlit as st

from models import load_model
from visualization import get_feature_importance
from utils.embeddings import get_embeddings
from utils.vector_store import VectorStore
from utils.rag_builder import (
    load_model_metadata as load_model_metadata_rag,
    build_model_metrics_documents,
    build_feature_importance_documents,
    build_forecast_summary_documents
)
from utils.rag_engine import RAGEngine


def generate_forecast_summary(results_df, state, horizon_days, df_features=None, metadata=None):
    """
    Generate structured forecast summary for RAG system.
    Creates a single locked object per forecast with exact values.

    Parameters:
        results_df: DataFrame with forecast results
        state (str): State name
        horizon_days (int): Forecast horizon in days
        df_features: Optional DataFrame with features
        metadata: Optional model metadata for RMSE and confidence_level

    Returns:
        dict: Forecast summary with locked schema, or None
    """
    if results_df is None or len(results_df) == 0:
        return None

    state_rmse = None
    confidence_level = 0.9
    if metadata:
        state_rmse_dict = metadata.get('state_rmse', {})
        state_rmse = state_rmse_dict.get(state)
        if state_rmse is None and state == "Jammu and Kashmir":
            state_rmse = state_rmse_dict.get("J&K")
        confidence_level = metadata.get('confidence_level', 0.9)

    high_risk_dates = []
    extreme_heat_dates = []

    if 'extreme_heat' in results_df.columns:
        extreme_heat_rows = results_df[results_df['extreme_heat'] == 1]
        extreme_heat_dates = [row['date'].strftime('%Y-%m-%d') for _, row in extreme_heat_rows.iterrows()]

        high_demand_threshold = results_df['forecasted_demand_MU'].quantile(0.9)
        high_risk_rows = results_df[
            (results_df['extreme_heat'] == 1) |
            (results_df['forecasted_demand_MU'] >= high_demand_threshold)
        ]
        high_risk_dates = [row['date'].strftime('%Y-%m-%d') for _, row in high_risk_rows.iterrows()]

    summary = {
        'state': state,
        'horizon_days': int(horizon_days),
        'start_date': results_df['date'].min().strftime('%Y-%m-%d'),
        'end_date': results_df['date'].max().strftime('%Y-%m-%d'),
        'avg_demand': round(float(results_df['forecasted_demand_MU'].mean()), 1),
        'peak_demand': round(float(results_df['forecasted_demand_MU'].max()), 1),
        'peak_date': results_df.loc[results_df['forecasted_demand_MU'].idxmax(), 'date'].strftime('%Y-%m-%d'),
        'min_demand': round(float(results_df['forecasted_demand_MU'].min()), 1),
        'high_risk_days_count': len(high_risk_dates),
        'high_risk_dates': high_risk_dates,
        'extreme_heat_dates': extreme_heat_dates,
        'rmse': round(float(state_rmse), 2) if state_rmse is not None else None,
        'confidence_level': confidence_level
    }

    return summary


@st.cache_resource
def initialize_rag_system():
    """
    Initialize RAG system with knowledge base.
    Creates vector store and loads initial knowledge documents.

    Returns:
        Tuple of (VectorStore, RAGEngine) or (None, None) if initialization fails
    """
    try:
        rag_index_dir = "data/rag_index"
        os.makedirs(rag_index_dir, exist_ok=True)
        index_path = os.path.join(rag_index_dir, "knowledge_base.index")

        vector_store = VectorStore(dimension=384, index_path=index_path)

        if vector_store.get_size() > 0:
            rag_engine = RAGEngine(vector_store)
            return vector_store, rag_engine

        try:
            metadata = load_model_metadata_rag("artifacts/model_metadata.json")
            model_metrics_docs = build_model_metrics_documents(metadata)

            model = load_model()
            if model:
                feature_importance = get_feature_importance(model, top_n=15)
                if feature_importance:
                    feature_docs = build_feature_importance_documents(feature_importance)
                    model_metrics_docs.extend(feature_docs)

            if model_metrics_docs:
                texts = [doc['text'] for doc in model_metrics_docs]
                metadata_list = [doc['metadata'] for doc in model_metrics_docs]

                try:
                    embeddings = get_embeddings(texts)
                    vector_store.add_documents(embeddings, texts, metadata_list)
                    vector_store.save_index()
                except Exception as e:
                    st.warning(f"Could not initialize embeddings: {str(e)}")
                    return None, None

            rag_engine = RAGEngine(vector_store)
            return vector_store, rag_engine

        except FileNotFoundError:
            st.warning("Model metadata file not found. RAG system will work with forecast data only.")
            rag_engine = RAGEngine(vector_store)
            return vector_store, rag_engine
        except Exception as e:
            st.error(f"Error building knowledge base: {str(e)}")
            return None, None

    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return None, None
