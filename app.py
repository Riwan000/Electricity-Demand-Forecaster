"""
Smart Energy Forecasting Platform - Streamlit Dashboard
Stage 1.5: Forecast + Explain

Entry point. All logic lives in the modules below:
  config.py          - Constants (STATE_COORDINATES, ALL_STATES)
  models.py          - Model loading, metadata, RMSE lookup
  data_loading.py    - Historical data, baselines, denormalization
  weather.py         - Open-Meteo API + climatology fallback
  features.py        - Feature engineering + prediction prep
  visualization.py   - All Plotly charts + weather insights
  rag.py             - Forecast summary builder + RAG init
  ui/forecast_tab.py - Tab 1: Demand Forecast
  ui/weather_tab.py  - Tab 2: Weather Impact Analysis
  ui/assistant_tab.py- Tab 3: AI Energy Assistant
"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# PAGE CONFIGURATION  (must be first Streamlit call)
# ============================================================================
st.set_page_config(
    page_title="Smart Energy Forecasting Platform",
    page_icon="⚡",
    layout="wide"
)

from models import load_model, load_model_metadata
from utils.embeddings import get_openrouter_api_key
from ui.forecast_tab import render_forecast_tab
from ui.weather_tab import render_weather_tab
from ui.assistant_tab import render_assistant_tab

# ============================================================================
# HEADER
# ============================================================================
st.title("⚡ Smart Energy Forecasting Platform")
st.markdown("**Stage 1.5: Forecast + Explain**")
st.markdown("Interactive dashboard for electricity demand forecasting using ML and real-time weather data")

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("---")

    model = load_model()
    metadata = load_model_metadata()

    if model:
        st.success("✅ Model loaded successfully")
        st.caption("XGBoost model ready for predictions")
    else:
        st.error("❌ Model not found")
        st.caption("Please check model path")

    st.markdown("---")

    st.subheader("🌤️ Weather Data Source")
    use_weather_api = st.checkbox(
        "Use Weather API (Open-Meteo)",
        value=True,
        help="Use real-time weather API. If unchecked, uses historical averages."
    )

    st.markdown("---")
    st.caption("💡 Tip: Weather API provides more accurate forecasts but requires internet connection.")

    st.markdown("---")
    st.subheader("🤖 AI Assistant")

    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        st.success("✅ Local embeddings available")
        st.caption("Using sentence-transformers (no API needed)")
    except ImportError:
        st.warning("⚠️ sentence-transformers not installed")
        st.caption("Install for free local embeddings: `pip install sentence-transformers`")
        with st.expander("Installation Instructions"):
            st.code("pip install sentence-transformers", language="bash")
            st.markdown("""
            **Why install this?**
            - Free local embeddings (no API costs)
            - Works offline
            - Faster than API calls
            - First download is ~90MB, then cached locally
            """)

    api_key = get_openrouter_api_key()
    if api_key:
        st.info("ℹ️ API key configured (optional - local embeddings preferred)")
    else:
        st.caption("💡 Tip: Local embeddings work without an API key")

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3 = st.tabs([
    "📊 Forecast",
    "🌡️ Weather Impact",
    "🤖 AI Energy Assistant"
])

with tab1:
    render_forecast_tab(model, metadata, use_weather_api)

with tab2:
    render_weather_tab(model, metadata, use_weather_api)

with tab3:
    render_assistant_tab(model, metadata)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption("⚡ Smart Energy Forecasting Platform | Stage 1.5: Forecast + Explain")
