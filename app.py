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
from utils.health_monitor import HealthMonitor
from utils.health_checks import check_model, check_rag, check_api_key
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
# HEALTH CHECKS (Run once at startup)
# ============================================================================
if "health_monitor" not in st.session_state:
    health_monitor = HealthMonitor()
    st.session_state.health_monitor = health_monitor

    model_ok, model_reason = check_model()
    rag_ok, rag_reason = check_rag()
    api_key_ok, api_key_reason = check_api_key()

    health_monitor.set_model_status(model_ok)
    health_monitor.set_rag_status(rag_ok)
else:
    health_monitor = st.session_state.health_monitor

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("---")

    # System Status (new in Phase 5)
    st.subheader("🏥 System Status")
    status = health_monitor.get_status_dict()

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"{status['model']} Model")
    with col2:
        st.write(f"{status['rag']} RAG")

    if status['last_error']:
        st.warning(f"⚠️ Last Error:\n{status['last_error']}")

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
    render_forecast_tab(model, metadata, use_weather_api, health_monitor)

with tab2:
    render_weather_tab(model, metadata, use_weather_api, health_monitor)

with tab3:
    render_assistant_tab(model, metadata, health_monitor)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption("⚡ Smart Energy Forecasting Platform | Stage 1.5: Forecast + Explain")
