# ============================================================================
# TAB 3: AI ENERGY ASSISTANT
# ============================================================================
import streamlit as st

from models import load_model_metadata
from rag import initialize_rag_system, generate_forecast_summary
from utils.embeddings import get_embeddings, get_openrouter_api_key
from utils.rag_builder import build_forecast_summary_documents


def render_assistant_tab(model, metadata):
    """
    Render the AI Energy Assistant tab UI.

    Parameters:
        model: Loaded XGBoost model
        metadata (dict): Model metadata
    """
    st.header("🤖 AI Energy Assistant")
    st.markdown("Ask natural language questions about forecasts, weather impacts, model performance, and risk factors.")

    # RAG initialization
    rag_enabled = True
    vector_store = None
    rag_engine = None

    try:
        vector_store, rag_engine = initialize_rag_system()
        if vector_store is None or rag_engine is None:
            rag_enabled = False
    except (FileNotFoundError, ImportError, Exception):
        rag_enabled = False

    if not rag_enabled or vector_store is None or rag_engine is None:
        st.warning("⚠️ AI Assistant is running in limited mode (knowledge base unavailable).")
        st.info("""
        **What this means:**
        - The knowledge base could not be loaded (missing files, path issues, or dependencies)
        - You can still ask questions, but responses will be limited
        - To enable full functionality:
          1. Ensure sentence-transformers is installed: `pip install sentence-transformers`
          2. Check that `model_metadata.json` exists in the project root
          3. Restart the application
        """)
        rag_enabled = False

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Get context from last forecast
    forecast_context = None
    forecast_state = None
    forecast_horizon = None

    if 'last_forecast' in st.session_state:
        forecast_data = st.session_state['last_forecast']
        forecast_state = forecast_data.get('state')
        forecast_horizon = forecast_data.get('horizon_days')

        forecast_summary = forecast_data.get('forecast_summary')
        results_df = forecast_data.get('results_df')

        if forecast_summary is None and results_df is not None and len(results_df) > 0:
            metadata = load_model_metadata()
            forecast_summary = generate_forecast_summary(
                results_df,
                forecast_state,
                forecast_horizon,
                forecast_data.get('df_features'),
                metadata=metadata
            )

        if rag_enabled and forecast_summary and vector_store is not None:
            forecast_id = f"forecast_{forecast_state}_{forecast_summary['start_date']}"
            if 'last_forecast_id' not in st.session_state or st.session_state['last_forecast_id'] != forecast_id:
                forecast_docs = build_forecast_summary_documents(forecast_summary)
                if forecast_docs:
                    try:
                        # Remove stale forecast docs from previous runs before adding new ones
                        vector_store.remove_documents_by_type('forecast')
                        texts = [doc['text'] for doc in forecast_docs]
                        metadata_list = [doc['metadata'] for doc in forecast_docs]
                        embeddings = get_embeddings(texts)
                        vector_store.add_documents(embeddings, texts, metadata_list)
                        vector_store.save_index()
                        st.session_state['last_forecast_id'] = forecast_id
                    except Exception as e:
                        st.warning(f"Could not add forecast to knowledge base: {str(e)}")

    # Example questions
    with st.expander("💡 Example Questions", expanded=False):
        st.markdown("""
        **Forecast Questions:**
        - "What is the forecast for [state]?"
        - "When is peak demand expected?"
        - "Are there any high-risk days in the forecast?"

        **Weather Impact:**
        - "How does temperature affect demand?"
        - "What is the impact of extreme heat on electricity demand?"
        - "Which weather variables are most important?"

        **Model Performance:**
        - "What is the model's RMSE for [state]?"
        - "How accurate is the model?"
        - "What are the top features for forecasting?"

        **Risk & Diagnostics:**
        - "What are the confidence intervals?"
        - "Why might the forecast be uncertain?"
        """)

    if forecast_state:
        st.info(f"📊 Current context: {forecast_state} ({forecast_horizon} days forecast available)")

    # Chat interface
    st.markdown("### 💬 Chat")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources", expanded=False):
                    for j, source in enumerate(message["sources"][:3], 1):
                        st.caption(f"Source {j}: {source['text'][:200]}...")
                        if 'metadata' in source:
                            meta = source['metadata']
                            if 'state' in meta:
                                st.caption(f"  State: {meta['state']}")
                            if 'type' in meta:
                                st.caption(f"  Type: {meta['type']}")

            if "avg_similarity" in message:
                similarity_pct = message["avg_similarity"] * 100
                similarity_text = f"Retrieval similarity: {similarity_pct:.0f}% (average cosine similarity of retrieved sources)"
                if similarity_pct >= 80:
                    st.success(similarity_text)
                elif similarity_pct >= 60:
                    st.warning(similarity_text)
                else:
                    st.info(similarity_text)

    user_query = st.chat_input("Ask a question about forecasts, weather, or model performance...")

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            if not rag_enabled or rag_engine is None:
                limited_msg = (
                    "**Limited Mode Active**\n\n"
                    "The AI Assistant is running in limited mode because the knowledge base is unavailable.\n\n"
                    "To answer your question properly, please:\n"
                    "1. Ensure sentence-transformers is installed: `pip install sentence-transformers`\n"
                    "2. Check that model_metadata.json exists\n"
                    "3. Restart the application"
                )
                st.info(limited_msg)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "AI Assistant is in limited mode. Please check the setup instructions above."
                })
            else:
                try:
                    # Pass history excluding the current user message (already appended above)
                    history_for_llm = st.session_state.chat_history[:-1]
                    contexts, avg_similarity, chunk_gen = rag_engine.query_stream(
                        user_query,
                        current_state=forecast_state,
                        forecast_horizon=forecast_horizon,
                        top_k=5,
                        min_similarity=0.3,
                        chat_history=history_for_llm,
                    )
                    # Renders chunks live as they arrive; returns the full concatenated string
                    response = st.write_stream(chunk_gen)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response,
                        "sources": contexts,
                        "avg_similarity": avg_similarity,
                    })
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg
                    })

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
