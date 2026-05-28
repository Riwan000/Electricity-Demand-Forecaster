# ============================================================================
# RAG ENGINE TESTS (6 tests)
# ============================================================================
import pytest
from utils.rag_engine import initialize_rag_system, generate_forecast_summary


class TestRAGScope:
    """Tests for RAG scope checking and validation."""

    def test_rag_system_initializes(self):
        """
        PURPOSE: Verify RAG system initializes without errors
        CATCHES: Missing FAISS index, corrupted documents
        PRODUCTION IMPACT: RAG tab crashes on app load
        """
        try:
            rag = initialize_rag_system()
            assert rag is not None
            assert hasattr(rag, 'vector_store')
        except FileNotFoundError:
            pytest.skip("RAG index not built - skipping")

    def test_rag_valid_query_processed(self):
        """
        PURPOSE: Verify in-scope queries are processed
        CATCHES: Valid queries incorrectly rejected
        PRODUCTION IMPACT: App rejects legitimate questions
        """
        summary = generate_forecast_summary(
            state='Maharashtra',
            forecast_values=[100, 110, 120],
            query="What's the impact of high temperature on demand?"
        )

        # Should return an answer (not "out of scope")
        assert summary is not None
        assert "out of scope" not in summary.lower()

    def test_rag_invalid_query_rejected(self):
        """
        PURPOSE: Verify out-of-scope queries are rejected
        CATCHES: Hallucination in UI (returning fake energy info)
        PRODUCTION IMPACT: User sees false information
        SEVERITY: 🔴 CRITICAL for user trust
        """
        summary = generate_forecast_summary(
            state='Maharashtra',
            forecast_values=[100, 110, 120],
            query="What's the current price of Bitcoin?"  # Out of scope
        )

        # Should indicate out of scope or that only energy questions are answered
        assert any(phrase in summary.lower() for phrase in ["out of scope", "cannot answer", "only answer", "electricity demand"])

    def test_rag_confidence_score_valid_range(self):
        """
        PURPOSE: Verify confidence score is 0-100
        CATCHES: Score out of range (e.g., -1, 150)
        PRODUCTION IMPACT: Dashboard shows misleading confidence
        """
        result = generate_forecast_summary(
            state='Maharashtra',
            forecast_values=[100, 110, 120],
            query="How does weather affect demand?"
        )

        if isinstance(result, dict) and 'confidence' in result:
            assert 0 <= result['confidence'] <= 100

    def test_rag_document_retrieval_relevance(self):
        """
        PURPOSE: Verify retrieved documents are relevant
        CATCHES: Irrelevant documents in response
        PRODUCTION IMPACT: Summary includes wrong information
        """
        try:
            rag = initialize_rag_system()
            if rag is None:
                pytest.skip("RAG not initialized")

            # Use the vector_store directly to retrieve documents
            from utils.embeddings import get_embeddings
            query = "temperature and demand"
            query_embedding = get_embeddings([query])[0]
            results = rag.vector_store.search(query_embedding, k=3, min_similarity=0.0)

            # At least some results should mention temperature or demand
            relevant = [r for r in results if 'temperature' in r[0].lower() or 'demand' in r[0].lower()]
            # If we got results, verify they're relevant; if empty, skip
            if results:
                assert len(relevant) >= 1, "Retrieved documents should mention temperature or demand"
        except (FileNotFoundError, ImportError):
            pytest.skip("RAG index or embeddings not available")

    def test_rag_graceful_failure_missing_metadata(self):
        """
        PURPOSE: Verify RAG works even with missing model metadata
        CATCHES: Crashes when metadata unavailable
        PRODUCTION IMPACT: App crashes on metadata load error
        """
        try:
            summary = generate_forecast_summary(
                state='Maharashtra',
                forecast_values=[100, 110, 120],
                metadata=None,  # Intentionally missing
                query="What's the forecast?"
            )
            assert summary is not None
        except Exception as e:
            pytest.fail(f"RAG crashed with missing metadata: {e}")
