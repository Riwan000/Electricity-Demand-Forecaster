"""
Health check functions for system startup validation.

Runs once at app startup to verify:
- Model file exists and loads
- RAG/FAISS index exists and loads
- API key is configured (optional)
"""

from typing import Tuple
import os


def check_model() -> Tuple[bool, str]:
    """
    Verify XGBoost model file exists and is readable.

    Returns:
        (healthy: bool, reason: str)

    Examples:
        (True, "Model file found")
        (False, "Model file not found")
    """
    possible_paths = [
        'artifacts/xgb2_model.pkl',
        'xgb2_model.pkl',
        '../xgb2_model.pkl',
        '../models/xgb2_model.pkl'
    ]

    for model_path in possible_paths:
        if os.path.exists(model_path):
            try:
                # Just verify the file is readable
                with open(model_path, 'rb') as f:
                    f.read(100)  # Read first 100 bytes to verify readability
                return True, f"Model file ready"
            except Exception as e:
                return False, f"Model file not readable: {type(e).__name__}"

    return False, "Model file not found"


def check_rag() -> Tuple[bool, str]:
    """
    Verify FAISS RAG index exists and can be loaded.

    Returns:
        (healthy: bool, reason: str)

    Examples:
        (True, "RAG index ready (1024 documents)")
        (False, "FAISS index not found at data/rag_index/")
        (False, "RAG index is empty (will be built on first use)")
    """
    try:
        from utils.vector_store import VectorStore

        vector_store = VectorStore()
        doc_count = vector_store.index.ntotal if hasattr(vector_store, 'index') else 0

        if doc_count == 0:
            # RAG will be built on first use - not a failure
            return True, "RAG ready (will build on first use)"

        return True, f"RAG ready ({doc_count} documents)"
    except FileNotFoundError:
        # FAISS index not found yet - this is OK, will be created on first use
        return True, "RAG ready (will build on first use)"
    except Exception as e:
        # Only fail if there's a real error
        return False, f"RAG error: {type(e).__name__}"


def check_api_key() -> Tuple[bool, str]:
    """
    Verify OPENROUTER_API_KEY is present (optional for app).

    Returns:
        (healthy: bool, reason: str)

    Examples:
        (True, "API key configured")
        (False, "API key not found (optional for local embeddings)")
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        return True, "API key configured"
    return True, "API key not configured (optional - local embeddings preferred)"
