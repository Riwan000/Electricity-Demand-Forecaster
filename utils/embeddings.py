"""
Embedding service for RAG system.
Supports both OpenRouter API and local sentence-transformers as fallback.
"""

import os
import numpy as np
from typing import List, Optional
import requests
from functools import lru_cache

# Try to import sentence-transformers for local embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None


def get_openrouter_api_key() -> Optional[str]:
    """Get OpenRouter API key from environment variable."""
    return os.getenv("OPENROUTER_API_KEY")


# Global model cache for sentence-transformers
_embedding_model = None

def _get_local_embedding_model():
    """Get or initialize local embedding model."""
    global _embedding_model
    if _embedding_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
        # Use a lightweight, fast model that works well for RAG
        # all-MiniLM-L6-v2 is 384 dimensions, fast and accurate
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model

def get_embeddings(texts: List[str], model: str = "openai/text-embedding-3-small", use_local: bool = True) -> np.ndarray:
    """
    Get embeddings for a list of texts using OpenRouter API.
    
    Parameters:
        texts: List of text strings to embed
        model: Model identifier for OpenRouter (default: OpenAI text-embedding-3-small)
        
    Returns:
        numpy array of shape (n_texts, embedding_dim) with embeddings
        
    Raises:
        ValueError: If API key is not set or API call fails
    """
    """
    Get embeddings for a list of texts.
    
    Parameters:
        texts: List of text strings to embed
        model: Model identifier (for OpenRouter API, ignored for local)
        use_local: If True, use local sentence-transformers (default: True)
                   If False and API fails, will fallback to local anyway
        
    Returns:
        numpy array of shape (n_texts, embedding_dim) with embeddings
    """
    if not texts:
        return np.array([])
    
    # Try local embeddings first if available and use_local is True
    if use_local and SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            embedding_model = _get_local_embedding_model()
            embeddings = embedding_model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            # Fall through to API attempt
            pass
    
    # Try OpenRouter API if local not available or failed
    api_key = get_openrouter_api_key()
    
    if not api_key:
        # If no API key and local not available, raise error
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ValueError(
                "No embedding method available. Either set OPENROUTER_API_KEY or install sentence-transformers: pip install sentence-transformers"
            )
        # Otherwise, we already tried local above, so this shouldn't happen
        raise ValueError("Unexpected state: local embeddings should have been used")
    
    # OpenRouter API endpoint for embeddings
    url = "https://openrouter.ai/api/v1/embeddings"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/your-repo",
        "X-Title": "Smart Energy Forecasting Platform"
    }
    
    payload = {
        "model": model,
        "input": texts
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if "data" not in data:
            raise ValueError(f"Unexpected API response format: {data}")
        
        embedding_items = data["data"]
        if all("index" in item for item in embedding_items):
            embedding_items = sorted(embedding_items, key=lambda x: x.get("index", 0))
        
        embeddings = [item["embedding"] for item in embedding_items]
        return np.array(embeddings)
        
    except requests.exceptions.HTTPError as e:
        # If API fails (e.g., 402 Payment Required), fallback to local if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            embedding_model = _get_local_embedding_model()
            embeddings = embedding_model.encode(texts, convert_to_numpy=True)
            return embeddings
        else:
            raise ValueError(f"Failed to get embeddings from OpenRouter API: {str(e)}. Install sentence-transformers for local fallback: pip install sentence-transformers")
    
    except requests.exceptions.RequestException as e:
        # Fallback to local if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            embedding_model = _get_local_embedding_model()
            embeddings = embedding_model.encode(texts, convert_to_numpy=True)
            return embeddings
        else:
            raise ValueError(f"Failed to get embeddings from OpenRouter API: {str(e)}")
    
    except (KeyError, ValueError) as e:
        raise ValueError(f"Error parsing embeddings response: {str(e)}")


@lru_cache(maxsize=100)
def get_embedding_cached(text: str, model: str = "openai/text-embedding-3-small") -> np.ndarray:
    """
    Get embedding for a single text with caching.
    Useful for frequently accessed knowledge base documents.
    
    Parameters:
        text: Text string to embed
        model: Model identifier for OpenRouter
        
    Returns:
        numpy array of shape (embedding_dim,) with embedding
    """
    embeddings = get_embeddings([text], model=model)
    return embeddings[0] if len(embeddings) > 0 else np.array([])