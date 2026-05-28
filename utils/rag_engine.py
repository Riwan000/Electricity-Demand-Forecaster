"""
RAG query engine for processing user queries and generating grounded responses.
"""

import os
import json
from typing import Generator, List, Dict, Optional, Tuple
import requests
from utils.embeddings import get_embeddings, get_openrouter_api_key
from utils.vector_store import VectorStore


def load_prompt_template(template_path: str = "prompts/assistant.md") -> str:
    """
    Load prompt template from file.
    
    Parameters:
        template_path: Path to prompt template file
        
    Returns:
        Template string with placeholders
    """
    if not os.path.exists(template_path):
        # Return default template if file doesn't exist
        return get_default_prompt_template()
    
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def get_default_prompt_template() -> str:
    """Get default prompt template if file is not found."""
    return """You are an AI Energy Assistant for electricity demand forecasting in India. Your role is to help users understand forecasts, weather impacts, model performance, and risk factors.

INSTRUCTIONS:
1. Only answer questions based on the retrieved context provided below.
2. If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question based on the available data."
3. Always cite specific metrics, dates, states, or values from the context when providing answers.
4. Explain weather impacts and technical concepts in non-technical terms when possible.
5. If asked about forecasts, always mention the state, date range, and key metrics.
6. Be concise but thorough in your responses.

ACTIVE FORECAST DATA (authoritative — use these exact numbers, do not recompute):
{forecast_context}

RETRIEVED KNOWLEDGE BASE CONTEXT:
{retrieved_context}

CURRENT CONTEXT:
- State: {current_state}
- Forecast Horizon: {forecast_horizon}

USER QUESTION:
{user_query}

RESPONSE:"""


def _format_forecast_context(forecast_summary: Optional[Dict]) -> str:
    """
    Serialize a forecast_summary dict into a concise text block for the prompt.

    Returns an empty marker string when no forecast is available so the
    {forecast_context} placeholder in the template is always filled.
    """
    if not forecast_summary:
        return "No active forecast data."

    state = forecast_summary.get('state', 'Unknown')
    start = forecast_summary.get('start_date', '')
    end = forecast_summary.get('end_date', '')
    horizon = forecast_summary.get('horizon_days', '')
    avg = forecast_summary.get('avg_demand', '')
    peak = forecast_summary.get('peak_demand', '')
    peak_date = forecast_summary.get('peak_date', '')
    min_d = forecast_summary.get('min_demand', '')
    rmse = forecast_summary.get('rmse', '')
    confidence = forecast_summary.get('confidence_level', '')
    high_risk_count = forecast_summary.get('high_risk_days_count', 0)
    high_risk_dates = forecast_summary.get('high_risk_dates', [])
    extreme_heat = forecast_summary.get('extreme_heat_dates', [])

    lines = [
        f"- State: {state}",
        f"- Period: {start} to {end} ({horizon} days)",
        f"- Average daily demand: {avg} MU",
        f"- Peak demand: {peak} MU on {peak_date}",
        f"- Minimum demand: {min_d} MU",
    ]
    if rmse:
        lines.append(f"- Model RMSE: {rmse} MU")
    if confidence:
        lines.append(f"- Confidence level: {confidence}")
    if high_risk_count:
        dates_str = ', '.join(high_risk_dates) if high_risk_dates else 'none listed'
        lines.append(f"- High-risk days: {high_risk_count} ({dates_str})")
    else:
        lines.append("- High-risk days: 0")
    if extreme_heat:
        lines.append(f"- Extreme heat dates: {', '.join(extreme_heat)}")

    return '\n'.join(lines)


def is_query_in_scope(query: str) -> bool:
    """
    Check if query is within scope of the assistant.

    Parameters:
        query: User query string

    Returns:
        True if query is in scope, False otherwise
    """
    query_lower = query.lower()

    # Hard out-of-scope topics
    out_of_scope_keywords = [
        'recipe', 'cook', 'movie', 'sport', 'game', 'music', 'celebrity',
        'stock', 'crypto', 'bitcoin', 'joke', 'poem', 'write a story'
    ]
    if any(keyword in query_lower for keyword in out_of_scope_keywords):
        return False

    return True


def call_openrouter_llm(messages: List[Dict], model: str = "nvidia/nemotron-nano-9b-v2:free") -> str:
    """
    Call OpenRouter API for LLM completion.
    
    Parameters:
        messages: List of message dictionaries with 'role' and 'content'
        model: Model identifier for OpenRouter
        
    Returns:
        Generated response text
        
    Raises:
        ValueError: If API key is not set or API call fails
    """
    api_key = get_openrouter_api_key()
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not found in environment variables. "
            "Please set it in your .env file or environment."
        )
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://electricity-demand-forecaster.streamlit.app",
        "X-Title": "Smart Energy Forecasting Platform"
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        
        if "choices" not in data or len(data["choices"]) == 0:
            raise ValueError(f"Unexpected API response format: {data}")
        
        return data["choices"][0]["message"]["content"]
        
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to call OpenRouter API: {str(e)}")
    except (KeyError, ValueError) as e:
        raise ValueError(f"Error parsing LLM response: {str(e)}")


def stream_openrouter_llm(
    messages: List[Dict],
    model: str = "meta-llama/llama-3.1-8b-instruct:free"
) -> Generator[str, None, None]:
    """
    Call OpenRouter API with SSE streaming, yielding text chunks as they arrive.

    Parameters:
        messages: List of message dicts with 'role' and 'content'
        model: Model identifier for OpenRouter

    Yields:
        Text chunks from the streaming response

    Raises:
        ValueError: If API key is missing or the request fails before streaming begins
    """
    api_key = get_openrouter_api_key()
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not found in environment variables. "
            "Please set it in your .env file or environment."
        )

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://electricity-demand-forecaster.streamlit.app",
        "X-Title": "Smart Energy Forecasting Platform"
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1000,
        "stream": True,
    }

    try:
        with requests.post(url, json=payload, headers=headers, timeout=60, stream=True) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    text = chunk["choices"][0].get("delta", {}).get("content")
                    if text:
                        yield text
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to stream from OpenRouter API: {str(e)}")


def initialize_rag_system() -> 'RAGEngine':
    """Initialize and return a RAG engine instance."""
    try:
        vector_store = VectorStore()
        return RAGEngine(vector_store)
    except Exception as e:
        raise ValueError(f"Failed to initialize RAG system: {str(e)}")


def generate_forecast_summary(
    state: str,
    forecast_values: List[float],
    query: str,
    metadata: Optional[Dict] = None
) -> str:
    """
    Generate a forecast summary response to a user query.

    Parameters:
        state: State name for context filtering
        forecast_values: Predicted demand values
        query: User question
        metadata: Optional model metadata dict

    Returns:
        Response text from the RAG system
    """
    try:
        rag = initialize_rag_system()
        forecast_context = {
            'state': state,
            'avg_demand': sum(forecast_values) / len(forecast_values) if forecast_values else 0,
            'peak_demand': max(forecast_values) if forecast_values else 0,
            'min_demand': min(forecast_values) if forecast_values else 0,
        }
        response, _, _ = rag.query(
            user_query=query,
            current_state=state,
            forecast_horizon=len(forecast_values),
            forecast_context=forecast_context
        )
        return response
    except Exception as e:
        # Graceful error handling for tests
        return f"Error generating forecast summary: {str(e)}"


class RAGEngine:
    """RAG engine for processing queries and generating responses."""

    def __init__(self, vector_store: VectorStore, prompt_template_path: str = "prompts/assistant.md"):
        """
        Initialize RAG engine.
        
        Parameters:
            vector_store: Initialized VectorStore instance
            prompt_template_path: Path to prompt template file
        """
        self.vector_store = vector_store
        self.prompt_template = load_prompt_template(prompt_template_path)
    
    def query(
        self,
        user_query: str,
        current_state: Optional[str] = None,
        forecast_horizon: Optional[int] = None,
        forecast_context: Optional[Dict] = None,
        top_k: int = 5,
        min_similarity: float = 0.3,  # Low threshold to tolerate typos in queries
        model: str = "meta-llama/llama-3.1-8b-instruct:free"
    ) -> Tuple[str, List[Dict], float]:
        """
        Process a user query and generate a grounded response.

        Parameters:
            user_query: User's question
            current_state: Optional state name for context filtering
            forecast_horizon: Optional forecast horizon for context
            forecast_context: Optional forecast_summary dict injected directly into prompt
            top_k: Number of documents to retrieve
            min_similarity: Minimum similarity threshold for retrieval
            model: LLM model identifier for OpenRouter

        Returns:
            Tuple of (response_text, retrieved_contexts, avg_similarity)
        """
        # Check if query is in scope
        if not is_query_in_scope(user_query):
            return (
                "I can only answer questions about electricity demand forecasts, "
                "weather impacts, model performance, and related topics. "
                "Please ask a question related to energy forecasting.",
                [],
                0.0
            )

        # Embed query
        try:
            query_embedding = get_embeddings([user_query])[0]
        except Exception as e:
            return (
                f"Error processing query: {str(e)}",
                [],
                0.0
            )

        # Search vector store (filtered to current state + generic docs)
        results = self.vector_store.search(
            query_embedding,
            k=top_k,
            min_similarity=min_similarity,
            state_filter=current_state,
        )

        if not results:
            return (
                "I don't have forecast data to answer this yet. "
                "Please go to the **Demand Forecast** tab, run a forecast for your state and horizon, "
                "then come back here to ask questions about it.",
                [],
                0.0
            )

        # Extract contexts and metadata
        contexts = []
        similarities = []
        for doc_text, similarity, metadata in results:
            contexts.append({
                'text': doc_text,
                'similarity': similarity,
                'metadata': metadata
            })
            similarities.append(similarity)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

        # Build context string
        context_text = "\n\n".join([
            f"[Context {i+1}]: {ctx['text']}"
            for i, ctx in enumerate(contexts)
        ])

        # Build prompt
        prompt = self.prompt_template.format(
            forecast_context=_format_forecast_context(forecast_context),
            retrieved_context=context_text,
            current_state=current_state or "Not specified",
            forecast_horizon=f"{forecast_horizon} days" if forecast_horizon else "Not specified",
            user_query=user_query,
        )

        # Call LLM
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant for electricity demand forecasting."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = call_openrouter_llm(messages, model=model)
        except Exception as e:
            return (
                f"Error generating response: {str(e)}",
                contexts,
                avg_similarity
            )

        return response, contexts, avg_similarity

    def query_stream(
        self,
        user_query: str,
        current_state: Optional[str] = None,
        forecast_horizon: Optional[int] = None,
        forecast_context: Optional[Dict] = None,
        top_k: int = 5,
        min_similarity: float = 0.3,
        model: str = "meta-llama/llama-3.1-8b-instruct:free",
        chat_history: Optional[List[Dict]] = None,
    ) -> Tuple[List[Dict], float, Generator[str, None, None]]:
        """
        Process a user query and return a streaming text generator.

        Parameters:
            user_query: User's question
            current_state: Optional state name for context
            forecast_horizon: Optional forecast horizon in days
            forecast_context: Optional forecast_summary dict injected directly into prompt
            top_k: Number of documents to retrieve
            min_similarity: Minimum similarity threshold for retrieval
            model: LLM model identifier for OpenRouter
            chat_history: Previous turns as [{role, content}, ...].
                          Pass st.session_state.chat_history[:-1] to exclude
                          the current user message already appended to history.

        Returns:
            Tuple of (retrieved_contexts, avg_similarity, chunk_generator).
            chunk_generator yields str chunks and must be consumed exactly once.
        """
        def _error_gen(msg: str) -> Generator[str, None, None]:
            yield msg

        if not is_query_in_scope(user_query):
            return (
                [],
                0.0,
                _error_gen(
                    "I can only answer questions about electricity demand forecasts, "
                    "weather impacts, model performance, and related topics. "
                    "Please ask a question related to energy forecasting."
                ),
            )

        try:
            query_embedding = get_embeddings([user_query])[0]
        except Exception as e:
            return [], 0.0, _error_gen(f"Error processing query: {str(e)}")

        # Search vector store (filtered to current state + generic docs)
        results = self.vector_store.search(
            query_embedding, k=top_k, min_similarity=min_similarity,
            state_filter=current_state,
        )

        if not results and forecast_context is None:
            return (
                [],
                0.0,
                _error_gen(
                    "I don't have forecast data to answer this yet. "
                    "Please go to the **Demand Forecast** tab, run a forecast for your "
                    "state and horizon, then come back here to ask questions about it."
                ),
            )

        contexts = []
        similarities = []
        for doc_text, similarity, metadata in results:
            contexts.append({"text": doc_text, "similarity": similarity, "metadata": metadata})
            similarities.append(similarity)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

        context_text = "\n\n".join(
            f"[Context {i+1}]: {ctx['text']}" for i, ctx in enumerate(contexts)
        )

        prompt = self.prompt_template.format(
            forecast_context=_format_forecast_context(forecast_context),
            retrieved_context=context_text,
            current_state=current_state or "Not specified",
            forecast_horizon=f"{forecast_horizon} days" if forecast_horizon else "Not specified",
            user_query=user_query,
        )

        # Build messages: system + last 3 history turns + current prompt
        messages: List[Dict] = [
            {"role": "system", "content": "You are a helpful AI assistant for electricity demand forecasting."}
        ]
        if chat_history:
            for turn in chat_history[-3:]:
                if turn.get("role") in ("user", "assistant") and turn.get("content"):
                    messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": prompt})

        try:
            chunk_gen = stream_openrouter_llm(messages, model=model)
        except ValueError as e:
            return contexts, avg_similarity, _error_gen(f"Error generating response: {str(e)}")

        return contexts, avg_similarity, chunk_gen