"""
RAG query engine for processing user queries and generating grounded responses.
"""

import os
import json
from typing import List, Dict, Optional, Tuple
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

CONTEXT:
{retrieved_context}

CURRENT CONTEXT:
- State: {current_state}
- Forecast Horizon: {forecast_horizon}

USER QUESTION:
{user_query}

RESPONSE:"""


def is_query_in_scope(query: str) -> bool:
    """
    Check if query is within scope of the assistant.
    
    Parameters:
        query: User query string
        
    Returns:
        True if query is in scope, False otherwise
    """
    query_lower = query.lower()
    
    # Scope keywords
    in_scope_keywords = [
        'forecast', 'demand', 'electricity', 'energy', 'weather', 'temperature',
        'model', 'prediction', 'risk', 'confidence', 'rmse', 'mae', 'error',
        'feature', 'importance', 'diagnostic', 'state', 'horizon', 'heat',
        'holiday', 'cdd', 'cooling', 'peak', 'average'
    ]
    
    # Check if query contains any in-scope keywords
    return any(keyword in query_lower for keyword in in_scope_keywords)


def call_openrouter_llm(messages: List[Dict], model: str = "openai/gpt-4o") -> str:
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
        "HTTP-Referer": "https://github.com/your-repo",
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
        top_k: int = 5,
        min_similarity: float = 0.5,  # Lowered from 0.7 to allow more results
        model: str = "xiaomi/mimo-v2-flash:free"
    ) -> Tuple[str, List[Dict], float]:
        """
        Process a user query and generate a grounded response.
        
        Parameters:
            user_query: User's question
            current_state: Optional state name for context filtering
            forecast_horizon: Optional forecast horizon for context
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
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding,
            k=top_k,
            min_similarity=min_similarity
        )
        
        if not results:
            return (
                "I don't have enough information to answer this question based on the available data. "
                "Please try asking about forecasts, model performance, or weather impacts.",
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
            retrieved_context=context_text,
            current_state=current_state or "Not specified",
            forecast_horizon=f"{forecast_horizon} days" if forecast_horizon else "Not specified",
            user_query=user_query
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