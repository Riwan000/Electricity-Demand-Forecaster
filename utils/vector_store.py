"""
FAISS-based vector store for RAG system.
Handles document storage, indexing, and semantic search.
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple
import faiss


class VectorStore:
    """FAISS-based vector store for semantic search."""
    
    def __init__(self, dimension: int = 384, index_path: Optional[str] = None):
        """
        Initialize vector store.
        
        Parameters:
            dimension: Embedding dimension (384 for all-MiniLM-L6-v2, the default local model)
            index_path: Path to save/load FAISS index
        """
        self.dimension = dimension
        self.index_path = index_path
        self.index = None
        self.documents = []  # Store original documents for retrieval
        self.metadata = []  # Store metadata for each document
        self._embeddings: List[np.ndarray] = []  # Cached post-normalization embeddings

        # Initialize or load index
        if index_path and os.path.exists(index_path):
            self.load_index()
        else:
            self.create_index()
    
    def create_index(self):
        """Create a new FAISS index."""
        # Use L2 distance (Euclidean) - common for embeddings
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.metadata = []
        self._embeddings = []
    
    def add_documents(self, embeddings: np.ndarray, documents: List[str], metadata: Optional[List[Dict]] = None):
        """
        Add documents to the vector store.
        
        Parameters:
            embeddings: numpy array of shape (n_docs, dimension)
            documents: List of document text strings
            metadata: Optional list of metadata dictionaries for each document
        """
        if self.index is None:
            self.create_index()
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, "
                f"got {embeddings.shape[1]}"
            )
        
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Number of documents ({len(documents)}) doesn't match "
                f"number of embeddings ({len(embeddings)})"
            )
        
        # Normalize embeddings for cosine similarity (L2 normalization)
        # FAISS IndexFlatL2 works with normalized vectors for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype('float32'))

        # Store documents, metadata, and cached embeddings (already normalized above)
        self.documents.extend(documents)
        self._embeddings.extend([embeddings[i].copy() for i in range(len(embeddings))])
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))
    
    def search(self, query_embedding: np.ndarray, k: int = 5, min_similarity: float = 0.7) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar documents.
        
        Parameters:
            query_embedding: Query embedding vector of shape (dimension,)
            k: Number of results to return
            min_similarity: Minimum similarity score (0-1, converted from distance)
            
        Returns:
            List of tuples: (document_text, similarity_score, metadata)
            Results are sorted by similarity (highest first)
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        if query_embedding.shape[0] != self.dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.dimension}, "
                f"got {query_embedding.shape[0]}"
            )
        
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        # Convert distances to similarity scores (1 - normalized distance)
        # For L2 normalized vectors, distance ranges from 0 to 2
        # Similarity = 1 - (distance / 2)
        similarities = 1 - (distances[0] / 2.0)
        
        # Filter by minimum similarity and format results
        results = []
        for idx, sim in zip(indices[0], similarities):
            if sim >= min_similarity and idx < len(self.documents):
                results.append((
                    self.documents[idx],
                    float(sim),
                    self.metadata[idx]
                ))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def save_index(self):
        """Save the FAISS index and associated data to disk."""
        if self.index_path is None:
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        
        # Save documents, metadata, and cached embeddings
        data_path = self.index_path.replace('.index', '_data.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata,
                'dimension': self.dimension,
                'embeddings': self._embeddings,
            }, f)
    
    def load_index(self):
        """Load the FAISS index and associated data from disk."""
        if self.index_path is None or not os.path.exists(self.index_path):
            self.create_index()
            return
        
        # Load FAISS index
        self.index = faiss.read_index(self.index_path)
        
        # Load documents and metadata
        data_path = self.index_path.replace('.index', '_data.pkl')
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data.get('documents', [])
                self.metadata = data.get('metadata', [])
                self.dimension = data.get('dimension', self.dimension)
                # 'embeddings' key absent in indexes saved before this version;
                # the fallback guard in remove_documents_by_type handles that case.
                self._embeddings = data.get('embeddings', [])
    
    def remove_documents_by_type(self, doc_type: str):
        """
        Remove all documents of a given metadata type and rebuild the index
        using cached embeddings — no API calls required.

        Parameters:
            doc_type: Value of the 'type' key in metadata to remove (e.g. 'forecast')
        """
        keep_indices = [
            i for i, meta in enumerate(self.metadata)
            if meta.get('type') != doc_type
        ]

        if not keep_indices:
            self.create_index()
            return

        # Guard: if _embeddings cache is out of sync (e.g. index loaded from disk
        # before this version), fall back to re-embedding once to warm the cache.
        if len(self._embeddings) != len(self.documents):
            from utils.embeddings import get_embeddings as _ge
            self.documents = [self.documents[i] for i in keep_indices]
            self.metadata = [self.metadata[i] for i in keep_indices]
            self.index = faiss.IndexFlatL2(self.dimension)
            embs = _ge(self.documents).astype('float32')
            faiss.normalize_L2(embs)
            self.index.add(embs)
            self._embeddings = [embs[i].copy() for i in range(len(embs))]
            return

        # Fast path: rebuild from cached embeddings — zero API calls
        self.documents = [self.documents[i] for i in keep_indices]
        self.metadata = [self.metadata[i] for i in keep_indices]
        self._embeddings = [self._embeddings[i] for i in keep_indices]
        self.index = faiss.IndexFlatL2(self.dimension)
        # Cached embeddings are already L2-normalized; add directly
        self.index.add(np.stack(self._embeddings).astype('float32'))

    def clear(self):
        """Clear all documents from the index."""
        self.create_index()
    
    def get_size(self) -> int:
        """Get the number of documents in the index."""
        return self.index.ntotal if self.index is not None else 0