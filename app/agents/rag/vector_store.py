"""
FAISS Vector Store for RAG Pipeline

Manages document embeddings and similarity search.
"""

import os
import pickle
from typing import List, Dict, Optional, Tuple
import numpy as np
import faiss
from pathlib import Path


class VectorStore:
    """FAISS-based vector store for document embeddings."""
    
    def __init__(self, embedding_dimension: int, index_path: Optional[str] = None):
        """
        Initialize FAISS vector store.
        
        Args:
            embedding_dimension: Dimension of embeddings
            index_path: Optional path to save/load FAISS index
        """
        self.embedding_dimension = embedding_dimension
        self.index_path = index_path
        
        # Initialize FAISS index (L2 distance)
        self.index = faiss.IndexFlatL2(embedding_dimension)
        
        # Store document metadata
        self.documents: List[Dict[str, str]] = []
        self.metadata: List[Dict] = []
        
        # Load existing index if path provided
        if index_path and os.path.exists(index_path):
            self.load(index_path)
    
    def add_documents(self, documents: List[Dict[str, str]], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the vector store.
        
        Args:
            documents: List of documents with 'content' and 'metadata'
            embeddings: numpy array of embeddings (n_docs, embedding_dim)
        """
        if len(documents) != embeddings.shape[0]:
            raise ValueError(f"Number of documents ({len(documents)}) doesn't match number of embeddings ({embeddings.shape[0]})")
        
        if embeddings.shape[1] != self.embedding_dimension:
            raise ValueError(f"Embedding dimension ({embeddings.shape[1]}) doesn't match expected ({self.embedding_dimension})")
        
        # Add embeddings to FAISS index
        # FAISS requires float32
        embeddings_float32 = embeddings.astype('float32')
        self.index.add(embeddings_float32)
        
        # Store documents and metadata
        for doc in documents:
            self.documents.append(doc)
            self.metadata.append(doc.get('metadata', {}))
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Dict[str, str], float]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding (embedding_dim,) or (1, embedding_dim)
            k: Number of results to return
            
        Returns:
            List of tuples (document, similarity_score)
        """
        if self.index.ntotal == 0:
            return []
        
        # Ensure query_embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Convert to float32
        query_embedding = query_embedding.astype('float32')
        
        # Search in FAISS
        k = min(k, self.index.ntotal)  # Don't request more than available
        distances, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):
                # Convert L2 distance to similarity (lower distance = higher similarity)
                # Using inverse distance as similarity score
                similarity = 1.0 / (1.0 + distance)
                results.append((self.documents[idx], similarity))
        
        return results
    
    def save(self, index_path: Optional[str] = None):
        """
        Save FAISS index and metadata to disk.
        
        Args:
            index_path: Path to save index (uses self.index_path if None)
        """
        save_path = index_path or self.index_path
        if not save_path:
            raise ValueError("No index_path provided")
        
        # Create directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, save_path)
        
        # Save documents and metadata
        metadata_path = save_path.replace('.index', '_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata,
                'embedding_dimension': self.embedding_dimension
            }, f)
        
        print(f"Vector store saved to {save_path}")
    
    def load(self, index_path: Optional[str] = None):
        """
        Load FAISS index and metadata from disk.
        
        Args:
            index_path: Path to load index from (uses self.index_path if None)
        """
        load_path = index_path or self.index_path
        if not load_path or not os.path.exists(load_path):
            raise FileNotFoundError(f"Index file not found: {load_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(load_path)
        
        # Load documents and metadata
        metadata_path = load_path.replace('.index', '_metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data.get('documents', [])
                self.metadata = data.get('metadata', [])
                self.embedding_dimension = data.get('embedding_dimension', self.embedding_dimension)
        
        print(f"Vector store loaded from {load_path}. {self.index.ntotal} documents indexed.")
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        return {
            'total_documents': self.index.ntotal,
            'embedding_dimension': self.embedding_dimension,
            'index_type': type(self.index).__name__
        }
    
    def clear(self):
        """Clear all documents from the vector store."""
        self.index.reset()
        self.documents = []
        self.metadata = []

