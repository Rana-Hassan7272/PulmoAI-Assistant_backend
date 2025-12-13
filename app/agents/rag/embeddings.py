"""
Embedding Model for RAG Pipeline

Uses sentence-transformers for generating embeddings.
"""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import os


class EmbeddingModel:
    """Generate embeddings for text using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model.
        
        Args:
            model_name: Name of sentence-transformers model
                - "all-MiniLM-L6-v2": Fast, 384 dimensions (default)
                - "all-mpnet-base-v2": Better quality, 768 dimensions (slower)
                - "paraphrase-multilingual-MiniLM-L12-v2": Multilingual support
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence-transformers model."""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Embedding model loaded successfully. Dimension: {self.get_embedding_dimension()}")
        except Exception as e:
            raise Exception(f"Failed to load embedding model {self.model_name}: {str(e)}")
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        if self.model is None:
            return 0
        # Get dimension by encoding a test string
        test_embedding = self.model.encode(["test"], convert_to_numpy=True)
        return test_embedding.shape[1]
    
    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings (n_texts, embedding_dim)
        """
        if self.model is None:
            raise Exception("Model not loaded")
        
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string
            
        Returns:
            numpy array of embedding (embedding_dim,)
        """
        return self.encode([text])[0]

