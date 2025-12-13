"""
RAG Retriever

Combines document loading, chunking, embedding, and retrieval.
"""

import os
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np

from .document_loader import DocumentLoader
from .embeddings import EmbeddingModel
from .vector_store import VectorStore


class RAGRetriever:
    """Complete RAG retrieval system."""
    
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        vector_store_path: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize RAG Retriever.
        
        Args:
            embedding_model_name: Name of sentence-transformers model
            vector_store_path: Path to save/load FAISS index
            chunk_size: Size of text chunks (characters)
            chunk_overlap: Overlap between chunks (characters)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.document_loader = DocumentLoader()
        self.embedding_model = EmbeddingModel(embedding_model_name)
        
        # Initialize vector store
        embedding_dim = self.embedding_model.get_embedding_dimension()
        self.vector_store = VectorStore(embedding_dim, vector_store_path)
    
    def _chunk_text(self, text: str, metadata: Dict) -> List[Dict[str, str]]:
        """
        Split text into chunks.
        
        Args:
            text: Text to chunk
            metadata: Metadata for the document
            
        Returns:
            List of chunked documents
        """
        chunks = []
        
        # Simple chunking by character count
        # You can enhance this with sentence-aware chunking
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            if chunk_text.strip():
                chunks.append({
                    'content': chunk_text.strip(),
                    'metadata': {
                        **metadata,
                        'chunk_index': len(chunks),
                        'start_char': start,
                        'end_char': end
                    }
                })
            
            start = end - self.chunk_overlap
        
        return chunks
    
    def add_documents(
        self,
        documents: List[Dict[str, str]],
        chunk: bool = True
    ):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents with 'content' and 'metadata'
            chunk: Whether to chunk documents before adding
        """
        all_chunks = []
        
        for doc in documents:
            if chunk:
                chunks = self._chunk_text(doc['content'], doc.get('metadata', {}))
                all_chunks.extend(chunks)
            else:
                all_chunks.append(doc)
        
        if not all_chunks:
            print("Warning: No documents to add")
            return
        
        # Generate embeddings
        texts = [chunk['content'] for chunk in all_chunks]
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(texts, show_progress=True)
        
        # Add to vector store
        self.vector_store.add_documents(all_chunks, embeddings)
        print(f"Added {len(all_chunks)} document chunks to vector store")
    
    def add_file(self, file_path: str, chunk: bool = True):
        """Add a single file to the vector store."""
        documents = self.document_loader.load_file(file_path)
        self.add_documents(documents, chunk=chunk)
    
    def add_directory(self, directory_path: str, chunk: bool = True, recursive: bool = True):
        """Add all documents from a directory."""
        documents = self.document_loader.load_directory(directory_path, recursive=recursive)
        self.add_documents(documents, chunk=chunk)
    
    def add_pdf_bytes(self, pdf_bytes: bytes, source_name: str = "uploaded_file", chunk: bool = True):
        """Add PDF from bytes (for file uploads)."""
        documents = self.document_loader.load_pdf_from_bytes(pdf_bytes, source_name)
        self.add_documents(documents, chunk=chunk)
    
    def add_text(self, text: str, source_name: str = "text_input", chunk: bool = True):
        """Add text string to the vector store."""
        documents = self.document_loader.load_text_from_string(text, source_name)
        self.add_documents(documents, chunk=chunk)
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        min_similarity: float = 0.0
    ) -> List[Tuple[Dict[str, str], float]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query text
            k: Number of results to return
            min_similarity: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            List of tuples (document, similarity_score)
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode_single(query)
        
        # Search in vector store
        results = self.vector_store.search(query_embedding, k=k)
        
        # Filter by minimum similarity
        filtered_results = [
            (doc, score) for doc, score in results
            if score >= min_similarity
        ]
        
        return filtered_results
    
    def retrieve_formatted(
        self,
        query: str,
        k: int = 5,
        min_similarity: float = 0.0
    ) -> str:
        """
        Retrieve and format results as a string for LLM prompts.
        
        Args:
            query: Query text
            k: Number of results to return
            min_similarity: Minimum similarity score
            
        Returns:
            Formatted string with retrieved context
        """
        results = self.retrieve(query, k=k, min_similarity=min_similarity)
        
        if not results:
            return "No relevant documents found."
        
        formatted_context = "Retrieved Medical Knowledge:\n\n"
        for i, (doc, score) in enumerate(results, 1):
            source = doc.get('metadata', {}).get('source', 'Unknown')
            page = doc.get('metadata', {}).get('page', '')
            page_info = f" (Page {page})" if page else ""
            
            formatted_context += f"[{i}] Source: {source}{page_info} (Relevance: {score:.2%})\n"
            formatted_context += f"{doc['content']}\n\n"
        
        return formatted_context
    
    def save(self, index_path: Optional[str] = None):
        """Save vector store to disk."""
        self.vector_store.save(index_path)
    
    def load(self, index_path: Optional[str] = None):
        """Load vector store from disk."""
        self.vector_store.load(index_path)
    
    def get_stats(self) -> Dict:
        """Get statistics about the retriever."""
        return self.vector_store.get_stats()

