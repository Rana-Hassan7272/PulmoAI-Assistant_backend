"""
RAG Agent for LangGraph

Provides RAG functionality as a tool/agent in the LangGraph workflow.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path

from .retriever import RAGRetriever


class RAGAgent:
    """RAG Agent that can be used in LangGraph workflows."""
    
    def __init__(
        self,
        knowledge_base_path: Optional[str] = None,
        vector_store_path: Optional[str] = None,
        embedding_model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize RAG Agent.
        
        Args:
            knowledge_base_path: Path to directory containing medical knowledge documents
                If None, defaults to documents folder in rag module
            vector_store_path: Path to save/load FAISS index
            embedding_model_name: Name of sentence-transformers model
        """
        # Default vector store path
        if vector_store_path is None:
            vector_store_path = os.path.join(
                Path(__file__).parent.parent.parent.parent,
                "data",
                "rag_index",
                "vector_store.index"
            )
            # Create directory if it doesn't exist
            Path(vector_store_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.vector_store_path = vector_store_path
        
        # Default knowledge base path to documents folder in rag module
        if knowledge_base_path is None:
            knowledge_base_path = os.path.join(
                Path(__file__).parent,
                "documents"
            )
        
        self.default_knowledge_base_path = knowledge_base_path
        
        # Initialize retriever (VectorStore will auto-load if index exists)
        self.retriever = RAGRetriever(
            embedding_model_name=embedding_model_name,
            vector_store_path=vector_store_path
        )
        
        # Check if we have documents loaded
        stats = self.retriever.get_stats()
        has_documents = stats.get('total_documents', 0) > 0
        
        if not has_documents:
            # No documents in index, try to load from documents folder
            if os.path.exists(knowledge_base_path) and os.path.isdir(knowledge_base_path):
                # Check if folder has any files
                files = [f for f in os.listdir(knowledge_base_path) 
                        if os.path.isfile(os.path.join(knowledge_base_path, f)) 
                        and f.lower().endswith(('.pdf', '.txt'))]
                
                if files:
                    print(f"📚 Loading {len(files)} document(s) from {knowledge_base_path}...")
                    print(f"   Files: {', '.join(files)}")
                    self.load_knowledge_base(knowledge_base_path)
                else:
                    print(f"ℹ️  Documents folder exists but contains no PDF/txt files: {knowledge_base_path}")
            else:
                print(f"ℹ️  Knowledge base path does not exist: {knowledge_base_path}")
                print("   You can add documents later using the API or load_knowledge_base() method.")
        else:
            print(f"✓ RAG system ready with {stats.get('total_documents', 0)} document chunks indexed")
    
    def load_knowledge_base(self, knowledge_base_path: str, clear_existing: bool = False):
        """
        Load medical knowledge documents from directory.
        
        Args:
            knowledge_base_path: Path to directory containing PDF/text files
            clear_existing: If True, clear existing index before loading
        """
        if clear_existing:
            print("Clearing existing index...")
            self.retriever.vector_store.clear()
        
        print(f"Loading knowledge base from {knowledge_base_path}...")
        self.retriever.add_directory(knowledge_base_path, chunk=True, recursive=True)
        self.retriever.save(self.vector_store_path)
        stats = self.retriever.get_stats()
        print(f"✓ Knowledge base loaded and indexed: {stats.get('total_documents', 0)} document chunks")
    
    def reload_documents(self):
        """
        Reload documents from the default knowledge base path.
        Useful when new documents are added to the documents folder.
        """
        if self.default_knowledge_base_path and os.path.exists(self.default_knowledge_base_path):
            print("Reloading documents from default knowledge base path...")
            self.load_knowledge_base(self.default_knowledge_base_path, clear_existing=True)
        else:
            print("Warning: Default knowledge base path not set or does not exist")
    
    def add_document(self, file_path: str):
        """Add a single document to the knowledge base."""
        self.retriever.add_file(file_path, chunk=True)
        self.retriever.save(self.vector_store_path)
    
    def add_pdf_bytes(self, pdf_bytes: bytes, source_name: str = "uploaded_file"):
        """Add PDF from bytes (for API uploads)."""
        self.retriever.add_pdf_bytes(pdf_bytes, source_name, chunk=True)
        self.retriever.save(self.vector_store_path)
    
    def add_text(self, text: str, source_name: str = "text_input"):
        """Add text to the knowledge base."""
        self.retriever.add_text(text, source_name, chunk=True)
        self.retriever.save(self.vector_store_path)
    
    def retrieve_context(
        self,
        query: str,
        k: int = 5,
        min_similarity: float = 0.3
    ) -> str:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: Query text (e.g., patient symptoms, test results)
            k: Number of documents to retrieve
            min_similarity: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            Formatted context string for LLM prompts
        """
        import time
        from ...core.performance import log_rag_retrieval
        
        start_time = time.time()
        context = self.retriever.retrieve_formatted(query, k=k, min_similarity=min_similarity)
        duration = time.time() - start_time
        
        # Count documents retrieved (rough estimate)
        documents_retrieved = context.count("[") if context else 0
        
        # Log performance
        log_rag_retrieval(query, duration, documents_retrieved, k)
        
        return context
    
    def retrieve_documents(
        self,
        query: str,
        k: int = 5,
        min_similarity: float = 0.3
    ) -> list:
        """
        Retrieve relevant documents (raw format).
        
        Args:
            query: Query text
            k: Number of documents to retrieve
            min_similarity: Minimum similarity score
            
        Returns:
            List of tuples (document, similarity_score)
        """
        import time
        from ...core.performance import log_rag_retrieval
        
        start_time = time.time()
        documents = self.retriever.retrieve(query, k=k, min_similarity=min_similarity)
        duration = time.time() - start_time
        
        # Log performance
        log_rag_retrieval(query, duration, len(documents), k)
        
        return documents
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system."""
        return self.retriever.get_stats()


# Global RAG Agent instance (singleton pattern)
_rag_agent_instance: Optional[RAGAgent] = None


def get_rag_agent(
    knowledge_base_path: Optional[str] = None,
    vector_store_path: Optional[str] = None,
    force_reload: bool = False
) -> RAGAgent:
    """
    Get or create the global RAG Agent instance.
    
    Automatically loads documents from backend/app/agents/rag/documents/ folder
    if no knowledge_base_path is provided.
    
    Args:
        knowledge_base_path: Path to knowledge base directory (defaults to documents folder)
        vector_store_path: Path to vector store index
        force_reload: Force reload even if instance exists
        
    Returns:
        RAGAgent instance
    """
    global _rag_agent_instance
    
    if _rag_agent_instance is None or force_reload:
        # If no knowledge_base_path provided, use default documents folder
        if knowledge_base_path is None:
            knowledge_base_path = os.path.join(
                Path(__file__).parent,
                "documents"
            )
        
        _rag_agent_instance = RAGAgent(
            knowledge_base_path=knowledge_base_path,
            vector_store_path=vector_store_path
        )
    
    return _rag_agent_instance

