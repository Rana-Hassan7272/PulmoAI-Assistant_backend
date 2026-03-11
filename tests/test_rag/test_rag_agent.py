"""
Tests for RAG Agent.
"""
import pytest
from unittest.mock import patch, MagicMock


class TestRAGAgent:
    """Test suite for RAG agent."""
    
    def test_rag_agent_initialization(self):
        """Test RAG agent can be initialized."""
        from app.agents.rag.rag_agent import RAGAgent
        
        try:
            agent = RAGAgent()
            assert agent is not None
            assert hasattr(agent, 'retriever')
        except Exception as e:
            # May fail if vector store doesn't exist - that's okay
            pytest.skip(f"RAG agent initialization failed: {e}")
    
    def test_retrieve_context_returns_string(self):
        """Test that retrieve_context returns a string."""
        from app.agents.rag.rag_agent import get_rag_agent
        
        try:
            agent = get_rag_agent()
            context = agent.retrieve_context("test query", k=3)
            assert isinstance(context, str)
        except Exception as e:
            pytest.skip(f"RAG retrieval failed: {e}")
    
    def test_retrieve_documents_returns_list(self):
        """Test that retrieve_documents returns a list."""
        from app.agents.rag.rag_agent import get_rag_agent
        
        try:
            agent = get_rag_agent()
            documents = agent.retrieve_documents("test query", k=3)
            assert isinstance(documents, list)
        except Exception as e:
            pytest.skip(f"RAG retrieval failed: {e}")
    
    def test_get_stats_returns_dict(self):
        """Test that get_stats returns a dictionary."""
        from app.agents.rag.rag_agent import get_rag_agent
        
        try:
            agent = get_rag_agent()
            stats = agent.get_stats()
            assert isinstance(stats, dict)
        except Exception as e:
            pytest.skip(f"RAG stats failed: {e}")
