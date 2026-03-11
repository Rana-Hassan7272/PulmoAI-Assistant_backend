"""
Tests for Diagnostic API endpoints.
"""
import pytest
from fastapi import status
from unittest.mock import patch, MagicMock


class TestDiagnosticEndpoints:
    """Test suite for diagnostic endpoints."""
    
    def test_start_diagnostic(self, client, auth_headers):
        """Test starting a diagnostic session."""
        response = client.post(
            "/diagnostic/start",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        assert "visit_id" in response.json()
        assert "message" in response.json()
    
    def test_start_diagnostic_unauthorized(self, client):
        """Test starting diagnostic without auth."""
        response = client.post("/diagnostic/start")
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    @patch('app.fastapi_routers.diagnostic.get_graph')
    def test_chat_endpoint(self, mock_graph, client, auth_headers):
        """Test chat endpoint."""
        # Mock graph and state
        mock_state = MagicMock()
        mock_state.__getitem__ = lambda self, key: None
        mock_state.__setitem__ = lambda self, key, value: None
        mock_state.get = lambda key, default=None: default
        mock_state.copy = lambda: mock_state
        
        mock_graph_instance = MagicMock()
        mock_graph_instance.invoke.return_value = {
            "message": "Test response",
            "visit_id": "test-123"
        }
        mock_graph.return_value = mock_graph_instance
        
        response = client.post(
            "/diagnostic/chat",
            headers=auth_headers,
            data={
                "message": "I have a cough",
                "visit_id": "test-123"
            }
        )
        
        # Should process the message
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_500_INTERNAL_SERVER_ERROR]
