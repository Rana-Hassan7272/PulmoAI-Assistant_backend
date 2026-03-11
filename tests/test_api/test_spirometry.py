"""
Tests for Spirometry API endpoints.
"""
import pytest
from fastapi import status
from unittest.mock import patch


class TestSpirometryEndpoints:
    """Test suite for spirometry endpoints."""
    
    @patch('app.ml_models.spirometry.featurizer.predict_spirometry')
    @patch('app.ml_models.spirometry.featurizer.predict_spirometry_proba')
    def test_analyze_spirometry_success(self, mock_proba, mock_predict, client):
        """Test successful spirometry analysis."""
        # Mock predictions
        mock_predict.return_value = {
            "obstruction": 0,
            "restriction": 0,
            "prism": 0,
            "mixed": 0
        }
        mock_proba.return_value = {
            "obstruction": 0.1,
            "restriction": 0.05,
            "prism": 0.02,
            "mixed": 0.01
        }
        
        response = client.post(
            "/spirometry/analyze",
            json={
                "fev1": 5.0,
                "fvc": 5.0
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        # Response uses "prediction" not "predictions"
        assert "prediction" in response.json()
        assert "probabilities" in response.json()
    
    def test_analyze_spirometry_with_missing_values(self, client):
        """Test spirometry analysis with missing values (should fill defaults)."""
        response = client.post(
            "/spirometry/analyze",
            json={
                "fev1": 5.0
                # fvc missing, should be filled with default
            }
        )
        
        # Should still work with defaults
        assert response.status_code == status.HTTP_200_OK
