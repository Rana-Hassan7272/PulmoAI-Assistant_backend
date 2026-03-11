"""
Tests for Imaging API endpoints.
"""
import pytest
from fastapi import status
from unittest.mock import patch, MagicMock


class TestImagingEndpoints:
    """Test suite for imaging endpoints."""
    
    @patch('app.ml_models.xray.preprocessor.predict_xray')
    @patch('app.ml_models.xray.preprocessor.predict_xray_proba')
    def test_analyze_xray_success(self, mock_proba, mock_predict, client, sample_image):
        """Test successful X-ray analysis."""
        # Mock predictions
        mock_predict.return_value = {
            "disease_name": "Viral pneumonia",
            "confidence": 0.85
        }
        mock_proba.return_value = {
            "No disease": 0.10,
            "Bacterial pneumonia": 0.05,
            "Viral pneumonia": 0.85
        }
        
        response = client.post(
            "/imaging/analyze-xray",
            files={"file": ("test.png", sample_image, "image/png")}
        )
        
        assert response.status_code == status.HTTP_200_OK
        assert "prediction" in response.json()
        assert "probabilities" in response.json()
    
    def test_analyze_xray_invalid_file(self, client):
        """Test X-ray analysis with invalid file."""
        response = client.post(
            "/imaging/analyze-xray",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        
        # Should return error for invalid image
        assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_422_UNPROCESSABLE_ENTITY]
    
    def test_analyze_xray_missing_file(self, client):
        """Test X-ray analysis without file."""
        response = client.post("/imaging/analyze-xray")
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
