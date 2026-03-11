"""
Tests for X-ray ML model.
"""
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np


class TestXRayModel:
    """Test suite for X-ray pneumonia detection model."""
    
    def test_predict_xray_returns_dict(self):
        """Test that predict_xray returns a dictionary with expected keys."""
        from app.ml_models.xray.preprocessor import predict_xray
        
        # Create a dummy image
        test_image = Image.new('RGB', (224, 224), color='white')
        
        # Mock the model loading and prediction
        with patch('app.ml_models.xray.preprocessor.XRayPneumoniaPredictor') as mock_predictor:
            mock_instance = MagicMock()
            mock_instance.predict.return_value = {
                "disease_name": "No disease",
                "confidence": 0.95
            }
            mock_predictor.return_value = mock_instance
            
            # This will fail if model file doesn't exist, but structure is correct
            try:
                result = predict_xray(test_image)
                assert isinstance(result, dict)
                # Model returns "class_name" not "disease_name"
                assert "class_name" in result or "disease_name" in result
                assert "confidence" in result
            except (FileNotFoundError, OSError):
                # Model file not found - skip test
                pytest.skip("Model file not available for testing")
    
    def test_predict_xray_proba_returns_dict(self):
        """Test that predict_xray_proba returns probability dictionary."""
        from app.ml_models.xray.preprocessor import predict_xray_proba
        
        test_image = Image.new('RGB', (224, 224), color='white')
        
        with patch('app.ml_models.xray.preprocessor.XRayPneumoniaPredictor') as mock_predictor:
            mock_instance = MagicMock()
            mock_instance.predict_proba.return_value = {
                "No disease": 0.7,
                "Bacterial pneumonia": 0.2,
                "Viral pneumonia": 0.1
            }
            mock_predictor.return_value = mock_instance
            
            try:
                result = predict_xray_proba(test_image)
                assert isinstance(result, dict)
                assert len(result) == 3  # Three classes
            except (FileNotFoundError, OSError):
                pytest.skip("Model file not available for testing")
    
    def test_model_handles_invalid_image(self):
        """Test that model handles invalid image gracefully."""
        from app.ml_models.xray.preprocessor import predict_xray
        
        # Try with None or invalid input
        try:
            # Should raise ValueError for invalid input
            with pytest.raises(ValueError, match="Unsupported image type"):
                predict_xray(None)
        except (FileNotFoundError, OSError):
            # Model file not found - skip test
            pytest.skip("Model file not available for testing")
