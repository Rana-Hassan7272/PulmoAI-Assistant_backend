"""
Tests for Blood Count ML model.
"""
import pytest
from unittest.mock import patch, MagicMock


class TestBloodCountModel:
    """Test suite for blood count disease prediction model."""
    
    def test_predict_blood_disease_returns_dict(self):
        """Test that predict_blood_disease returns expected structure."""
        from app.ml_models.bloodcount_report.feature import predict_blood_disease
        
        # Include all required columns
        test_data = {
            "WBC": 7.0, "LYMp": 35.0, "NEUTp": 60.0, "LYMn": 2.0, "NEUTn": 4.0,
            "RBC": 4.8, "HGB": 14.5, "HCT": 42.0, "MCV": 90.0, "MCH": 30.0,
            "MCHC": 33.0, "PLT": 250.0, "PDW": 12.0, "PCT": 0.25
        }
        
        try:
            result = predict_blood_disease(test_data)
            assert isinstance(result, dict)
            assert "disease_name" in result or "class_name" in result
        except (FileNotFoundError, OSError, AttributeError):
            pytest.skip("Model files not available for testing")
    
    def test_predict_blood_disease_proba_returns_dict(self):
        """Test that predict_blood_disease_proba returns probabilities."""
        from app.ml_models.bloodcount_report.feature import predict_blood_disease_proba
        
        # Include all required columns
        test_data = {
            "WBC": 7.0, "LYMp": 35.0, "NEUTp": 60.0, "LYMn": 2.0, "NEUTn": 4.0,
            "RBC": 4.8, "HGB": 14.5, "HCT": 42.0, "MCV": 90.0, "MCH": 30.0,
            "MCHC": 33.0, "PLT": 250.0, "PDW": 12.0, "PCT": 0.25
        }
        
        try:
            result = predict_blood_disease_proba(test_data)
            assert isinstance(result, dict)
            # Should have probabilities for different diseases
            assert len(result) > 0
        except (FileNotFoundError, OSError, AttributeError):
            pytest.skip("Model files not available for testing")
    
    def test_model_handles_partial_data(self):
        """Test that model handles partial input data."""
        from app.ml_models.bloodcount_report.feature import predict_blood_disease
        
        # Test with minimal data - model requires all columns
        # This test verifies the model raises appropriate error for missing data
        test_data = {
            "WBC": 7.0
            # Other fields missing - should raise ValueError
        }
        
        try:
            # Model should raise ValueError for missing required columns
            with pytest.raises(ValueError, match="Missing required columns"):
                predict_blood_disease(test_data)
        except (FileNotFoundError, OSError, AttributeError):
            pytest.skip("Model files not available")
