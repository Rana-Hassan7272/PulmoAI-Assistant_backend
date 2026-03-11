"""
Tests for Lab Results API endpoints.
"""
import pytest
from fastapi import status
from unittest.mock import patch


class TestLabResultsEndpoints:
    """Test suite for lab results endpoints."""
    
    @patch('app.ml_models.bloodcount_report.feature.predict_blood_disease')
    @patch('app.ml_models.bloodcount_report.feature.predict_blood_disease_proba')
    def test_analyze_blood_count_success(self, mock_proba, mock_predict, client):
        """Test successful blood count analysis."""
        # Mock predictions
        mock_predict.return_value = {
            "disease_name": "Normal",
            "confidence": 0.92
        }
        mock_proba.return_value = {
            "Normal": 0.92,
            "Anemia": 0.05,
            "Other": 0.03
        }
        
        response = client.post(
            "/labs/analyze-blood-count",
            json={
                "wbc": 7.0,
                "rbc": 4.5,
                "hgb": 14.0
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        assert "prediction" in response.json()
        assert "probabilities" in response.json()
    
    def test_create_lab_result(self, client, auth_headers, test_db):
        """Test creating a lab result record."""
        response = client.post(
            "/labs/",
            headers=auth_headers,
            json={
                "patient_id": 1,
                "test_name": "CBC",
                "value": 7.0,
                "unit": "x1000/µL"
            }
        )
        
        # Should create lab result
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_201_CREATED]
