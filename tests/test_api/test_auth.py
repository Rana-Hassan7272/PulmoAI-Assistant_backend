"""
Tests for Authentication API endpoints.
"""
import pytest
from fastapi import status


class TestAuthEndpoints:
    """Test suite for authentication endpoints."""
    
    def test_signup_success(self, client, test_db):
        """Test successful user registration."""
        response = client.post(
            "/auth/signup",
            json={
                "email": "newuser@example.com",
                "password": "securepassword123",
                "name": "New User",
                "age": 25,
                "gender": "Female"
            }
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        assert "access_token" in response.json()
        assert response.json()["token_type"] == "bearer"
        assert response.json()["patient_id"] is not None
    
    def test_signup_duplicate_email(self, client, test_user):
        """Test signup with duplicate email fails."""
        response = client.post(
            "/auth/signup",
            json={
                "email": "test@example.com",
                "password": "password123"
            }
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "already registered" in response.json()["detail"].lower()
    
    def test_login_success(self, client, test_user):
        """Test successful login."""
        response = client.post(
            "/auth/login",
            json={
                "email": "test@example.com",
                "password": "testpassword123"
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        assert "access_token" in response.json()
        assert response.json()["token_type"] == "bearer"
    
    def test_login_invalid_credentials(self, client, test_user):
        """Test login with invalid credentials."""
        response = client.post(
            "/auth/login",
            json={
                "email": "test@example.com",
                "password": "wrongpassword"
            }
        )
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "incorrect" in response.json()["detail"].lower()
    
    def test_get_current_user(self, client, auth_headers):
        """Test getting current user info."""
        response = client.get("/auth/me", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        assert "email" in response.json()
        assert "id" in response.json()
    
    def test_get_current_user_unauthorized(self, client):
        """Test getting user info without token."""
        response = client.get("/auth/me")
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
