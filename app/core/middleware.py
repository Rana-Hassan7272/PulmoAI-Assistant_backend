"""
FastAPI Middleware for Performance Monitoring

Adds response time logging to all API requests.
"""
import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from .performance import log_api_performance

logger = logging.getLogger(__name__)


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log API request performance.
    
    Logs:
    - Request path and method
    - Response time
    - Status code
    """
    
    async def dispatch(self, request: Request, call_next):
        # Skip logging for health checks and static files
        if request.url.path in ["/health", "/docs", "/openapi.json", "/redoc"]:
            return await call_next(request)
        
        # Record start time
        start_time = time.time()
        
        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            # If exception occurs, still log the time
            duration = time.time() - start_time
            status_code = 500
            log_api_performance(
                path=str(request.url.path),
                method=request.method,
                duration=duration,
                status_code=status_code
            )
            raise
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log performance
        log_api_performance(
            path=str(request.url.path),
            method=request.method,
            duration=duration,
            status_code=status_code
        )
        
        # Add performance header (optional)
        response.headers["X-Response-Time"] = f"{duration:.3f}"
        
        return response
