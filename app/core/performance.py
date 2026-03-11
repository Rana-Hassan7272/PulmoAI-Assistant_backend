"""
Performance Monitoring and Metrics Module

Tracks and logs performance metrics for:
- API response times
- ML model inference times
- RAG retrieval performance
- Database query performance
"""
import time
import logging
import statistics
from typing import Dict, List, Optional, Callable
from functools import wraps
from datetime import datetime
from collections import defaultdict, deque
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# In-memory storage for performance metrics
_performance_metrics = {
    "api_requests": deque(maxlen=1000),  # Store last 1000 API requests
    "ml_inference": defaultdict(list),   # Model name -> list of inference times
    "rag_retrieval": deque(maxlen=500),  # RAG query performance
    "db_queries": deque(maxlen=500),     # Database query performance
}

# Performance thresholds (in seconds)
PERFORMANCE_THRESHOLDS = {
    "api_fast": 0.5,      # Fast API response
    "api_slow": 2.0,      # Slow API response
    "ml_fast": 0.1,       # Fast ML inference
    "ml_slow": 1.0,       # Slow ML inference
    "rag_fast": 0.05,     # Fast RAG retrieval
    "rag_slow": 0.5,      # Slow RAG retrieval
    "db_fast": 0.01,      # Fast DB query
    "db_slow": 0.1,       # Slow DB query
}


def log_api_performance(path: str, method: str, duration: float, status_code: int):
    """
    Log API request performance.
    
    Args:
        path: API endpoint path
        method: HTTP method
        duration: Request duration in seconds
        status_code: HTTP status code
    """
    metric = {
        "timestamp": datetime.utcnow().isoformat(),
        "path": path,
        "method": method,
        "duration": duration,
        "status_code": status_code,
    }
    
    _performance_metrics["api_requests"].append(metric)
    
    # Log slow requests
    if duration > PERFORMANCE_THRESHOLDS["api_slow"]:
        logger.warning(
            f"SLOW API REQUEST: {method} {path} took {duration:.3f}s (status: {status_code})"
        )
    elif duration > PERFORMANCE_THRESHOLDS["api_fast"]:
        logger.info(
            f"API REQUEST: {method} {path} took {duration:.3f}s (status: {status_code})"
        )


def log_ml_inference(model_name: str, duration: float, input_size: Optional[Dict] = None):
    """
    Log ML model inference performance.
    
    Args:
        model_name: Name of the ML model (e.g., "xray", "spirometry", "bloodcount")
        duration: Inference time in seconds
        input_size: Optional dict with input dimensions/size info
    """
    metric = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": model_name,
        "duration": duration,
        "input_size": input_size,
    }
    
    _performance_metrics["ml_inference"][model_name].append(duration)
    
    # Keep only last 100 measurements per model
    if len(_performance_metrics["ml_inference"][model_name]) > 100:
        _performance_metrics["ml_inference"][model_name].pop(0)
    
    # Log slow inferences
    if duration > PERFORMANCE_THRESHOLDS["ml_slow"]:
        logger.warning(
            f"SLOW ML INFERENCE: {model_name} took {duration:.3f}s"
        )
    elif duration > PERFORMANCE_THRESHOLDS["ml_fast"]:
        logger.info(
            f"ML INFERENCE: {model_name} took {duration:.3f}s"
        )


def log_rag_retrieval(query: str, duration: float, documents_retrieved: int, k: int):
    """
    Log RAG retrieval performance.
    
    Args:
        query: Search query
        duration: Retrieval time in seconds
        documents_retrieved: Number of documents retrieved
        k: Requested number of documents
    """
    metric = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query[:100],  # Truncate long queries
        "duration": duration,
        "documents_retrieved": documents_retrieved,
        "k_requested": k,
    }
    
    _performance_metrics["rag_retrieval"].append(metric)
    
    # Log slow retrievals
    if duration > PERFORMANCE_THRESHOLDS["rag_slow"]:
        logger.warning(
            f"SLOW RAG RETRIEVAL: {duration:.3f}s for {documents_retrieved} documents (k={k})"
        )


def log_db_query(query_type: str, duration: float, table: Optional[str] = None):
    """
    Log database query performance.
    
    Args:
        query_type: Type of query (SELECT, INSERT, UPDATE, DELETE)
        duration: Query duration in seconds
        table: Table name (optional)
    """
    metric = {
        "timestamp": datetime.utcnow().isoformat(),
        "query_type": query_type,
        "duration": duration,
        "table": table,
    }
    
    _performance_metrics["db_queries"].append(metric)
    
    # Log slow queries
    if duration > PERFORMANCE_THRESHOLDS["db_slow"]:
        logger.warning(
            f"SLOW DB QUERY: {query_type} on {table or 'unknown'} took {duration:.3f}s"
        )


def get_performance_stats() -> Dict:
    """
    Get aggregated performance statistics.
    
    Returns:
        Dictionary with performance statistics
    """
    stats = {
        "api_stats": _calculate_api_stats(),
        "ml_stats": _calculate_ml_stats(),
        "rag_stats": _calculate_rag_stats(),
        "db_stats": _calculate_db_stats(),
    }
    return stats


def _calculate_api_stats() -> Dict:
    """Calculate API performance statistics."""
    requests = list(_performance_metrics["api_requests"])
    if not requests:
        return {"total_requests": 0}
    
    durations = [r["duration"] for r in requests]
    
    return {
        "total_requests": len(requests),
        "avg_response_time": statistics.mean(durations),
        "median_response_time": statistics.median(durations),
        "min_response_time": min(durations),
        "max_response_time": max(durations),
        "p95_response_time": _percentile(durations, 95),
        "p99_response_time": _percentile(durations, 99),
        "slow_requests": sum(1 for d in durations if d > PERFORMANCE_THRESHOLDS["api_slow"]),
        "fast_requests": sum(1 for d in durations if d < PERFORMANCE_THRESHOLDS["api_fast"]),
    }


def _calculate_ml_stats() -> Dict:
    """Calculate ML model performance statistics."""
    ml_data = _performance_metrics["ml_inference"]
    if not ml_data:
        return {"models": {}}
    
    model_stats = {}
    for model_name, durations in ml_data.items():
        if durations:
            model_stats[model_name] = {
                "total_inferences": len(durations),
                "avg_inference_time": statistics.mean(durations),
                "median_inference_time": statistics.median(durations),
                "min_inference_time": min(durations),
                "max_inference_time": max(durations),
                "p95_inference_time": _percentile(durations, 95),
                "p99_inference_time": _percentile(durations, 99),
                "slow_inferences": sum(1 for d in durations if d > PERFORMANCE_THRESHOLDS["ml_slow"]),
                "fast_inferences": sum(1 for d in durations if d < PERFORMANCE_THRESHOLDS["ml_fast"]),
            }
    
    return {"models": model_stats}


def _calculate_rag_stats() -> Dict:
    """Calculate RAG retrieval performance statistics."""
    retrievals = list(_performance_metrics["rag_retrieval"])
    if not retrievals:
        return {"total_retrievals": 0}
    
    durations = [r["duration"] for r in retrievals]
    docs_retrieved = [r["documents_retrieved"] for r in retrievals]
    
    return {
        "total_retrievals": len(retrievals),
        "avg_retrieval_time": statistics.mean(durations),
        "median_retrieval_time": statistics.median(durations),
        "min_retrieval_time": min(durations),
        "max_retrieval_time": max(durations),
        "avg_documents_retrieved": statistics.mean(docs_retrieved) if docs_retrieved else 0,
        "slow_retrievals": sum(1 for d in durations if d > PERFORMANCE_THRESHOLDS["rag_slow"]),
    }


def _calculate_db_stats() -> Dict:
    """Calculate database query performance statistics."""
    queries = list(_performance_metrics["db_queries"])
    if not queries:
        return {"total_queries": 0}
    
    durations = [q["duration"] for q in queries]
    
    return {
        "total_queries": len(queries),
        "avg_query_time": statistics.mean(durations),
        "median_query_time": statistics.median(durations),
        "min_query_time": min(durations),
        "max_query_time": max(durations),
        "slow_queries": sum(1 for d in durations if d > PERFORMANCE_THRESHOLDS["db_slow"]),
    }


def _percentile(data: List[float], percentile: int) -> float:
    """Calculate percentile of a list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = int(len(sorted_data) * percentile / 100)
    return sorted_data[min(index, len(sorted_data) - 1)]


def measure_time(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Usage:
        @measure_time
        def my_function():
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.debug(f"{func.__name__} took {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
            raise
    return wrapper


def save_performance_report(filepath: Optional[str] = None) -> str:
    """
    Save performance metrics to a JSON file.
    
    Args:
        filepath: Optional path to save report. If None, uses default location.
        
    Returns:
        Path to saved report file
    """
    if filepath is None:
        filepath = Path(__file__).parent.parent.parent / "performance_reports" / f"performance_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    else:
        filepath = Path(filepath)
    
    # Create directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "statistics": get_performance_stats(),
        "raw_metrics": {
            "api_requests": list(_performance_metrics["api_requests"]),
            "ml_inference": {k: v for k, v in _performance_metrics["ml_inference"].items()},
            "rag_retrieval": list(_performance_metrics["rag_retrieval"]),
            "db_queries": list(_performance_metrics["db_queries"]),
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Performance report saved to {filepath}")
    return str(filepath)
