"""
Database Query Performance Tracking

Wraps database operations to track query performance.
"""
import time
import logging
from functools import wraps
from sqlalchemy.orm import Session
from .performance import log_db_query

logger = logging.getLogger(__name__)


def track_db_query(query_type: str, table: str = None):
    """
    Decorator to track database query performance.
    
    Usage:
        @track_db_query("SELECT", "patients")
        def get_patients(db: Session):
            return db.query(Patient).all()
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                log_db_query(query_type, duration, table)
                return result
            except Exception as e:
                duration = time.time() - start_time
                log_db_query(query_type, duration, table)
                raise
        return wrapper
    return decorator


def track_db_operation(operation_func):
    """
    Context manager for tracking database operations.
    
    Usage:
        with track_db_operation("SELECT", "patients") as track:
            result = db.query(Patient).all()
    """
    class DBOperationTracker:
        def __init__(self, query_type: str, table: str = None):
            self.query_type = query_type
            self.table = table
            self.start_time = None
        
        def __enter__(self):
            self.start_time = time.time()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            duration = time.time() - self.start_time
            log_db_query(self.query_type, duration, self.table)
            return False
    
    return DBOperationTracker
