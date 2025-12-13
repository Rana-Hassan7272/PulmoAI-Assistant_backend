"""
Session Manager for LangGraph State Persistence

Manages in-memory storage of AgentState for multi-turn conversations.
Uses visit_id as the session key.
"""

from typing import Dict, Optional
from .state import AgentState
import threading
from datetime import datetime, timedelta


class SessionManager:
    """
    Thread-safe session manager for storing AgentState.
    
    Sessions are stored in memory and automatically expire after 24 hours.
    """
    
    def __init__(self, session_timeout_hours: int = 24):
        """
        Initialize session manager.
        
        Args:
            session_timeout_hours: Hours before session expires (default: 24)
        """
        self._sessions: Dict[str, Dict] = {}
        self._session_timestamps: Dict[str, datetime] = {}
        self._lock = threading.Lock()
        self._session_timeout = timedelta(hours=session_timeout_hours)
    
    def save_state(self, visit_id: str, state: AgentState) -> None:
        """
        Save state for a visit session.
        
        Args:
            visit_id: Unique visit identifier
            state: AgentState to save
        """
        with self._lock:
            # Convert state to dict (it's already a dict, but ensure it's serializable)
            state_dict = dict(state)
            self._sessions[visit_id] = state_dict
            self._session_timestamps[visit_id] = datetime.now()
    
    def load_state(self, visit_id: str) -> Optional[AgentState]:
        """
        Load state for a visit session.
        
        Args:
            visit_id: Unique visit identifier
            
        Returns:
            AgentState if found and not expired, None otherwise
        """
        with self._lock:
            # Check if session exists
            if visit_id not in self._sessions:
                return None
            
            # Check if session expired
            if visit_id in self._session_timestamps:
                timestamp = self._session_timestamps[visit_id]
                if datetime.now() - timestamp > self._session_timeout:
                    # Session expired, remove it
                    self._delete_session(visit_id)
                    return None
            
            # Return state
            return self._sessions[visit_id]
    
    def update_state(self, visit_id: str, updates: Dict) -> Optional[AgentState]:
        """
        Update state incrementally.
        
        Args:
            visit_id: Unique visit identifier
            updates: Dictionary of fields to update
            
        Returns:
            Updated AgentState, or None if session doesn't exist
        """
        with self._lock:
            if visit_id not in self._sessions:
                return None
            
            # Update state with new values
            current_state = self._sessions[visit_id]
            for key, value in updates.items():
                if value is not None:  # Only update non-None values
                    current_state[key] = value
            
            # Special handling for conversation_history (append)
            if "conversation_history" in updates:
                if "conversation_history" not in current_state:
                    current_state["conversation_history"] = []
                # Append new messages
                new_messages = updates["conversation_history"]
                if isinstance(new_messages, list):
                    current_state["conversation_history"].extend(new_messages)
            
            # Update timestamp
            self._session_timestamps[visit_id] = datetime.now()
            
            return current_state
    
    def delete_session(self, visit_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            visit_id: Unique visit identifier
            
        Returns:
            True if session was deleted, False if it didn't exist
        """
        with self._lock:
            return self._delete_session(visit_id)
    
    def _delete_session(self, visit_id: str) -> bool:
        """Internal method to delete session (must be called with lock held)."""
        if visit_id in self._sessions:
            del self._sessions[visit_id]
            if visit_id in self._session_timestamps:
                del self._session_timestamps[visit_id]
            return True
        return False
    
    def cleanup_expired_sessions(self) -> int:
        """
        Remove all expired sessions.
        
        Returns:
            Number of sessions removed
        """
        with self._lock:
            expired_visits = []
            now = datetime.now()
            
            for visit_id, timestamp in self._session_timestamps.items():
                if now - timestamp > self._session_timeout:
                    expired_visits.append(visit_id)
            
            for visit_id in expired_visits:
                self._delete_session(visit_id)
            
            return len(expired_visits)
    
    def get_session_count(self) -> int:
        """Get number of active sessions."""
        with self._lock:
            return len(self._sessions)
    
    def has_session(self, visit_id: str) -> bool:
        """Check if session exists and is not expired."""
        with self._lock:
            if visit_id not in self._sessions:
                return False
            
            if visit_id in self._session_timestamps:
                timestamp = self._session_timestamps[visit_id]
                if datetime.now() - timestamp > self._session_timeout:
                    self._delete_session(visit_id)
                    return False
            
            return True


# Global session manager instance (singleton)
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get or create the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager

