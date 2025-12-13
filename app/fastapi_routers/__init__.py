"""
FastAPI Routers Module

This module contains all FastAPI route handlers.
"""

from . import patients, visits, lab_results, imaging, spirometry, diagnostic, rag

__all__ = ['patients', 'visits', 'lab_results', 'imaging', 'spirometry', 'diagnostic', 'rag']

