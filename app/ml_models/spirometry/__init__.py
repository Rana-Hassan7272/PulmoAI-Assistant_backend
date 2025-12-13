"""
Spirometry ML Models Module

This module provides access to spirometry prediction models.
"""

from .featurizer import (
    SpirometryFeaturizer,
    get_featurizer,
    predict_spirometry,
    predict_spirometry_proba
)

__all__ = [
    'SpirometryFeaturizer',
    'get_featurizer',
    'predict_spirometry',
    'predict_spirometry_proba'
]

