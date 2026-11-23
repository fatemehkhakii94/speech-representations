"""Frozen model extraction and versioned feature storage."""

from .extractor import SpeechFeatureExtractor
from .store import FeatureStore, extract_features

__all__ = ["FeatureStore", "SpeechFeatureExtractor", "extract_features"]
