"""API module for serving R2R models."""

from .app import create_app
from .schemas import PredictionRequest, PredictionResponse

__all__ = ["create_app", "PredictionRequest", "PredictionResponse"]
