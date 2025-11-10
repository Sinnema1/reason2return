"""API request/response schemas."""

from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint."""

    ticker: str = Field(..., description="Stock ticker symbol")
    date: str = Field(..., description="Analysis date (ISO format)")
    features: dict[str, Any] = Field(default_factory=dict, description="Feature dictionary")


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""

    correlation_id: str = Field(..., description="Request correlation ID")
    thesis: dict[str, Any] = Field(..., description="Generated thesis")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Response metadata")
