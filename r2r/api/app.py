"""FastAPI application."""

from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from r2r.api.schemas import PredictionRequest, PredictionResponse
from r2r.models.pipeline import ThesisPipeline
from r2r.models.validator import ThesisValidator
from r2r.utils.logging import get_logger, set_correlation_id

logger = get_logger(__name__)


def create_app(
    model_path: Optional[Path] = None,
    schema_path: Optional[Path] = None,
) -> FastAPI:
    """Create FastAPI application.

    Args:
        model_path: Path to model checkpoint
        schema_path: Path to thesis schema

    Returns:
        FastAPI app instance
    """
    app = FastAPI(
        title="Reason2Return API",
        description="Structured LLM reasoning for financial decision support",
        version="0.1.0",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize model and validator
    # TODO: Load actual model checkpoint
    model_config = {"hidden_size": 768}
    model = ThesisPipeline(model_config)
    validator = ThesisValidator(schema_path)

    logger.info("Initialized R2R API")

    @app.get("/")
    def root() -> dict[str, str]:
        """Root endpoint."""
        return {
            "service": "Reason2Return API",
            "version": "0.1.0",
            "status": "running",
        }

    @app.get("/health")
    def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        """Generate thesis and prediction.

        Args:
            request: Prediction request

        Returns:
            Prediction response with thesis
        """
        # Set correlation ID for request tracing
        correlation_id = set_correlation_id()
        logger.info(f"Received prediction request for {request.ticker}")

        try:
            # Prepare inputs
            inputs = {
                "ticker": request.ticker,
                "date": request.date,
                "features": request.features,
            }

            # Generate thesis
            thesis = model.generate_thesis(inputs)

            # Validate thesis
            is_valid, errors = validator.validate(thesis)

            if not is_valid:
                logger.warning(f"Thesis validation failed: {errors}")
                raise HTTPException(
                    status_code=422,
                    detail={"message": "Invalid thesis structure", "errors": errors},
                )

            # Prepare response
            response = PredictionResponse(
                correlation_id=correlation_id,
                thesis=thesis,
                metadata={
                    "model_version": "0.1.0",
                    "validation_passed": is_valid,
                },
            )

            logger.info(f"Generated thesis for {request.ticker}")
            return response

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/validate")
    def validate_thesis(thesis: dict[str, Any]) -> dict[str, Any]:
        """Validate a thesis against schema.

        Args:
            thesis: Thesis dictionary

        Returns:
            Validation result
        """
        is_valid, errors = validator.validate(thesis)

        return {
            "valid": is_valid,
            "errors": errors if not is_valid else [],
        }

    return app
