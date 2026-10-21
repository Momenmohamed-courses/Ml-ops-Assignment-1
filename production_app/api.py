from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response, status

from production_app.logging_config import get_logger, setup_logging
from production_app.model_service import (
    InvalidInputError,
    ModelNotFoundError,
    load_model,
    predict_one,
)
from production_app.schemas import HealthResponse, PatientFeatures, PredictionResponse

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    try:
        app.state.model = load_model()
    except ModelNotFoundError:
        app.state.model = None
        logger.warning("API started without a model: /predict will return 503")
    yield


app = FastAPI(title="Breast Cancer Classifier API", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health(response: Response) -> HealthResponse:
    if app.state.model is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(status="unavailable", model_loaded=False)
    return HealthResponse(status="ok", model_loaded=True)


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures) -> PredictionResponse:
    if app.state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not available. Please try again later.",
        )
    try:
        result = predict_one(app.state.model, patient.model_dump(by_alias=True))
    except InvalidInputError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid prediction input.",
        ) from error
    return PredictionResponse(**result)
