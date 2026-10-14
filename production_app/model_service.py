from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from production_app.config import settings
from production_app.logging_config import get_logger

logger = get_logger(__name__)

LABELS = {0: "malignant", 1: "benign"}


class ModelNotFoundError(FileNotFoundError):
    """No saved model exists at the expected path."""


class InvalidInputError(ValueError):
    """Prediction input does not match what the model was trained on."""


def save_model(pipeline: Pipeline, path: Path = settings.model_path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info("Model saved to %s", path)
    return path


def load_model(path: Path = settings.model_path) -> Pipeline:
    if not path.exists():
        logger.error("No saved model at %s", path)
        raise ModelNotFoundError(f"No saved model at {path}. Run training first.")
    pipeline = joblib.load(path)
    logger.info("Model loaded from %s", path)
    return pipeline


def validate_input(pipeline: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
    expected = list(pipeline.feature_names_in_)
    missing = [col for col in expected if col not in features.columns]

    if missing:
        logger.error("Invalid input: missing columns %s", missing)
        raise InvalidInputError(f"Invalid prediction input. Missing columns: {missing}")

    if features.empty:
        logger.error("Invalid input: no rows")
        raise InvalidInputError("Invalid prediction input: no rows to predict")
    try:
        clean = features[expected].astype(float)
    except ValueError as error:
        logger.error("Invalid input: non-numeric values")
        raise InvalidInputError(
            "Invalid prediction input: values must be numbers"
        ) from error

    if clean.isnull().any().any():
        logger.error("Invalid input: missing values")
        raise InvalidInputError("Invalid prediction input: contains missing values")

    return clean


def predict(pipeline: Pipeline, features: pd.DataFrame) -> list[str]:
    clean = validate_input(pipeline, features)
    labels = [LABELS[int(value)] for value in pipeline.predict(clean)]
    logger.info("Predicted %d row(s): %s", len(labels), labels)
    return labels
