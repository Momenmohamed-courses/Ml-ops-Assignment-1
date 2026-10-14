import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from production_app.logging_config import get_logger

logger = get_logger(__name__)

MALIGNANT = 0  # in this dataset: 0 = malignant, 1 = benign


def evaluate_model(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label=MALIGNANT),
        "recall": recall_score(y_test, y_pred, pos_label=MALIGNANT),
        "f1": f1_score(y_test, y_pred, pos_label=MALIGNANT),
    }

    logger.info(
        "Evaluation (malignant class): accuracy=%.3f precision=%.3f recall=%.3f f1=%.3f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )
    logger.info(
        "Confusion matrix (rows=real, cols=predicted, malignant/benign): %s",
        confusion_matrix(y_test, y_pred).tolist(),
    )
    return metrics
