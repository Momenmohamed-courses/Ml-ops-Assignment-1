from pathlib import Path

from production_app.data import load_dataset, load_prediction_input, split_data
from production_app.evaluation import evaluate_model
from production_app.model_service import load_model, predict, save_model
from production_app.training import train_model


def run_training() -> dict[str, float]:
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipeline = train_model(X_train, y_train)
    metrics = evaluate_model(pipeline, X_test, y_test)
    save_model(pipeline)
    return metrics


def run_prediction(input_path: Path) -> list[str]:
    features = load_prediction_input(input_path)
    pipeline = load_model()
    return predict(pipeline, features)
