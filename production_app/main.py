from pathlib import Path

from production_app.data import load_dataset, load_prediction_input, split_data
from production_app.evaluation import evaluate_model
from production_app.feature_selection import select_features
from production_app.model_service import (
    load_model,
    predict,
    save_feature_list,
    save_model,
)
from production_app.training import train_model


def run_training() -> dict[str, float]:
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)

    baseline = train_model(X_train, y_train)
    evaluate_model(
        baseline, X_test, y_test, name=f"baseline, {X_train.shape[1]} features"
    )

    selected = select_features(X_train, y_train)
    pipeline = train_model(X_train[selected], y_train)
    metrics = evaluate_model(
        pipeline, X_test[selected], y_test, name=f"selected, {len(selected)} features"
    )

    save_model(pipeline)
    save_feature_list(selected)
    return metrics


def run_prediction(input_path: Path) -> list[str]:
    features = load_prediction_input(input_path)
    pipeline = load_model()
    return predict(pipeline, features)
