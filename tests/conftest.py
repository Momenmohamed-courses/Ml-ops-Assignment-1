"""Shared test fixtures: train one model per test session, in a temporary folder."""

from pathlib import Path

import pandas as pd
import pytest

from production_app.data import load_dataset, split_data
from production_app.feature_selection import select_features
from production_app.model_service import save_feature_list, save_model
from production_app.training import train_model


@pytest.fixture(scope="session")
def split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X, y = load_dataset()
    return split_data(X, y)


@pytest.fixture(scope="session")
def selected_features(split) -> list[str]:
    X_train, _, y_train, _ = split
    return select_features(X_train, y_train)


@pytest.fixture(scope="session")
def model_path(tmp_path_factory, split, selected_features) -> Path:
    X_train, _, y_train, _ = split
    pipeline = train_model(X_train[selected_features], y_train)
    path = tmp_path_factory.mktemp("models") / "pipeline.joblib"
    save_model(pipeline, path)
    save_feature_list(selected_features, path)
    return path


@pytest.fixture
def valid_patient(split, selected_features) -> dict[str, float]:
    _, X_test, _, _ = split
    return X_test[selected_features].iloc[0].to_dict()
