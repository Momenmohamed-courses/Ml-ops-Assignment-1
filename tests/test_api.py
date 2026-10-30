"""API tests: /health and /predict success and error cases, without a real server."""

import pytest
from fastapi.testclient import TestClient

from production_app.api import app
from production_app.model_service import load_model


@pytest.fixture
def client(model_path):
    with TestClient(app) as test_client:
        app.state.model = load_model(model_path)
        yield test_client


@pytest.fixture
def client_without_model():
    with TestClient(app) as test_client:
        app.state.model = None
        yield test_client


def test_health_ok_when_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_valid_input_returns_prediction(client, valid_patient):
    response = client.post("/predict", json=valid_patient)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in (0, 1)
    assert body["label"] in ("malignant", "benign")
    assert 0 <= body["probability"] <= 1


@pytest.mark.parametrize(
    "change, error_type",
    [
        ("missing", "missing"),
        ("extra", "extra_forbidden"),
        ("text", "float_type"),
    ],
)
def test_predict_invalid_input_returns_422(client, valid_patient, change, error_type):
    patient = dict(valid_patient)
    if change == "missing":
        del patient["worst area"]
    elif change == "extra":
        patient["mean radius"] = 17.99
    else:
        patient["worst area"] = "large"

    response = client.post("/predict", json=patient)
    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == error_type


def test_predict_malformed_json_returns_422(client):
    response = client.post(
        "/predict", content="{not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


def test_model_missing_returns_503(client_without_model, valid_patient):
    health = client_without_model.get("/health")
    assert health.status_code == 503
    assert health.json() == {"status": "unavailable", "model_loaded": False}

    response = client_without_model.post("/predict", json=valid_patient)
    assert response.status_code == 503
    assert response.json() == {
        "detail": "Model is not available. Please try again later."
    }
