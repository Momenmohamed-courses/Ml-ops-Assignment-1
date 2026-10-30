"""Model workflow tests: selection, saved artifact, feature list, schema agreement."""

import pytest

from production_app.model_service import (
    ModelNotFoundError,
    load_feature_list,
    load_model,
    predict_one,
)
from production_app.schemas import PatientFeatures


def test_feature_selection_keeps_fewer_real_features(split, selected_features):
    X_train, _, _, _ = split
    assert 0 < len(selected_features) < X_train.shape[1]
    assert set(selected_features) <= set(X_train.columns)


def test_saved_model_loads_and_predicts_without_retraining(
    model_path, selected_features, valid_patient
):
    model = load_model(model_path)
    assert list(model.feature_names_in_) == selected_features

    result = predict_one(model, valid_patient)
    assert result["predicted_class"] in (0, 1)
    assert result["label"] in ("malignant", "benign")
    assert 0 <= result["probability"] <= 1


def test_load_model_missing_file_raises(tmp_path):
    with pytest.raises(ModelNotFoundError):
        load_model(tmp_path / "missing.joblib")


def test_feature_list_round_trip(model_path, selected_features):
    assert load_feature_list(model_path) == selected_features


def test_api_schema_matches_model_features(model_path):
    schema_names = [field.alias for field in PatientFeatures.model_fields.values()]
    assert schema_names == load_feature_list(model_path)
