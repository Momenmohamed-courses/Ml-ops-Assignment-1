"""Streamlit UI: collect the selected features and ask the FastAPI service for a prediction."""

import json
from pathlib import Path

import requests
import streamlit as st

from production_app.config import settings
from production_app.schemas import PatientFeatures

SAMPLE_FILE = Path("examples/sample_request.json")
FEATURES = [field.alias for field in PatientFeatures.model_fields.values()]


def load_sample() -> dict[str, float]:
    return json.loads(SAMPLE_FILE.read_text())


def to_number(text: str) -> float | str:
    try:
        return float(text)
    except ValueError:
        return text


def show_prediction(patient: dict[str, float | str]) -> None:
    try:
        response = requests.post(
            f"{settings.api_url}/predict",
            json=patient,
            timeout=settings.request_timeout,
        )
    except requests.Timeout:
        st.error(
            f"The API didn't answer within {settings.request_timeout:g} seconds. Try again."
        )
        return
    except requests.ConnectionError:
        st.error(f"Can't reach the API at {settings.api_url}. Is it running?")
        return

    if response.status_code == 200:
        result = response.json()
        message = (
            f"Prediction: {result['label']} ({result['probability']:.1%} confidence)"
        )
        if result["label"] == "malignant":
            st.error(message)
        else:
            st.success(message)
        st.caption(
            f"Predicted class: {result['predicted_class']} (0 = malignant, 1 = benign)"
        )
    elif response.status_code == 422:
        st.error("Some values were rejected by the API:")
        detail = response.json().get("detail")
        if isinstance(detail, list):
            for error in detail:
                st.write(f"- {error['loc'][-1]}: {error['msg']}")
        else:
            st.write(detail)
    elif response.status_code == 503:
        st.error("The model isn't available right now. Train it, then restart the API.")
    else:
        st.error(f"Unexpected answer from the API (HTTP {response.status_code}).")


st.title("Breast cancer classifier")
st.write(
    f"Enter the {len(FEATURES)} tumor measurements, or fill in sample values, then click Predict. "
    f"Predictions come from the API at `{settings.api_url}`."
)
st.caption("Learning project, not a medical tool.")

for name in FEATURES:
    st.session_state.setdefault(name, "")

if st.button("Fill sample values"):
    for name, value in load_sample().items():
        st.session_state[name] = str(value)

raw = {
    name: st.text_input(name, key=name, placeholder="required").strip()
    for name in FEATURES
}

if st.button("Predict", type="primary"):
    empty = [name for name, text in raw.items() if text == ""]
    if empty:
        st.error(f"Please fill in: {', '.join(empty)}")
    else:
        show_prediction({name: to_number(text) for name, text in raw.items()})
