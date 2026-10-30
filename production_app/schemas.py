from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Feature = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


class PatientFeatures(BaseModel):

    model_config = ConfigDict(extra="forbid")

    mean_texture: Feature = Field(alias="mean texture")
    mean_concave_points: Feature = Field(alias="mean concave points")
    radius_error: Feature = Field(alias="radius error")
    fractal_dimension_error: Feature = Field(alias="fractal dimension error")
    worst_radius: Feature = Field(alias="worst radius")
    worst_texture: Feature = Field(alias="worst texture")
    worst_area: Feature = Field(alias="worst area")
    worst_smoothness: Feature = Field(alias="worst smoothness")
    worst_concavity: Feature = Field(alias="worst concavity")
    worst_concave_points: Feature = Field(alias="worst concave points")
    worst_symmetry: Feature = Field(alias="worst symmetry")


class PredictionResponse(BaseModel):
    predicted_class: Literal[0, 1]
    label: Literal["malignant", "benign"]
    probability: float = Field(ge=0, le=1)


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
