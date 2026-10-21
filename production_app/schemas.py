from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Feature = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


class PatientFeatures(BaseModel):

    model_config = ConfigDict(extra="forbid")

    mean_radius: Feature = Field(alias="mean radius")
    mean_texture: Feature = Field(alias="mean texture")
    mean_perimeter: Feature = Field(alias="mean perimeter")
    mean_area: Feature = Field(alias="mean area")
    mean_smoothness: Feature = Field(alias="mean smoothness")
    mean_compactness: Feature = Field(alias="mean compactness")
    mean_concavity: Feature = Field(alias="mean concavity")
    mean_concave_points: Feature = Field(alias="mean concave points")
    mean_symmetry: Feature = Field(alias="mean symmetry")
    mean_fractal_dimension: Feature = Field(alias="mean fractal dimension")

    radius_error: Feature = Field(alias="radius error")
    texture_error: Feature = Field(alias="texture error")
    perimeter_error: Feature = Field(alias="perimeter error")
    area_error: Feature = Field(alias="area error")
    smoothness_error: Feature = Field(alias="smoothness error")
    compactness_error: Feature = Field(alias="compactness error")
    concavity_error: Feature = Field(alias="concavity error")
    concave_points_error: Feature = Field(alias="concave points error")
    symmetry_error: Feature = Field(alias="symmetry error")
    fractal_dimension_error: Feature = Field(alias="fractal dimension error")

    worst_radius: Feature = Field(alias="worst radius")
    worst_texture: Feature = Field(alias="worst texture")
    worst_perimeter: Feature = Field(alias="worst perimeter")
    worst_area: Feature = Field(alias="worst area")
    worst_smoothness: Feature = Field(alias="worst smoothness")
    worst_compactness: Feature = Field(alias="worst compactness")
    worst_concavity: Feature = Field(alias="worst concavity")
    worst_concave_points: Feature = Field(alias="worst concave points")
    worst_symmetry: Feature = Field(alias="worst symmetry")
    worst_fractal_dimension: Feature = Field(alias="worst fractal dimension")


class PredictionResponse(BaseModel):
    predicted_class: Literal[0, 1]
    label: Literal["malignant", "benign"]
    probability: float = Field(ge=0, le=1)


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
