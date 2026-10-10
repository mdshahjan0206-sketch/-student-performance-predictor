"""Request bodies. Responses are plain dicts built by the service layer, which
keeps the JSON shape in one place instead of duplicating it here."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class TargetAnalysisRequest(BaseModel):
    dataset_id: str
    target_column: str
    problem_override: Literal["auto", "classification", "regression"] = "auto"


class PreflightRequest(BaseModel):
    dataset_id: str
    target_column: str
    problem_type: Literal["classification", "regression"]
    model_choice: str


class TrainRequest(BaseModel):
    dataset_id: str
    target_column: str
    problem_type: Literal["classification", "regression"]
    model_choice: str


class NumericFilter(BaseModel):
    column: str
    min: float | None = None
    max: float | None = None


class CategoricalFilter(BaseModel):
    column: str
    values: list[str] = Field(default_factory=list)


class FilterRequest(BaseModel):
    numeric: NumericFilter | None = None
    categorical: CategoricalFilter | None = None
    drop_missing: bool = False
    limit: int = 50


class SinglePredictionRequest(BaseModel):
    model_id: str
    inputs: dict[str, Any]


class ExplainInstanceRequest(BaseModel):
    inputs: dict[str, Any]


class StudentCompareRequest(BaseModel):
    row_indices: list[int] = Field(default_factory=list)
