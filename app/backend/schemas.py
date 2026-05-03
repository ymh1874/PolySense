from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field


class FeaturePrepRequest(BaseModel):
    dataset_id: str
    target_col: str
    phase: Optional[str] = None
    exclude_cols: List[str] = Field(default_factory=list)
    min_numeric_ratio: float = 0.8
    force_include_features: List[str] = Field(default_factory=list)


class CVConfig(BaseModel):
    type: Literal["RepeatedStratifiedKFold", "StratifiedKFold"] = "RepeatedStratifiedKFold"
    n_splits: int = 5
    n_repeats: int = 3
    random_state: int = 42


class TrainRequest(BaseModel):
    dataset_id: str
    target_col: str
    phase: Optional[str] = None
    selected_models: List[str] = Field(default_factory=lambda: ["Logistic Regression", "Random Forest", "Gradient Boosting"])
    exclude_cols: List[str] = Field(default_factory=list)
    min_numeric_ratio: float = 0.8
    force_include_features: List[str] = Field(default_factory=list)
    cv: CVConfig = Field(default_factory=CVConfig)
    hyperparams: Dict[str, Dict[str, object]] = Field(default_factory=dict)


class SinglePredictRequest(BaseModel):
    model_id: str
    features: Dict[str, object]


class BatchPredictRequest(BaseModel):
    model_id: str
    dataset_id: Optional[str] = None


class DatasetSummary(BaseModel):
    dataset_id: str
    name: str
    rows: int
    columns: int
    source: str


class DatasetPathRequest(BaseModel):
    path: str
    name: Optional[str] = None
