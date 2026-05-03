# Import necessary type hints from Python's standard typing library.
# These tell Python exactly what kind of data is expected inside variables and lists.
# Dict (dictionary), List (array), Optional (can be a value or None), Literal (must be one of a specific set of choices).
from typing import Dict, List, Optional, Literal

# Import BaseModel (the foundational class for building Pydantic schemas) 
# and Field (used to add extra rules, default values, or metadata to specific attributes).
from pydantic import BaseModel, Field


# Define a schema for requests to the "/api/features/prepare" endpoint.
# Inheriting from BaseModel tells Pydantic to enforce these rules.
class FeaturePrepRequest(BaseModel):
    # A required string field: the ID of the dataset you want to prepare.
    dataset_id: str
    # A required string field: the name of the column you are trying to predict.
    target_col: str
    # An optional string field: used to label the process (e.g., "training"), defaults to None.
    phase: Optional[str] = None
    # A list of strings: columns to drop. 'default_factory=list' ensures every new request 
    # gets its own fresh empty list if the user doesn't provide one, preventing data leakage between requests.
    exclude_cols: List[str] = Field(default_factory=list)
    # A float field: strictness for numeric data. Requires 80% (0.8) numeric values to keep a column by default.
    min_numeric_ratio: float = 0.8
    # A list of strings: columns you want to force the model to use, ignoring the rules above. Defaults to an empty list.
    force_include_features: List[str] = Field(default_factory=list)


# Define a schema specifically for Cross-Validation (CV) settings, used during model training.
class CVConfig(BaseModel):
    # A string field restricted to two exact values. If the user sends anything else, it fails. 
    # Defaults to "RepeatedStratifiedKFold".
    type: Literal["RepeatedStratifiedKFold", "StratifiedKFold"] = "RepeatedStratifiedKFold"
    # An integer field: how many chunks (folds) to split the dataset into. Defaults to 5.
    n_splits: int = 5
    # An integer field: how many times to repeat the CV process. Defaults to 3.
    n_repeats: int = 3
    # An integer field: a fixed seed for the random number generator so results are reproducible. Defaults to 42.
    random_state: int = 42


# Define a schema for requests to the "/api/models/train" endpoint.
class TrainRequest(BaseModel):
    # Required string: the dataset to train on.
    dataset_id: str
    # Required string: the column we want to predict.
    target_col: str
    # Optional string: defaults to None.
    phase: Optional[str] = None
    # A list of strings: which algorithms to train. Defaults to three common models using a lambda (anonymous) function.
    selected_models: List[str] = Field(default_factory=lambda: ["Logistic Regression", "Random Forest", "Gradient Boosting"])
    # A list of strings: columns to ignore during training. Defaults to an empty list.
    exclude_cols: List[str] = Field(default_factory=list)
    # A float field: numeric data threshold, same as in FeaturePrepRequest.
    min_numeric_ratio: float = 0.8
    # A list of strings: columns forced into the training data. Defaults to an empty list.
    force_include_features: List[str] = Field(default_factory=list)
    # A nested schema: injects the rules from the CVConfig class above. Defaults to standard CVConfig settings.
    cv: CVConfig = Field(default_factory=CVConfig)
    # A dictionary of dictionaries: allows users to send custom tweaks (hyperparameters) for specific models 
    # (e.g., {"Random Forest": {"max_depth": 5}}). Defaults to an empty dictionary.
    hyperparams: Dict[str, Dict[str, object]] = Field(default_factory=dict)


# Define a schema for requests to the "/api/predictions/single" endpoint.
class SinglePredictRequest(BaseModel):
    # Required string: the unique ID of the already-trained model you want to use.
    model_id: str
    # Required dictionary: the actual data point to predict on. Keys are feature names (strings), 
    # values are the actual data (objects, meaning they can be text, numbers, etc.).
    features: Dict[str, object]


# Define a schema for requests to the "/api/predictions/batch" endpoint.
class BatchPredictRequest(BaseModel):
    # Required string: the unique ID of the trained model to use.
    model_id: str
    # Optional string: the ID of the dataset to run predictions on. 
    # It is optional because the API has a separate endpoint where you can upload a CSV directly instead of using an ID.
    dataset_id: Optional[str] = None


# Define a schema detailing the structure of how a dataset's metadata is returned to the user.
# (This is often used for formatting responses rather than incoming requests).
class DatasetSummary(BaseModel):
    # The unique system ID for the dataset.
    dataset_id: str
    # The human-readable name of the dataset.
    name: str
    # An integer showing how many rows of data it contains.
    rows: int
    # An integer showing how many columns (features) it contains.
    columns: int
    # A string showing how the dataset got there (e.g., "uploaded", "local_file").
    source: str


# Define a schema for requests to the "/api/datasets/register-path" endpoint.
class DatasetPathRequest(BaseModel):
    # A required string: the absolute or relative file path on the server where the dataset lives.
    path: str
    # An optional string: a custom name for the dataset. If not provided, the server will likely guess it from the filename.
    name: Optional[str] = None