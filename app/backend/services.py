# Enable modern type hinting features (like using '|' for Union types) in older Python versions.
from __future__ import annotations

# Import Path for safe file system navigation across different OS types.
from pathlib import Path
# Import type hints to define the expected structures of variables.
from typing import Dict, Any, List, Tuple
# Import uuid to generate unique randomized IDs for datasets and models.
import uuid

# Import joblib, which is used to save (serialize) and load Python objects (like trained ML models) to/from the disk.
import joblib
# Import numpy for high-performance numerical operations.
import numpy as np
# Import pandas for handling tabular data structures (DataFrames).
import pandas as pd
# Import three popular machine learning algorithms from scikit-learn.
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
# Import an imputer to fill in missing data points (e.g., replacing blanks with the median value).
from sklearn.impute import SimpleImputer
# Import a standard linear classification algorithm.
from sklearn.linear_model import LogisticRegression
# Import functions to calculate how well our models are performing.
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
# Import tools for Cross-Validation (training on different slices of data to prevent overfitting).
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
# Import Pipeline to chain data-cleaning and model-training steps together.
from sklearn.pipeline import Pipeline

# Import shared state/memory dictionaries and file directory paths from a local 'store.py' file.
from .store import ARTIFACT_DIR, DATASETS, MODELS, PROCESSED_DATA_DIR, TRAIN_RUNS, UPLOAD_DATA_DIR


# Define a helper function to clean up column names in a pandas DataFrame.
def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize column labels by stripping leading/trailing whitespace, converting them all to strings.
    df.columns = [str(c).strip() for c in df.columns]
    return df


# Define a robust helper function to read CSV files safely.
def _read_csv_robust(path: Path, nrows: int | None = None) -> pd.DataFrame:
    # Try using pandas' default, fast CSV parser first.
    try:
        return pd.read_csv(path, nrows=nrows)
    # If it fails (usually due to a malformed row with too many commas), catch the error.
    except pd.errors.ParserError:
        try:
            # Fall back to a slower but more tolerant parser that simply skips the broken rows.
            return pd.read_csv(path, nrows=nrows, engine="python", on_bad_lines="skip")
        except Exception as exc:
            # If it still fails, raise a hard error to the user.
            raise ValueError(f"Unable to parse CSV file: {path}. {exc}") from exc


# Define a helper function to figure out where a dataset file actually lives on the disk.
def _resolve_dataset_path(raw_path: str) -> Path:
    # Convert the string path to a Path object and expand any user shortcuts (like '~/' in Linux).
    candidate = Path(raw_path).expanduser()
    # If the file exists right where they said it is, return its absolute path.
    if candidate.exists() and candidate.is_file():
        return candidate.resolve()

    # The following block handles path mismatches when running the app inside a Docker container.
    parts = list(candidate.parts)
    # If the provided path includes the name of the folder where processed data is stored...
    if PROCESSED_DATA_DIR.parent.name in parts:
        # Find where that folder is in the path...
        idx = parts.index(PROCESSED_DATA_DIR.parent.name)
        # Grab everything after that folder...
        suffix = parts[idx + 1 :]
        # And stick it onto the end of the actual known system path.
        remapped = PROCESSED_DATA_DIR.parent.joinpath(*suffix)
        if remapped.exists() and remapped.is_file():
            return remapped.resolve()

    # A second Docker fallback: if the path points to a '/workspace' folder...
    workspace_hint = Path("/workspace")
    if workspace_hint.exists() and workspace_hint.is_dir() and "workspace" in parts:
        idx = parts.index("workspace")
        suffix = parts[idx + 1 :]
        remapped = workspace_hint.joinpath(*suffix)
        if remapped.exists() and remapped.is_file():
            return remapped.resolve()

    # If all else fails, just return the resolved version of what they gave us.
    return candidate.resolve()


# Helper function to standardize the phase name strings (e.g., handling typos or case differences).
def _normalize_phase(phase: str | None) -> str:
    # Default to "phaseA" if none is provided, strip spaces, make lowercase.
    phase_value = (phase or "phaseA").strip().lower()
    # Map acceptable variations to the strict "phaseA".
    if phase_value in {"a", "phasea", "phase_a"}:
        return "phaseA"
    # Map acceptable variations to the strict "phaseB".
    if phase_value in {"b", "phaseb", "phase_b"}:
        return "phaseB"
    # Reject unknown phases.
    raise ValueError(f"Unknown phase: {phase}")


# Fixed function - replace in app/backend/services.py lines 115-140

def _resolve_phase_target(df: pd.DataFrame, phase: str | None, requested_target: str | None) -> tuple[str, List[str]]:
    phase_value = _normalize_phase(phase)

    if phase_value == "phaseB":
        # Phase B relies on ending prices. If the dataset doesn't have it, we can't proceed.
        if "end_price" not in df.columns:
            raise ValueError("Phase B requires end_price so target_close_higher_prev can be derived.")
        # Return the derived target column name and a list of internal columns to exclude from training.
        # Excludes leakage columns that correlate with the same-day price movement
        return "target_close_higher_prev", [
            "trade_date", "target_close_higher_prev", "end_price", 
            "open_price", "close_price", "stock_return_daily",
            "pm_direction_up", "pm_direction_up_synth", "pm_direction_up_final",
            "up_price_final", "down_price_final"
        ]

    # For Phase A, try to use exactly what the user asked for if it exists.
    if requested_target in {"pm_direction_up", "pm_direction_up_final"}:
        if requested_target in df.columns:
            # Excludes leakage columns: open_price and close_price correlate with market direction
            return requested_target, ["trade_date", requested_target, "pm_return", "open_price", "close_price"]
    # Fallback cascade: look for known Polymarket direction columns in a specific order of preference.
    if "pm_direction_up" in df.columns:
        return "pm_direction_up", ["trade_date", "pm_direction_up", "pm_return", "open_price", "close_price"]
    if "pm_direction_up_final" in df.columns:
        return "pm_direction_up_final", ["trade_date", "pm_direction_up_final", "pm_return", "open_price", "close_price"]
    if "pm_direction_up_synth" in df.columns:
        return "pm_direction_up_synth", ["trade_date", "pm_direction_up_synth", "pm_return", "open_price", "close_price"]
    # If no appropriate target column exists, fail.
    raise ValueError(
        "Phase A requires a Polymarket direction column: pm_direction_up, pm_direction_up_final, or pm_direction_up_synth."
    )


# Automatically called on server startup to scan for and load existing datasets.
def register_existing_processed_datasets() -> None:
    # If the default folder doesn't exist, just stop.
    if not PROCESSED_DATA_DIR.exists():
        return

    # Loop through every .csv file in the folder.
    for csv_path in sorted(PROCESSED_DATA_DIR.glob("*.csv")):
        # Create a unique internal ID for it.
        dataset_id = f"processed::{csv_path.stem}"
        # If we've already registered it, skip.
        if dataset_id in DATASETS:
            continue
        # Save the dataset metadata to the global memory dictionary.
        DATASETS[dataset_id] = {
            "dataset_id": dataset_id,
            "name": csv_path.name,
            "path": str(csv_path),
            "source": "processed",
        }


# Returns a summary of all datasets the server currently knows about.
def list_datasets() -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    # Loop through our global DATASETS dictionary.
    for dataset in DATASETS.values():
        path = Path(dataset["path"])
        # If the file was deleted from the disk, skip it.
        if not path.exists():
            continue
        try:
            # Try reading just the first 50 rows to test it.
            preview = _read_csv_robust(path, nrows=50)
        except Exception:
            # Skip unreadable datasets so one bad file does not break the whole list feature.
            continue
        # Clean the column names.
        preview = _clean_columns(preview)
        # Append the summary information to our list.
        summaries.append(
            {
                "dataset_id": dataset["dataset_id"],
                "name": dataset["name"],
                # Count the total lines in the file to get the row count (subtracting 1 for the header).
                "rows": int(sum(1 for _ in open(path, "r", encoding="utf-8")) - 1),
                "columns": int(len(preview.columns)),
                "source": dataset["source"],
            }
        )
    # Sort the final list alphabetically by dataset name.
    return sorted(summaries, key=lambda x: x["name"])


# Registers a local file path as a usable dataset.
def register_dataset_path(path: str, name: str | None = None) -> Dict[str, Any]:
    # Resolve the physical path using our robust helper.
    target = _resolve_dataset_path(path)
    # Ensure it exists and is an actual file.
    if not target.exists() or not target.is_file():
        raise ValueError(
            f"Dataset path not found: {target}. If you are in Docker, use /workspace/... paths."
        )
    # Ensure it is a CSV.
    if target.suffix.lower() != ".csv":
        raise ValueError("Dataset path must point to a .csv file")

    # Read the first 20 rows to ensure it's not corrupt.
    preview = _clean_columns(_read_csv_robust(target, nrows=20))
    if preview.empty:
        raise ValueError("Dataset appears empty after CSV parsing.")

    # Check if this exact file is already registered to avoid duplicates.
    for dataset in DATASETS.values():
        if Path(dataset["path"]).expanduser().resolve() == target:
            return dataset

    # Generate a random 8-character ID.
    dataset_id = f"path::{uuid.uuid4().hex[:8]}"
    # Save it to the global dictionary.
    DATASETS[dataset_id] = {
        "dataset_id": dataset_id,
        "name": name or target.name,
        "path": str(target),
        "source": "path",
    }
    return DATASETS[dataset_id]


# Saves a file uploaded over the web API to the local disk.
def save_uploaded_dataset(filename: str, payload: bytes) -> Dict[str, Any]:
    # Generate a random ID.
    dataset_uuid = uuid.uuid4().hex[:8]
    # Replace spaces with underscores in the filename so it's safer for the OS.
    safe_name = filename.replace(" ", "_")
    # Determine exactly where to save it.
    target = UPLOAD_DATA_DIR / f"{dataset_uuid}_{safe_name}"
    # Write the raw bytes to the disk.
    target.write_bytes(payload)

    # Register it in the global dictionary.
    dataset_id = f"upload::{dataset_uuid}"
    DATASETS[dataset_id] = {
        "dataset_id": dataset_id,
        "name": filename,
        "path": str(target),
        "source": "upload",
    }
    return DATASETS[dataset_id]


# Loads a full dataset into memory based on its ID.
def load_dataset(dataset_id: str) -> pd.DataFrame:
    # Ensure the ID exists.
    if dataset_id not in DATASETS:
        raise ValueError(f"Dataset not found: {dataset_id}")
    path = Path(DATASETS[dataset_id]["path"])
    # Ensure the file still exists.
    if not path.exists():
        raise ValueError(f"Dataset file not found: {path}")
    # Read the full CSV and clean the column names.
    df = _read_csv_robust(path)
    return _clean_columns(df)


# Generates a quick summary and preview of the data for the frontend.
def preview_dataset(dataset_id: str, limit: int = 10) -> Dict[str, Any]:
    # Load the whole dataset.
    df = load_dataset(dataset_id)
    # Grab just the top X rows.
    head = df.head(limit)
    # Calculate what percentage of each column is missing (NaN) data.
    missing = (
        (df.isna().sum() / max(len(df), 1))
        .sort_values(ascending=False) # Put worst columns at the top.
        .round(4) # Round to 4 decimal places.
        .to_dict()
    )
    return {
        "dataset_id": dataset_id,
        "shape": [int(df.shape[0]), int(df.shape[1])], # Total rows and columns.
        "columns": list(df.columns),
        "rows": head.to_dict(orient="records"), # Return the top rows as a list of dictionaries.
        "missing_ratio": missing,
    }


# The most complex function: cleans and formats the data matrix (X) and target array (y) for machine learning.
def prepare_feature_matrix(
    df: pd.DataFrame,
    target_col: str,
    exclude_cols: List[str],
    min_numeric_ratio: float,
    force_include_features: List[str] | None = None,
    phase: str | None = None,
) -> Dict[str, Any]:
    # Initialize defaults and copy the dataframe so we don't alter the original memory.
    force_include_features = force_include_features or []
    working = df.copy()

    # Determine the actual target column based on the Phase logic.
    resolved_target_col, phase_exclude = _resolve_phase_target(working, phase, target_col)
    target_col = resolved_target_col
    # Combine user exclusions with phase-specific exclusions, removing duplicates.
    exclude_cols = list(dict.fromkeys(exclude_cols + phase_exclude))

    # If the target column is missing, try to create it.
    if target_col not in working.columns:
        # If we are in Phase B, we can calculate if the price closed higher than the previous day.
        if target_col == "target_close_higher_prev" and "end_price" in working.columns:
            # Sort by date first so we calculate previous days correctly.
            if "trade_date" in working.columns:
                working["trade_date"] = pd.to_datetime(working["trade_date"], errors="coerce")
                working = working.sort_values("trade_date").reset_index(drop=True)

            # Convert end prices to numbers.
            end_price_series = pd.to_numeric(working["end_price"], errors="coerce")
            # Shift the prices down 1 row to get the "previous" close.
            prev_close = end_price_series.shift(1)
            # Create the target: True (1.0) if today is > yesterday, else False (0.0).
            working[target_col] = (end_price_series > prev_close).astype(float)
            # If yesterday's price is missing, we can't calculate it, so set to NaN.
            working.loc[prev_close.isna(), target_col] = np.nan
        else:
            raise ValueError(
                f"Target column not found: {target_col}. "
                "For Phase B, use a dataset that contains end_price so target_close_higher_prev can be derived."
            )

    # Force the target column to be numeric.
    working[target_col] = pd.to_numeric(working[target_col], errors="coerce")
    # Drop any rows where we don't know the answer (the target is missing).
    working = working[working[target_col].notna()].copy()
    if working.empty:
        raise ValueError("No labeled rows after target cleanup.")

    selected: List[str] = []
    dropped: List[Dict[str, Any]] = []
    # Create a set of columns we absolutely do not want to train on (including the answer itself to prevent cheating).
    exclude_set = set(exclude_cols) | {target_col}

    # Loop through all remaining columns.
    for col in working.columns:
        if col in exclude_set:
            continue
        # Try to force the column into numbers. Strings turn into NaNs.
        numeric = pd.to_numeric(working[col], errors="coerce")
        # Check what percentage of the column successfully converted to a number.
        ratio = float(numeric.notna().mean())
        # If it meets our threshold (e.g., > 80% numeric), keep it.
        if ratio >= min_numeric_ratio:
            working[col] = numeric
            selected.append(col)
        # Otherwise, log that we dropped it.
        else:
            dropped.append({"column": col, "reason": "low_numeric_ratio", "numeric_ratio": round(ratio, 4)})

    # Add back any columns the user explicitly forced us to include.
    for forced in force_include_features:
        if forced in working.columns and forced not in selected:
            working[forced] = pd.to_numeric(working[forced], errors="coerce")
            selected.append(forced)

    if not selected:
        raise ValueError("No usable features selected.")

    # Create the final Feature Matrix (X) using only the selected columns.
    X = working[selected].copy()
    # Fill in any remaining missing values (NaNs) with the median value of that specific column.
    X = X.fillna(X.median(numeric_only=True))
    # Extract the target answers (y) and ensure they are integers (0 or 1).
    y = working[target_col].astype(int)
    # Calculate the balance between 0s and 1s.
    class_distribution = y.value_counts(normalize=True).round(4).to_dict()

    return {
        "X": X,
        "y": y,
        "selected_features": selected,
        "dropped_features": dropped,
        "class_distribution": class_distribution,
        "rows": int(len(working)),
    }


# Factory function to instantiate the specific machine learning models based on user selection.
def _make_estimator(name: str, params: Dict[str, Any]) -> Any:
    # Build estimator instance from UI-selected model type and hyperparameters.
    if name == "Logistic Regression":
        return LogisticRegression(
            max_iter=int(params.get("max_iter", 1000)),
            class_weight=params.get("class_weight", "balanced"), # Balances out unequal target classes.
            random_state=int(params.get("random_state", 42)),
        )
    if name == "Random Forest":
        max_depth = params.get("max_depth", None)
        max_depth = None if max_depth in [None, "", "None"] else int(max_depth)
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            class_weight=params.get("class_weight", "balanced"),
            max_depth=max_depth,
            random_state=int(params.get("random_state", 42)),
        )
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=int(params.get("max_depth", 3)),
            random_state=int(params.get("random_state", 42)),
        )
    raise ValueError(f"Unsupported model: {name}")


# Factory function to build the Cross-Validation splitting strategy.
def _build_cv(cv_cfg: Dict[str, Any]):
    cv_type = cv_cfg.get("type", "RepeatedStratifiedKFold")
    n_splits = int(cv_cfg.get("n_splits", 5))
    random_state = int(cv_cfg.get("random_state", 42))
    if cv_type == "StratifiedKFold":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n_repeats = int(cv_cfg.get("n_repeats", 3))
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)


# The main orchestration function for training the AI models.
def train_models(
    dataset_id: str,
    target_col: str,
    selected_models: List[str],
    exclude_cols: List[str],
    min_numeric_ratio: float,
    hyperparams: Dict[str, Dict[str, Any]],
    cv_cfg: Dict[str, Any],
    force_include_features: List[str] | None = None,
) -> Dict[str, Any]:
    
    # 1. Load the data.
    df = load_dataset(dataset_id)
    # 2. Prepare the data (clean it, drop bad columns, format X and y).
    prep = prepare_feature_matrix(
        df=df,
        target_col=target_col,
        exclude_cols=exclude_cols,
        min_numeric_ratio=min_numeric_ratio,
        force_include_features=force_include_features,
    )
    X = prep["X"]
    y = prep["y"]

    # Define the metrics we want to calculate during training.
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy", # Better for datasets where 0s and 1s are unequal.
        "f1_weighted": "f1_weighted",
    }

    # Setup the cross-validation strategy.
    cv = _build_cv(cv_cfg)
    # Generate a unique ID for this entire training event (which may contain multiple models).
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    model_rows = []

    # Loop through the list of models the user wants to train.
    for model_name in selected_models:
        # Get the specific algorithm.
        estimator = _make_estimator(model_name, hyperparams.get(model_name, {}))
        # Chain together an Imputer (to double check for missing values) and the model itself into a Pipeline.
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ])

        # 3. Perform Cross-Validation to get reliable metrics (tests the model on unseen data slices).
        scores = cross_validate(
            pipe,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1, # Use all available CPU cores.
            return_train_score=False,
        )
        # Get actual Out-Of-Fold predictions so we can build a confusion matrix.
        y_oof = cross_val_predict(pipe, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), n_jobs=-1)
        # Generate the Confusion Matrix (True Positives, False Negatives, etc.).
        cm = confusion_matrix(y, y_oof, labels=[0, 1]).tolist()

        # Compile all the final performance metrics.
        metrics = {
            "accuracy_mean": float(np.mean(scores["test_accuracy"])),
            "accuracy_std": float(np.std(scores["test_accuracy"], ddof=1)), # Calculate standard deviation to see how stable the model is.
            "balanced_accuracy_mean": float(np.mean(scores["test_balanced_accuracy"])),
            "balanced_accuracy_std": float(np.std(scores["test_balanced_accuracy"], ddof=1)),
            "f1_weighted_mean": float(np.mean(scores["test_f1_weighted"])),
            "f1_weighted_std": float(np.std(scores["test_f1_weighted"], ddof=1)),
            "classification_report": classification_report(y, y_oof, zero_division=0, output_dict=True),
            "confusion_matrix": cm,
        }

        # 4. Train the final model on 100% of the dataset so it's ready for production.
        pipe.fit(X, y)
        
        # Generate an ID for this specific model.
        model_id = f"mdl_{uuid.uuid4().hex[:8]}"
        # Define where to save the model file.
        artifact_path = ARTIFACT_DIR / f"{model_id}.joblib"
        
        # 5. Save the trained pipeline and metadata to the hard drive using joblib.
        joblib.dump(
            {
                "pipeline": pipe,
                "features": prep["selected_features"],
                "target_col": target_col,
            },
            artifact_path,
        )

        # Register the model's details in global memory.
        MODELS[model_id] = {
            "model_id": model_id,
            "run_id": run_id,
            "name": model_name,
            "dataset_id": dataset_id,
            "target_col": target_col,
            "features": prep["selected_features"],
            "artifact_path": str(artifact_path),
            "metrics": metrics,
        }

        # Append summarized results for the response payload.
        model_rows.append(
            {
                "model_id": model_id,
                "name": model_name,
                **{k: metrics[k] for k in [
                    "accuracy_mean",
                    "accuracy_std",
                    "balanced_accuracy_mean",
                    "balanced_accuracy_std",
                    "f1_weighted_mean",
                    "f1_weighted_std",
                ]},
                "confusion_matrix": cm,
            }
        )

    # Sort the models so the best performing one (highest F1 Score) is at the top of the list.
    model_rows = sorted(model_rows, key=lambda r: (r["f1_weighted_mean"], r["accuracy_mean"]), reverse=True)

    # Register the entire training run.
    TRAIN_RUNS[run_id] = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "target_col": target_col,
        "feature_count": len(prep["selected_features"]),
        "selected_features": prep["selected_features"],
        "dropped_features": prep["dropped_features"],
        "class_distribution": prep["class_distribution"],
        "models": model_rows,
    }
    return TRAIN_RUNS[run_id]


# Helper function to look up a model by its ID.
def get_model(model_id: str) -> Dict[str, Any]:
    # First look in the in-memory registry.
    if model_id in MODELS:
        return MODELS[model_id]

    # If not registered in memory, try to find a saved artifact on disk and reconstruct a minimal record.
    artifact_path = ARTIFACT_DIR / f"{model_id}.joblib"
    if artifact_path.exists():
        try:
            packed = joblib.load(artifact_path)
        except Exception as exc:
            raise ValueError(f"Model artifact found but failed to load: {artifact_path}. {exc}") from exc
        record = {
            "model_id": model_id,
            "artifact_path": str(artifact_path),
            "features": packed.get("features", []),
        }
        MODELS[model_id] = record
        return record

    raise ValueError(f"Model not found: {model_id}")


# Function to generate a single prediction (e.g., from user input in a web form).
def predict_single(model_id: str, features: Dict[str, Any]) -> Dict[str, Any]:
    # Look up the model.
    model = get_model(model_id)
    # Load the trained model from the hard drive.
    packed = joblib.load(model["artifact_path"])
    pipeline: Pipeline = packed["pipeline"]
    expected_features: List[str] = packed["features"]

    # Reconstruct the exact feature order the model expects, inserting user data or NaNs if missing.
    row = {f: pd.to_numeric(pd.Series([features.get(f)]), errors="coerce").iloc[0] for f in expected_features}
    # Create a 1-row DataFrame.
    X = pd.DataFrame([row], columns=expected_features)

    # Make the prediction (returns 0 or 1).
    pred = int(pipeline.predict(X)[0])
    payload = {"model_id": model_id, "prediction": pred}

    # If the algorithm supports calculating confidence percentages, add those to the payload.
    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        probs = pipeline.predict_proba(X)[0]
        payload["probabilities"] = {"0": float(probs[0]), "1": float(probs[1])}
    return payload


# Function to predict an entire dataset all at once.
def predict_batch(model_id: str, df: pd.DataFrame) -> Dict[str, Any]:
    # Look up and load the model.
    model = get_model(model_id)
    packed = joblib.load(model["artifact_path"])
    pipeline: Pipeline = packed["pipeline"]
    expected_features: List[str] = packed["features"]

    X = df.copy()
    X = _clean_columns(X)
    
    # Loop through the features the model was trained on. 
    # If the incoming dataset is missing any of them, create an empty column of NaNs so the pipeline doesn't crash.
    for col in expected_features:
        if col not in X.columns:
            X[col] = np.nan
            
    # Keep only the exact columns the model expects, in the exact right order.
    X = X[expected_features]
    # Ensure they are all numbers.
    for col in expected_features:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Run predictions on the whole table.
    preds = pipeline.predict(X)
    response_rows = []
    
    # Check if the model can provide confidence percentages (probabilities).
    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        probs = pipeline.predict_proba(X)
        # Package row index, prediction, and probability into a list.
        for idx, (pred, pr) in enumerate(zip(preds, probs)):
            response_rows.append({"row": idx, "prediction": int(pred), "probability_0": float(pr[0]), "probability_1": float(pr[1])})
    else:
        # Just package the row index and predictions.
        for idx, pred in enumerate(preds):
            response_rows.append({"row": idx, "prediction": int(pred)})

    return {"model_id": model_id, "rows": response_rows, "count": len(response_rows)}