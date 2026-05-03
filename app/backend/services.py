from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple
import uuid

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline

from .store import ARTIFACT_DIR, DATASETS, MODELS, PROCESSED_DATA_DIR, TRAIN_RUNS, UPLOAD_DATA_DIR


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize column labels because source CSV files sometimes include trailing spaces.
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _read_csv_robust(path: Path, nrows: int | None = None) -> pd.DataFrame:
    # Try fast parser first, then gracefully fall back to a tolerant parser.
    try:
        return pd.read_csv(path, nrows=nrows)
    except pd.errors.ParserError:
        try:
            return pd.read_csv(path, nrows=nrows, engine="python", on_bad_lines="skip")
        except Exception as exc:
            raise ValueError(f"Unable to parse CSV file: {path}. {exc}") from exc


def _resolve_dataset_path(raw_path: str) -> Path:
    # Resolve user-provided paths and support host absolute paths when running in Docker.
    candidate = Path(raw_path).expanduser()
    if candidate.exists() and candidate.is_file():
        return candidate.resolve()

    # If path includes the repo folder name, remap suffix under the runtime repo root.
    parts = list(candidate.parts)
    if PROCESSED_DATA_DIR.parent.name in parts:
        idx = parts.index(PROCESSED_DATA_DIR.parent.name)
        suffix = parts[idx + 1 :]
        remapped = PROCESSED_DATA_DIR.parent.joinpath(*suffix)
        if remapped.exists() and remapped.is_file():
            return remapped.resolve()

    # Also handle absolute paths that point outside container root but include /workspace suffix.
    workspace_hint = Path("/workspace")
    if workspace_hint.exists() and workspace_hint.is_dir() and "workspace" in parts:
        idx = parts.index("workspace")
        suffix = parts[idx + 1 :]
        remapped = workspace_hint.joinpath(*suffix)
        if remapped.exists() and remapped.is_file():
            return remapped.resolve()

    return candidate.resolve()


def _normalize_phase(phase: str | None) -> str:
    phase_value = (phase or "phaseA").strip().lower()
    if phase_value in {"a", "phasea", "phase_a"}:
        return "phaseA"
    if phase_value in {"b", "phaseb", "phase_b"}:
        return "phaseB"
    raise ValueError(f"Unknown phase: {phase}")


def _resolve_phase_target(df: pd.DataFrame, phase: str | None, requested_target: str | None) -> tuple[str, List[str]]:
    phase_value = _normalize_phase(phase)

    if phase_value == "phaseB":
        if "end_price" not in df.columns:
            raise ValueError("Phase B requires end_price so target_close_higher_prev can be derived.")
        return "target_close_higher_prev", ["trade_date", "target_close_higher_prev", "end_price", "pm_direction_up", "pm_direction_up_synth", "up_price_final", "down_price_final"]

    # Phase A prefers the original label, but falls back to the final/synth label when that is what the dataset provides.
    if requested_target in {"pm_direction_up", "pm_direction_up_final"}:
        if requested_target in df.columns:
            return requested_target, ["trade_date", requested_target, "pm_return"]
    if "pm_direction_up" in df.columns:
        return "pm_direction_up", ["trade_date", "pm_direction_up", "pm_return"]
    if "pm_direction_up_final" in df.columns:
        return "pm_direction_up_final", ["trade_date", "pm_direction_up_final", "pm_return"]
    if "pm_direction_up_synth" in df.columns:
        return "pm_direction_up_synth", ["trade_date", "pm_direction_up_synth", "pm_return"]
    raise ValueError(
        "Phase A requires a Polymarket direction column: pm_direction_up, pm_direction_up_final, or pm_direction_up_synth."
    )


def register_existing_processed_datasets() -> None:
    # Pre-register processed CSV files so user can select them in the app without upload.
    if not PROCESSED_DATA_DIR.exists():
        return

    for csv_path in sorted(PROCESSED_DATA_DIR.glob("*.csv")):
        dataset_id = f"processed::{csv_path.stem}"
        if dataset_id in DATASETS:
            continue
        DATASETS[dataset_id] = {
            "dataset_id": dataset_id,
            "name": csv_path.name,
            "path": str(csv_path),
            "source": "processed",
        }


def list_datasets() -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for dataset in DATASETS.values():
        path = Path(dataset["path"])
        if not path.exists():
            continue
        try:
            preview = _read_csv_robust(path, nrows=50)
        except Exception:
            # Skip unreadable datasets so one bad file does not break the whole list.
            continue
        preview = _clean_columns(preview)
        summaries.append(
            {
                "dataset_id": dataset["dataset_id"],
                "name": dataset["name"],
                "rows": int(sum(1 for _ in open(path, "r", encoding="utf-8")) - 1),
                "columns": int(len(preview.columns)),
                "source": dataset["source"],
            }
        )
    return sorted(summaries, key=lambda x: x["name"])


def register_dataset_path(path: str, name: str | None = None) -> Dict[str, Any]:
    target = _resolve_dataset_path(path)
    if not target.exists() or not target.is_file():
        raise ValueError(
            f"Dataset path not found: {target}. If you are in Docker, use /workspace/... paths."
        )
    if target.suffix.lower() != ".csv":
        raise ValueError("Dataset path must point to a .csv file")

    # Validate that the file is parseable at least for a small preview window.
    preview = _clean_columns(_read_csv_robust(target, nrows=20))
    if preview.empty:
        raise ValueError("Dataset appears empty after CSV parsing.")

    for dataset in DATASETS.values():
        if Path(dataset["path"]).expanduser().resolve() == target:
            return dataset

    dataset_id = f"path::{uuid.uuid4().hex[:8]}"
    DATASETS[dataset_id] = {
        "dataset_id": dataset_id,
        "name": name or target.name,
        "path": str(target),
        "source": "path",
    }
    return DATASETS[dataset_id]


def save_uploaded_dataset(filename: str, payload: bytes) -> Dict[str, Any]:
    dataset_uuid = uuid.uuid4().hex[:8]
    safe_name = filename.replace(" ", "_")
    target = UPLOAD_DATA_DIR / f"{dataset_uuid}_{safe_name}"
    target.write_bytes(payload)

    dataset_id = f"upload::{dataset_uuid}"
    DATASETS[dataset_id] = {
        "dataset_id": dataset_id,
        "name": filename,
        "path": str(target),
        "source": "upload",
    }
    return DATASETS[dataset_id]


def load_dataset(dataset_id: str) -> pd.DataFrame:
    if dataset_id not in DATASETS:
        raise ValueError(f"Dataset not found: {dataset_id}")
    path = Path(DATASETS[dataset_id]["path"])
    if not path.exists():
        raise ValueError(f"Dataset file not found: {path}")
    df = _read_csv_robust(path)
    return _clean_columns(df)


def preview_dataset(dataset_id: str, limit: int = 10) -> Dict[str, Any]:
    df = load_dataset(dataset_id)
    head = df.head(limit)
    missing = (
        (df.isna().sum() / max(len(df), 1))
        .sort_values(ascending=False)
        .round(4)
        .to_dict()
    )
    return {
        "dataset_id": dataset_id,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": list(df.columns),
        "rows": head.to_dict(orient="records"),
        "missing_ratio": missing,
    }


def prepare_feature_matrix(
    df: pd.DataFrame,
    target_col: str,
    exclude_cols: List[str],
    min_numeric_ratio: float,
    force_include_features: List[str] | None = None,
    phase: str | None = None,
) -> Dict[str, Any]:
    # Build numeric feature matrix with explicit filtering and drop reason tracking.
    force_include_features = force_include_features or []
    working = df.copy()

    resolved_target_col, phase_exclude = _resolve_phase_target(working, phase, target_col)
    target_col = resolved_target_col
    exclude_cols = list(dict.fromkeys(exclude_cols + phase_exclude))

    if target_col not in working.columns:
        # Phase B convenience: derive target_close_higher_prev from end_price if available.
        if target_col == "target_close_higher_prev" and "end_price" in working.columns:
            if "trade_date" in working.columns:
                working["trade_date"] = pd.to_datetime(working["trade_date"], errors="coerce")
                working = working.sort_values("trade_date").reset_index(drop=True)

            end_price_series = pd.to_numeric(working["end_price"], errors="coerce")
            prev_close = end_price_series.shift(1)
            working[target_col] = (end_price_series > prev_close).astype(float)
            working.loc[prev_close.isna(), target_col] = np.nan
        else:
            raise ValueError(
                f"Target column not found: {target_col}. "
                "For Phase B, use a dataset that contains end_price so target_close_higher_prev can be derived."
            )

    working[target_col] = pd.to_numeric(working[target_col], errors="coerce")
    working = working[working[target_col].notna()].copy()
    if working.empty:
        raise ValueError("No labeled rows after target cleanup.")

    selected: List[str] = []
    dropped: List[Dict[str, Any]] = []
    exclude_set = set(exclude_cols) | {target_col}

    for col in working.columns:
        if col in exclude_set:
            continue
        numeric = pd.to_numeric(working[col], errors="coerce")
        ratio = float(numeric.notna().mean())
        if ratio >= min_numeric_ratio:
            working[col] = numeric
            selected.append(col)
        else:
            dropped.append({"column": col, "reason": "low_numeric_ratio", "numeric_ratio": round(ratio, 4)})

    for forced in force_include_features:
        if forced in working.columns and forced not in selected:
            working[forced] = pd.to_numeric(working[forced], errors="coerce")
            selected.append(forced)

    if not selected:
        raise ValueError("No usable features selected.")

    X = working[selected].copy()
    X = X.fillna(X.median(numeric_only=True))
    y = working[target_col].astype(int)
    class_distribution = y.value_counts(normalize=True).round(4).to_dict()

    return {
        "X": X,
        "y": y,
        "selected_features": selected,
        "dropped_features": dropped,
        "class_distribution": class_distribution,
        "rows": int(len(working)),
    }


def _make_estimator(name: str, params: Dict[str, Any]) -> Any:
    # Build estimator instance from UI-selected model type and hyperparameters.
    if name == "Logistic Regression":
        return LogisticRegression(
            max_iter=int(params.get("max_iter", 1000)),
            class_weight=params.get("class_weight", "balanced"),
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


def _build_cv(cv_cfg: Dict[str, Any]):
    cv_type = cv_cfg.get("type", "RepeatedStratifiedKFold")
    n_splits = int(cv_cfg.get("n_splits", 5))
    random_state = int(cv_cfg.get("random_state", 42))
    if cv_type == "StratifiedKFold":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n_repeats = int(cv_cfg.get("n_repeats", 3))
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)


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
    # Train selected models with a shared prepared matrix and return comparable metrics.
    df = load_dataset(dataset_id)
    prep = prepare_feature_matrix(
        df=df,
        target_col=target_col,
        exclude_cols=exclude_cols,
        min_numeric_ratio=min_numeric_ratio,
        force_include_features=force_include_features,
    )
    X = prep["X"]
    y = prep["y"]

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_weighted": "f1_weighted",
    }

    cv = _build_cv(cv_cfg)
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    model_rows = []

    for model_name in selected_models:
        estimator = _make_estimator(model_name, hyperparams.get(model_name, {}))
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ])

        scores = cross_validate(
            pipe,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
        y_oof = cross_val_predict(pipe, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), n_jobs=-1)
        cm = confusion_matrix(y, y_oof, labels=[0, 1]).tolist()

        metrics = {
            "accuracy_mean": float(np.mean(scores["test_accuracy"])),
            "accuracy_std": float(np.std(scores["test_accuracy"], ddof=1)),
            "balanced_accuracy_mean": float(np.mean(scores["test_balanced_accuracy"])),
            "balanced_accuracy_std": float(np.std(scores["test_balanced_accuracy"], ddof=1)),
            "f1_weighted_mean": float(np.mean(scores["test_f1_weighted"])),
            "f1_weighted_std": float(np.std(scores["test_f1_weighted"], ddof=1)),
            "classification_report": classification_report(y, y_oof, zero_division=0, output_dict=True),
            "confusion_matrix": cm,
        }

        pipe.fit(X, y)
        model_id = f"mdl_{uuid.uuid4().hex[:8]}"
        artifact_path = ARTIFACT_DIR / f"{model_id}.joblib"
        joblib.dump(
            {
                "pipeline": pipe,
                "features": prep["selected_features"],
                "target_col": target_col,
            },
            artifact_path,
        )

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

    model_rows = sorted(model_rows, key=lambda r: (r["f1_weighted_mean"], r["accuracy_mean"]), reverse=True)

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


def get_model(model_id: str) -> Dict[str, Any]:
    if model_id not in MODELS:
        raise ValueError(f"Model not found: {model_id}")
    return MODELS[model_id]


def predict_single(model_id: str, features: Dict[str, Any]) -> Dict[str, Any]:
    # Predict one row by aligning incoming features to trained feature order.
    model = get_model(model_id)
    packed = joblib.load(model["artifact_path"])
    pipeline: Pipeline = packed["pipeline"]
    expected_features: List[str] = packed["features"]

    row = {f: pd.to_numeric(pd.Series([features.get(f)]), errors="coerce").iloc[0] for f in expected_features}
    X = pd.DataFrame([row], columns=expected_features)

    pred = int(pipeline.predict(X)[0])
    payload = {"model_id": model_id, "prediction": pred}

    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        probs = pipeline.predict_proba(X)[0]
        payload["probabilities"] = {"0": float(probs[0]), "1": float(probs[1])}
    return payload


def predict_batch(model_id: str, df: pd.DataFrame) -> Dict[str, Any]:
    # Predict many rows and return row-level outputs.
    model = get_model(model_id)
    packed = joblib.load(model["artifact_path"])
    pipeline: Pipeline = packed["pipeline"]
    expected_features: List[str] = packed["features"]

    X = df.copy()
    X = _clean_columns(X)
    for col in expected_features:
        if col not in X.columns:
            X[col] = np.nan
    X = X[expected_features]
    for col in expected_features:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    preds = pipeline.predict(X)
    response_rows = []
    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        probs = pipeline.predict_proba(X)
        for idx, (pred, pr) in enumerate(zip(preds, probs)):
            response_rows.append({"row": idx, "prediction": int(pred), "probability_0": float(pr[0]), "probability_1": float(pr[1])})
    else:
        for idx, pred in enumerate(preds):
            response_rows.append({"row": idx, "prediction": int(pred)})

    return {"model_id": model_id, "rows": response_rows, "count": len(response_rows)}
