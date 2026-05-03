from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    BatchPredictRequest,
    DatasetPathRequest,
    FeaturePrepRequest,
    SinglePredictRequest,
    TrainRequest,
)
from .services import (
    list_datasets,
    load_dataset,
    predict_batch,
    predict_single,
    prepare_feature_matrix,
    preview_dataset,
    register_dataset_path,
    register_existing_processed_datasets,
    save_uploaded_dataset,
    train_models,
)
from .store import MODELS, TRAIN_RUNS

app = FastAPI(title="PolySense QuAM API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    # Register default datasets from data/processed on server boot.
    register_existing_processed_datasets()


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/api/datasets/list")
def datasets_list() -> dict[str, Any]:
    return {"datasets": list_datasets()}


@app.post("/api/datasets/register-path")
def datasets_register_path(req: DatasetPathRequest) -> dict[str, Any]:
    try:
        dataset = register_dataset_path(req.path, req.name)
        return {"dataset": dataset}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/datasets/upload")
async def datasets_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        payload = await file.read()
        if not file.filename:
            raise ValueError("Missing filename")
        meta = save_uploaded_dataset(file.filename, payload)
        return {"dataset": meta}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/datasets/{dataset_id}/preview")
def datasets_preview(dataset_id: str, limit: int = Query(default=10, ge=1, le=50)) -> dict[str, Any]:
    try:
        return preview_dataset(dataset_id, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/features/prepare")
def features_prepare(req: FeaturePrepRequest) -> dict[str, Any]:
    try:
        df = load_dataset(req.dataset_id)
        prep = prepare_feature_matrix(
            df=df,
            target_col=req.target_col,
            exclude_cols=req.exclude_cols,
            min_numeric_ratio=req.min_numeric_ratio,
            force_include_features=req.force_include_features,
        )
        return {
            "dataset_id": req.dataset_id,
            "target_col": req.target_col,
            "rows": prep["rows"],
            "feature_count": len(prep["selected_features"]),
            "selected_features": prep["selected_features"],
            "dropped_features": prep["dropped_features"],
            "class_distribution": prep["class_distribution"],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/models/train")
def models_train(req: TrainRequest) -> dict[str, Any]:
    try:
        run = train_models(
            dataset_id=req.dataset_id,
            target_col=req.target_col,
            selected_models=req.selected_models,
            exclude_cols=req.exclude_cols,
            min_numeric_ratio=req.min_numeric_ratio,
            hyperparams=req.hyperparams,
            cv_cfg=req.cv.model_dump(),
            force_include_features=req.force_include_features,
        )
        return run
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/models/compare")
def models_compare(run_id: str) -> dict[str, Any]:
    if run_id not in TRAIN_RUNS:
        raise HTTPException(status_code=404, detail="Run not found")
    run = TRAIN_RUNS[run_id]
    return {
        "run_id": run_id,
        "dataset_id": run["dataset_id"],
        "target_col": run["target_col"],
        "models": run["models"],
    }


@app.get("/api/models/{model_id}/metrics")
def model_metrics(model_id: str) -> dict[str, Any]:
    if model_id not in MODELS:
        raise HTTPException(status_code=404, detail="Model not found")
    m = MODELS[model_id]
    return {
        "model_id": model_id,
        "name": m["name"],
        "dataset_id": m["dataset_id"],
        "target_col": m["target_col"],
        "metrics": m["metrics"],
        "feature_count": len(m["features"]),
    }


@app.post("/api/predictions/single")
def predictions_single(req: SinglePredictRequest) -> dict[str, Any]:
    try:
        return predict_single(req.model_id, req.features)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/predictions/batch")
def predictions_batch(req: BatchPredictRequest) -> dict[str, Any]:
    if not req.dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id is required for batch prediction")
    try:
        df = load_dataset(req.dataset_id)
        return predict_batch(req.model_id, df)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/predictions/batch-upload")
async def predictions_batch_upload(model_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    # Allow inference on uploaded batch files without adding them to dataset catalog.
    try:
        payload = await file.read()
        temp_path = Path("/tmp") / (file.filename or "batch.csv")
        temp_path.write_bytes(payload)
        df = pd.read_csv(temp_path)
        return predict_batch(model_id, df)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
