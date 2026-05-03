# Import Path from the standard library to handle file paths safely across different operating systems.
from pathlib import Path
# Import Any from the typing module to indicate a variable or return type can be anything.
from typing import Any

# Import pandas, a powerful data manipulation library, often used for handling tabular data (DataFrames).
import pandas as pd
# Import the main FastAPI application class, File/UploadFile for handling file uploads, 
# HTTPException for returning web errors (like 404), and Query for parsing URL parameters.
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
# Import CORSMiddleware to allow web browsers to make requests to this API from different domains/ports.
from fastapi.middleware.cors import CORSMiddleware

# Import custom Pydantic data models (schemas) from a local 'schemas.py' file. 
# These ensure the incoming data from users matches the expected formats.
from .schemas import (
    BatchPredictRequest,
    DatasetPathRequest,
    FeaturePrepRequest,
    SinglePredictRequest,
    TrainRequest,
)

# Import custom business logic functions from a local 'services.py' file. 
# This keeps the API file clean by hiding the complex machine learning and file processing code.
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

# Import shared memory dictionaries (MODELS and TRAIN_RUNS) from a local 'store.py' file to store data temporarily.
from .store import MODELS, TRAIN_RUNS

# Create the main FastAPI application instance, giving it a title and a version number for the documentation.
app = FastAPI(title="PolySense QuAM API", version="0.1.0")

# Add the CORS middleware to the application.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow requests from any website (origin).
    allow_credentials=True, # Allow cookies or authentication headers to be sent.
    allow_methods=["*"], # Allow any HTTP method (GET, POST, PUT, DELETE, etc.).
    allow_headers=["*"], # Allow any HTTP headers in the request.
)


# Create an event listener that runs this specific function just before the server starts up.
@app.on_event("startup")
def startup() -> None:
    # Register default datasets from data/processed on server boot.
    # Calls a service function to load previously processed datasets into the application's memory.
    register_existing_processed_datasets()


# Define a GET endpoint at "/api/health" to check if the API is running.
@app.get("/api/health")
def health() -> dict[str, Any]:
    # Return a simple JSON response confirming the server is operational.
    return {"status": "ok"}


# Define a GET endpoint at "/api/datasets/list" to retrieve all available datasets.
@app.get("/api/datasets/list")
def datasets_list() -> dict[str, Any]:
    # Call the list_datasets service function and return its result under the "datasets" key.
    return {"datasets": list_datasets()}


# Define a POST endpoint at "/api/datasets/register-path" to register a dataset already on the server's disk.
@app.post("/api/datasets/register-path")
# It expects a JSON body matching the DatasetPathRequest schema.
def datasets_register_path(req: DatasetPathRequest) -> dict[str, Any]:
    # Start a try block to catch any errors during registration.
    try:
        # Call the service function to register the dataset using the path and name provided in the request.
        dataset = register_dataset_path(req.path, req.name)
        # Return the registered dataset information.
        return {"dataset": dataset}
    # Catch any general Python Exceptions.
    except Exception as exc:
        # Raise an HTTP 400 (Bad Request) error, passing the exception's message to the user.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# Define a POST endpoint at "/api/datasets/upload" for users to upload a new dataset file.
@app.post("/api/datasets/upload")
# Define an asynchronous function that expects a file upload (FastAPI handles the file parsing).
async def datasets_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    # Start a try block to handle potential file reading or saving errors.
    try:
        # Read the entire uploaded file into memory asynchronously.
        payload = await file.read()
        # Check if the uploaded file has a filename.
        if not file.filename:
            # If not, throw a ValueError to be caught below.
            raise ValueError("Missing filename")
        # Pass the filename and the file's raw bytes to the service function to save it.
        meta = save_uploaded_dataset(file.filename, payload)
        # Return the metadata of the newly saved dataset.
        return {"dataset": meta}
    # Catch any errors (like missing filename or disk write errors).
    except Exception as exc:
        # Raise an HTTP 400 error, returning the error detail to the client.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# Define a GET endpoint to preview a specific dataset, taking the dataset ID from the URL path.
@app.get("/api/datasets/{dataset_id}/preview")
# Expect 'dataset_id' from the path, and 'limit' from the query string (defaults to 10, between 1 and 50).
def datasets_preview(dataset_id: str, limit: int = Query(default=10, ge=1, le=50)) -> dict[str, Any]:
    # Start a try block.
    try:
        # Call the service function to get a preview of the dataset, limited to the requested number of rows.
        return preview_dataset(dataset_id, limit=limit)
    # Catch any errors (like a missing dataset).
    except Exception as exc:
        # Raise an HTTP 400 error.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# Define a POST endpoint at "/api/features/prepare" to prepare the dataset for machine learning.
@app.post("/api/features/prepare")
# It expects a JSON body matching the FeaturePrepRequest schema.
def features_prepare(req: FeaturePrepRequest) -> dict[str, Any]:
    # Start a try block.
    try:
        # Load the dataset into a pandas DataFrame using its ID.
        df = load_dataset(req.dataset_id)
        # Call the service function to clean and select the mathematical features for the model.
        prep = prepare_feature_matrix(
            df=df, # Pass the dataframe.
            target_col=req.target_col, # Pass the column we want to predict.
            exclude_cols=req.exclude_cols, # Pass columns to ignore.
            min_numeric_ratio=req.min_numeric_ratio, # Rule for how much numeric data a column must have.
            force_include_features=req.force_include_features, # Specific columns to force into the model.
        )
        # Return a dictionary containing the results of the preparation phase.
        return {
            "dataset_id": req.dataset_id, # The ID of the dataset used.
            "target_col": req.target_col, # The target variable we want to predict.
            "rows": prep["rows"], # The total number of rows processed.
            "feature_count": len(prep["selected_features"]), # How many features (columns) were kept.
            "selected_features": prep["selected_features"], # The list of kept feature names.
            "dropped_features": prep["dropped_features"], # The list of rejected feature names.
            "class_distribution": prep["class_distribution"], # If classification, the balance of target labels.
        }
    # Catch processing errors.
    except Exception as exc:
        # Raise an HTTP 400 error.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# Define a POST endpoint at "/api/models/train" to start training machine learning models.
@app.post("/api/models/train")
# It expects a JSON body matching the TrainRequest schema.
def models_train(req: TrainRequest) -> dict[str, Any]:
    # Start a try block.
    try:
        # Call the service function to train the models based on the provided configuration.
        run = train_models(
            dataset_id=req.dataset_id, # Which dataset to train on.
            target_col=req.target_col, # What we are predicting.
            selected_models=req.selected_models, # Which algorithms to use (e.g., Random Forest, SVM).
            exclude_cols=req.exclude_cols, # Columns to ignore.
            min_numeric_ratio=req.min_numeric_ratio, # Data quality threshold.
            hyperparams=req.hyperparams, # Tweaks for the machine learning algorithms.
            cv_cfg=req.cv.model_dump(), # Cross-validation configuration, converted to a dictionary.
            force_include_features=req.force_include_features, # Columns forced into the training set.
        )
        # Return the results of the training run (metrics, model IDs, etc.).
        return run
    # Catch training errors.
    except Exception as exc:
        # Raise an HTTP 400 error.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# Define a GET endpoint at "/api/models/compare" to compare the results of a specific training run.
@app.get("/api/models/compare")
# Expects a 'run_id' passed as a query parameter in the URL.
def models_compare(run_id: str) -> dict[str, Any]:
    # Check if the requested run_id exists in our global TRAIN_RUNS dictionary.
    if run_id not in TRAIN_RUNS:
        # If it doesn't exist, raise an HTTP 404 (Not Found) error.
        raise HTTPException(status_code=404, detail="Run not found")
    # Retrieve the run details from the dictionary.
    run = TRAIN_RUNS[run_id]
    # Return a summarized dictionary of the run, including the models trained during that run.
    return {
        "run_id": run_id, # The ID of the training run.
        "dataset_id": run["dataset_id"], # The dataset used.
        "target_col": run["target_col"], # The target predicted.
        "models": run["models"], # A list of the models generated and their metrics.
    }


# Define a GET endpoint to fetch detailed metrics for one specific trained model.
@app.get("/api/models/{model_id}/metrics")
# Expects 'model_id' from the URL path.
def model_metrics(model_id: str) -> dict[str, Any]:
    # Check if the requested model_id exists in our global MODELS dictionary.
    if model_id not in MODELS:
        # If not, raise an HTTP 404 (Not Found) error.
        raise HTTPException(status_code=404, detail="Model not found")
    # Retrieve the specific model's metadata from the dictionary.
    m = MODELS[model_id]
    # Return a dictionary of the model's performance details.
    return {
        "model_id": model_id, # The unique ID of the model.
        "name": m["name"], # The readable name of the model algorithm.
        "dataset_id": m["dataset_id"], # The dataset it was trained on.
        "target_col": m["target_col"], # The column it predicts.
        "metrics": m["metrics"], # The performance scores (e.g., accuracy, f1 score).
        "feature_count": len(m["features"]), # How many features the model uses to make predictions.
    }


# Define a POST endpoint at "/api/predictions/single" to predict the outcome for a single data point.
@app.post("/api/predictions/single")
# Expects a JSON body matching the SinglePredictRequest schema.
def predictions_single(req: SinglePredictRequest) -> dict[str, Any]:
    # Start a try block.
    try:
        # Call the service function to generate a prediction using the specified model and provided features.
        return predict_single(req.model_id, req.features)
    # Catch any prediction errors (e.g., missing features).
    except Exception as exc:
        # Raise an HTTP 400 error.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# Define a POST endpoint at "/api/predictions/batch" to predict outcomes for an entire dataset.
@app.post("/api/predictions/batch")
# Expects a JSON body matching the BatchPredictRequest schema.
def predictions_batch(req: BatchPredictRequest) -> dict[str, Any]:
    # Check if a dataset_id was actually provided in the request.
    if not req.dataset_id:
        # If not, immediately raise a 400 error because we don't know what data to predict on.
        raise HTTPException(status_code=400, detail="dataset_id is required for batch prediction")
    # Start a try block.
    try:
        # Load the specified dataset from disk into a pandas DataFrame.
        df = load_dataset(req.dataset_id)
        # Call the service function to run predictions on the whole DataFrame using the chosen model.
        return predict_batch(req.model_id, df)
    # Catch any errors.
    except Exception as exc:
        # Raise an HTTP 400 error.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# Define a POST endpoint at "/api/predictions/batch-upload" for batch predicting on a newly uploaded CSV.
@app.post("/api/predictions/batch-upload")
# Expects 'model_id' from the URL query and a CSV file uploaded in the request body.
async def predictions_batch_upload(model_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    # Allow inference on uploaded batch files without adding them to dataset catalog.
    # Start a try block.
    try:
        # Read the raw bytes of the uploaded file asynchronously.
        payload = await file.read()
        # Create a temporary file path in the system's /tmp folder, using the uploaded filename (or "batch.csv" as fallback).
        temp_path = Path("/tmp") / (file.filename or "batch.csv")
        # Write the raw bytes we just read into this temporary file on the disk.
        temp_path.write_bytes(payload)
        # Use pandas to read the temporary CSV file into a DataFrame.
        df = pd.read_csv(temp_path)
        # Call the service function to run predictions on this new DataFrame.
        return predict_batch(model_id, df)
    # Catch any errors (like invalid CSV format).
    except Exception as exc:
        # Raise an HTTP 400 error.
        raise HTTPException(status_code=400, detail=str(exc)) from exc