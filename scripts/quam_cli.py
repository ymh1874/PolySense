#!/usr/bin/env python3
"""
PolySense CLI: Command-line interface for training and using ML models for market direction prediction.

This script provides a complete CLI wrapper around the FastAPI backend, allowing users to:
1. Explore and prepare datasets
2. Train machine learning models with customizable hyperparameters
3. Make single and batch predictions using trained models
4. Run all functionality locally without needing the web interface

The CLI supports two main prediction phases:
- Phase A: Predicts Polymarket direction (up/down) based on market and sentiment data
- Phase B: Predicts stock price direction (close higher than previous day) based on market data

Architecture:
- Backend logic lives in app/backend/services.py (dataset management, feature prep, model training)
- This CLI script imports and calls those functions directly
- Results are output as JSON for easy parsing by other tools

Key Features:
- Flexible argument handling: use --phase or --dataset-id, both work interchangeably
- Individual hyperparameter flags: --max-depth, --n-estimators, --learning-rate
- Raw JSON hyperparameter support for advanced users: --hyperparams '{...}'
- Automatic leakage exclusion: prevents using columns that directly predict the target
- Cross-validation with customizable splits and repeats
- Model persistence: trained models saved to disk for reuse

Usage Examples:
  # List all available datasets
  python3 scripts/quam_cli.py list

  # Preview a dataset (see columns and samples)
  python3 scripts/quam_cli.py preview processed::phase_a_prepared --limit 20

  # Prepare features (analyze without training)
  python3 scripts/quam_cli.py prepare --phase phaseA

  # Train with default settings
  python3 scripts/quam_cli.py train --phase phaseA

  # Train with custom hyperparameters
  python3 scripts/quam_cli.py train --phase phaseA --max-depth 5 --n-estimators 150 --learning-rate 0.08

  # Train multiple models
  python3 scripts/quam_cli.py train --phase phaseB --models "Random Forest" "Gradient Boosting" --max-depth 6

  # Make a single prediction
  python3 scripts/quam_cli.py predict-single mdl_abc12345 '{"volume": 1000000, "gap_pct": 2.5}'

  # Make batch predictions on a dataset
  python3 scripts/quam_cli.py predict-batch mdl_abc12345 processed::phase_b_prepared

For more information on each command, run:
  python3 scripts/quam_cli.py <command> --help
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path so imports work when running the script directly.
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from app.backend.services import (
    register_existing_processed_datasets,
    list_datasets,
    preview_dataset,
    load_dataset,
    _normalize_phase,
    _resolve_phase_target,
    prepare_feature_matrix,
    train_models,
    predict_single,
    predict_batch,
)


def cmd_list(args: argparse.Namespace) -> int:
    """List all registered datasets available for training or prediction.
    
    Datasets can come from multiple sources:
    - processed/ folder: Pre-cleaned datasets (phase_a_prepared, phase_b_prepared)
    - uploads/: User-uploaded CSV files
    - path registry: External datasets registered via --dataset-id
    
    Output: JSON array containing dataset metadata (name, rows, columns, source)
    """
    # Scan data/processed/ folder and register any new datasets found there
    register_existing_processed_datasets()
    
    # Fetch all registered datasets with their metadata
    ds = list_datasets()
    
    # Output as JSON
    print(json.dumps({"datasets": ds}, indent=2))
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    """Display a preview of a dataset: column names, sample rows, and missing data analysis.
    
    This command helps explore a dataset before training to understand:
    - What columns are available
    - Data types and ranges
    - How much data is missing (NaN values)
    - Class imbalance
    
    Args (from CLI):
        dataset_id: The dataset to preview (e.g., "processed::phase_a_prepared")
        --limit: Max number of sample rows to show (default: 10)
        
    Output: JSON object with shape, columns, sample rows, and missing value ratios
    """
    # Get dataset ID from positional argument
    dataset_id = args.dataset_id
    
    # Register all processed datasets to ensure they're available for lookup
    register_existing_processed_datasets()
    
    try:
        # Fetch the dataset preview
        prev = preview_dataset(dataset_id, limit=args.limit)
        # Output as JSON
        print(json.dumps(prev, indent=2, default=str))
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


def _select_prepared_for_phase(phase: str) -> str | None:
    """Find and return the prepared dataset ID for a given phase.
    
    Prepared datasets are CSV files in data/processed/ that have been cleaned and
    feature-engineered already. Phase A uses phase_a_prepared.csv, Phase B uses phase_b_prepared.csv.
    
    Args:
        phase: Phase name, one of "phaseA", "phaseB", "a", "A", "phase_a", etc.
               The function normalizes the input.
               
    Returns:
        Dataset ID string like "processed::phase_a_prepared" if found, None otherwise.
        
    Note:
        This function queries the registered datasets list, so register_existing_processed_datasets()
        should be called first to populate the cache.
    """
    # Normalize phase input (handle "phaseA", "phasea", "phase_a", "a", etc.)
    phase_val = _normalize_phase(phase)
    
    # Map normalized phase to the expected prepared dataset filename
    candidate = f"processed::phase_a_prepared" if phase_val == "phaseA" else f"processed::phase_b_prepared"
    
    # Check if the candidate dataset is registered by loading all datasets
    datasets = {d["dataset_id"]: d for d in list_datasets()}
    
    # Return the dataset ID if it exists, otherwise None
    return candidate if candidate in datasets else None


def _infer_phase_from_dataset_id(dataset_id: str) -> str | None:
    """Infer the phase (A or B) from a dataset ID or filename by pattern matching.
    
    This is a convenience function that allows users to omit the --phase argument
    if they provide a dataset ID, since the phase can often be inferred from the filename.
    
    Args:
        dataset_id: A dataset identifier, e.g., "processed::phase_a_prepared" or
                   "path::abc1234" where the path contains "phase_a" or "phase_b"
                   
    Returns:
        Phase name string: "phaseA" or "phaseB" if the pattern matches, None otherwise.
        
    Examples:
        "processed::phase_a_prepared" -> "phaseA"
        "/data/phase_b_data.csv" -> "phaseB"
        "path::unknown_file" -> None
    """
    if not dataset_id:
        return None
    
    # Convert to lowercase for case-insensitive pattern matching
    key = dataset_id.lower()
    
    # Check for Phase A patterns (phase_a, phasea)
    if "phase_a" in key or "phasea" in key:
        return "phaseA"
    
    # Check for Phase B patterns (phase_b, phaseb)
    if "phase_b" in key or "phaseb" in key:
        return "phaseB"
    
    # No recognized phase pattern found
    return None


def cmd_prepare(args: argparse.Namespace) -> int:
    """Prepare features for training by analyzing a dataset and resolving which columns to exclude.
    
    This command:
    1. Loads the specified dataset
    2. Determines the phase (A or B) and target column
    3. Identifies which columns are leakage (and must be excluded)
    4. Selects numeric features for training
    5. Reports feature statistics and class balance
    """
    # Register all processed datasets so they can be looked up by ID
    register_existing_processed_datasets()
    
    # User error handling: allow users to accidentally pass a dataset id into --phase
    # Detect by checking for dataset-id markers (::, .csv, /) and swap if found
    if args.phase and ("::" in args.phase or args.phase.endswith(".csv") or "/" in args.phase):
        args.dataset_id = args.dataset_id or args.phase
        args.phase = None

    # Select dataset: prefer explicit --dataset-id; otherwise default to prepared dataset for phase
    ds = args.dataset_id or (_select_prepared_for_phase(args.phase) if args.phase else None)
    
    # If dataset was provided but phase was missing, attempt to infer phase from dataset filename/id
    if not args.phase and ds:
        inferred = _infer_phase_from_dataset_id(ds)
        if inferred:
            args.phase = inferred
    
    # Validation: ensure we have a dataset
    if not ds:
        print("No dataset found for this phase; provide --dataset-id", file=sys.stderr)
        return 2
    
    try:
        # Load the full dataset into memory
        df = load_dataset(ds)
        
        # Resolve the target column and leakage exclusion list based on phase.
        # For prepared datasets that already have the target column computed (e.g., phase_b_prepared
        # which includes target_close_higher_prev), we use the pre-computed target directly rather
        # than trying to derive it from end_price (which may not exist in prepared datasets).
        phase_val = _normalize_phase(args.phase)
        
        # Check if this is a prepared Phase B dataset with target already computed
        if phase_val == "phaseB" and "target_close_higher_prev" in df.columns:
            # Use pre-computed target and complete leakage exclusion list for Phase B
            target_col = "target_close_higher_prev"
            exclude = [
                "trade_date", "target_close_higher_prev", "end_price", 
                "open_price", "close_price", "stock_return_daily",
                "pm_direction_up", "pm_direction_up_synth", "pm_direction_up_final",
                "up_price_final", "down_price_final"
            ]
        # Check if this is a prepared Phase A dataset with target already computed
        elif phase_val == "phaseA" and any(col in df.columns for col in ["pm_direction_up", "pm_direction_up_final", "pm_direction_up_synth"]):
            # Use pre-computed target and complete leakage exclusion list for Phase A
            target_col, exclude = _resolve_phase_target(df, args.phase, args.target_col)
        else:
            # For raw or non-prepared datasets, derive the target from raw data
            target_col, exclude = _resolve_phase_target(df, args.phase, args.target_col)
        
        # Prepare the feature matrix: clean data, handle missing values, select numeric features
        prep = prepare_feature_matrix(df=df, target_col=target_col, exclude_cols=exclude, min_numeric_ratio=args.min_numeric_ratio)
        
        # Format output with feature statistics
        out = {
            "dataset_id": ds,
            "phase": _normalize_phase(args.phase),
            "rows": prep["rows"],
            "feature_count": len(prep["selected_features"]),
            "selected_features": prep["selected_features"],
            "dropped_features": prep["dropped_features"],
            "class_distribution": prep["class_distribution"],
        }
        print(json.dumps(out, indent=2))
        return 0
    except Exception as exc:
        print(f"Error preparing features: {exc}", file=sys.stderr)
        return 3


def _build_hyperparams(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """Build hyperparameter dictionaries from individual CLI flags for all selected models.
    
    This function converts command-line arguments like --max-depth and --n-estimators
    into a nested dictionary structure that the training pipeline expects.
    
    Args:
        args: Parsed command-line arguments containing model names and hyperparameter flags
        
    Returns:
        Dictionary mapping model names to their hyperparameter dictionaries
        Example: {"Gradient Boosting": {"max_depth": 5, "n_estimators": 100}, ...}
        
    Model-specific hyperparameters:
        - Gradient Boosting: max_depth, n_estimators, learning_rate, random_state
        - Random Forest: max_depth, n_estimators, random_state
        - Logistic Regression: max_iter, random_state
    """
    hyperparams = {}
    
    # Iterate through each model the user selected for training
    for model_name in args.models:
        model_params = {}
        
        # Shared parameter: random_state ensures reproducibility (same split/seed every run)
        if hasattr(args, 'random_state') and args.random_state is not None:
            model_params['random_state'] = args.random_state
        
        # Gradient Boosting specific parameters
        # - max_depth: limits tree depth to prevent overfitting (shallow trees = less complexity)
        # - n_estimators: number of boosting stages (more iterations = better fitting but slower)
        # - learning_rate: shrinks contribution of each tree (lower = slower but more accurate)
        if model_name == "Gradient Boosting":
            if hasattr(args, 'max_depth') and args.max_depth is not None:
                model_params['max_depth'] = args.max_depth
            if hasattr(args, 'n_estimators') and args.n_estimators is not None:
                model_params['n_estimators'] = args.n_estimators
            if hasattr(args, 'learning_rate') and args.learning_rate is not None:
                model_params['learning_rate'] = args.learning_rate
        
        # Random Forest specific parameters
        # - max_depth: limits each tree depth (helps with overfitting)
        # - n_estimators: number of trees to grow (more trees = better but slower)
        elif model_name == "Random Forest":
            if hasattr(args, 'max_depth') and args.max_depth is not None:
                model_params['max_depth'] = args.max_depth
            if hasattr(args, 'n_estimators') and args.n_estimators is not None:
                model_params['n_estimators'] = args.n_estimators
        
        # Logistic Regression specific parameters
        # - max_iter: max iterations for convergence (if model doesn't converge, increase this)
        elif model_name == "Logistic Regression":
            if hasattr(args, 'max_iter') and args.max_iter is not None:
                model_params['max_iter'] = args.max_iter
        
        hyperparams[model_name] = model_params
    
    return hyperparams


def cmd_train(args: argparse.Namespace) -> int:
    """Train machine learning models for predicting market direction in a given phase.
    
    This command:
    1. Loads the dataset for the specified phase
    2. Resolves the target column and identifies leakage columns to exclude
    3. Prepares the feature matrix (cleans, selects numeric features)
    4. Performs cross-validated training on multiple models
    5. Saves the best model artifacts to disk
    6. Returns training metrics (accuracy, F1, confusion matrix, etc.)
    
    Users can customize:
    - Hyperparameters: --max-depth, --n-estimators, --learning-rate, --max-iter
    - Cross-validation: --n-splits, --n-repeats, --random-state
    - Feature selection: --min-numeric-ratio
    """
    # Register all processed datasets so they can be looked up by ID
    register_existing_processed_datasets()
    
    # User error handling: allow users to accidentally pass a dataset id into --phase
    # Detect by checking for dataset-id markers (::, .csv, /) and swap if found
    if args.phase and ("::" in args.phase or args.phase.endswith(".csv") or "/" in args.phase):
        args.dataset_id = args.dataset_id or args.phase
        args.phase = None

    # Select dataset: prefer explicit --dataset-id; otherwise default to prepared dataset for phase
    ds = args.dataset_id or (_select_prepared_for_phase(args.phase) if args.phase else None)
    
    # Validation: ensure we have a dataset
    if not ds:
        print("No dataset found for this phase; provide --dataset-id or --phase (phaseA/phaseB)", file=sys.stderr)
        return 2
    
    # If phase not provided but dataset id is, attempt to infer phase from dataset filename/id
    # This helps with phase-specific logic like target column resolution and leakage exclusion
    if not args.phase and ds:
        inferred = _infer_phase_from_dataset_id(ds)
        if inferred:
            args.phase = inferred
    
    try:
        # Load the full dataset into memory
        df = load_dataset(ds)
        
        # Resolve the target column and leakage exclusion list based on phase.
        # For prepared datasets that already have the target column computed (e.g., phase_b_prepared
        # which includes target_close_higher_prev), we use the pre-computed target directly rather
        # than trying to derive it from end_price (which may not exist in prepared datasets).
        phase_val = _normalize_phase(args.phase)
        
        # Check if this is a prepared Phase B dataset with target already computed
        if phase_val == "phaseB" and "target_close_higher_prev" in df.columns:
            # Use pre-computed target and complete leakage exclusion list for Phase B
            # IMPORTANT: stock_return_daily is excluded because it directly correlates with
            # whether price goes up/down, causing artificial accuracy inflation
            target_col = "target_close_higher_prev"
            exclude = [
                "trade_date", "target_close_higher_prev", "end_price", 
                "open_price", "close_price", "stock_return_daily",
                "pm_direction_up", "pm_direction_up_synth", "pm_direction_up_final",
                "up_price_final", "down_price_final"
            ]
        # Check if this is a prepared Phase A dataset with target already computed
        elif phase_val == "phaseA" and any(col in df.columns for col in ["pm_direction_up", "pm_direction_up_final", "pm_direction_up_synth"]):
            # Use pre-computed target and complete leakage exclusion list for Phase A
            target_col, exclude = _resolve_phase_target(df, args.phase, args.target_col)
        else:
            # For raw or non-prepared datasets, derive the target from raw data
            target_col, exclude = _resolve_phase_target(df, args.phase, args.target_col)
        
        # Build hyperparameter dictionary from CLI arguments
        # Priority: if --hyperparams JSON is provided, use it; otherwise build from individual flags
        # Individual flags like --max-depth, --n-estimators are model-agnostic
        if args.hyperparams:
            hyperparams = args.hyperparams
        else:
            hyperparams = _build_hyperparams(args)
        
        # Execute the training pipeline:
        # - Prepares features (cleans data, selects numeric columns, handles missing values)
        # - Performs k-fold cross-validation (trains and tests multiple times on different data slices)
        # - Trains final model on 100% of data (ready for production)
        # - Saves model artifact to disk and returns training metrics
        run = train_models(
            dataset_id=ds,
            target_col=target_col,
            selected_models=args.models,
            exclude_cols=exclude,
            min_numeric_ratio=args.min_numeric_ratio,
            hyperparams=hyperparams,
            cv_cfg={"type": "RepeatedStratifiedKFold", "n_splits": args.n_splits, "n_repeats": args.n_repeats, "random_state": args.random_state},
        )
        
        # Output training results as JSON
        print(json.dumps(run, indent=2, default=str))
        return 0
    except Exception as exc:
        print(f"Training error: {exc}", file=sys.stderr)
        return 4


def cmd_predict_single(args: argparse.Namespace) -> int:
    """Make a single prediction using a trained model with user-provided feature values.
    
    This command allows you to:
    1. Load a trained model from disk (identified by model_id)
    2. Provide feature values as a JSON object
    3. Get a prediction (0 or 1) with confidence scores
    
    Args (from CLI):
        model_id: The trained model to use (e.g., "mdl_abc12345")
        json: A JSON string containing feature names and their values
              Example: '{"volume": 1000000, "gap_pct": 2.5, "headline_count_x": 5}'
              
    Output: JSON object with prediction (0 or 1) and probabilities for each class
    
    Usage Example:
        .venv/bin/python scripts/quam_cli.py predict-single mdl_abc12345 '{"volume": 1000000}'
    """
    try:
        # Parse the JSON string containing feature values
        payload = json.loads(args.json)
    except Exception as exc:
        print(f"Invalid JSON: {exc}", file=sys.stderr)
        return 2
    
    try:
        # Run prediction: model will fill in missing features with NaN, then impute with median
        res = predict_single(args.model_id, payload)
        # Output result as JSON
        print(json.dumps(res, indent=2))
        return 0
    except Exception as exc:
        print(f"Prediction error: {exc}", file=sys.stderr)
        return 5


def cmd_predict_batch(args: argparse.Namespace) -> int:
    """Make predictions on an entire dataset at once using a trained model.
    
    This command allows you to:
    1. Load a trained model from disk (identified by model_id)
    2. Load a dataset (CSV or registered dataset)
    3. Generate predictions for every row in parallel
    4. Get predictions with confidence scores
    
    Args (from CLI):
        model_id: The trained model to use (e.g., "mdl_abc12345")
        dataset_id: The dataset to predict on (e.g., "processed::phase_b_prepared")
        
    Output: JSON object with array of predictions, one per row in the dataset.
            Each prediction includes row index, predicted class, and probabilities.
            
    Usage Example:
        .venv/bin/python scripts/quam_cli.py predict-batch mdl_abc12345 processed::phase_b_prepared
    """
    # Ensure all processed datasets are registered for lookup
    register_existing_processed_datasets()
    
    try:
        # Load the dataset to predict on
        df = load_dataset(args.dataset_id)
        
        # Run batch predictions: model applies to all rows simultaneously
        res = predict_batch(args.model_id, df)
        
        # Output results as JSON
        print(json.dumps(res, indent=2))
        return 0
    except Exception as exc:
        print(f"Batch prediction error: {exc}", file=sys.stderr)
        return 6


def build_parser() -> argparse.ArgumentParser:
    """Build and return the command-line argument parser for all CLI commands.
    
    This sets up the argument parsing structure for all subcommands:
    - list: Show all available datasets
    - preview: Display dataset details and samples
    - prepare: Analyze dataset and select features (no training)
    - train: Train models and save artifacts
    - predict-single: Make predictions on individual feature vectors
    - predict-batch: Make predictions on entire datasets
    
    Returns:
        ArgumentParser configured with all subcommands and their arguments
    """
    # Create the main parser
    p = argparse.ArgumentParser(prog="quam_cli")
    sub = p.add_subparsers(dest="cmd")

    # Subcommand: list - Show all registered datasets
    sub.add_parser("list", help="List registered datasets")

    # Subcommand: preview - Show dataset details (columns, samples, missing data)
    pv = sub.add_parser("preview", help="Preview dataset")
    pv.add_argument("dataset_id", help="Dataset ID to preview")
    pv.add_argument("--limit", type=int, default=10, help="Number of sample rows to display (default: 10)")

    # Subcommand: prepare - Analyze features without training
    prep = sub.add_parser("prepare", help="Prepare features for a phase")
    prep.add_argument("--phase", required=False, help="Phase: phaseA or phaseB (can be omitted if --dataset-id provided)")
    prep.add_argument("--dataset-id", default=None, help="Specific dataset ID to use (optional if --phase is provided)")
    prep.add_argument("--target-col", default=None, help="Override the target column name (usually auto-detected)")
    prep.add_argument("--min-numeric-ratio", type=float, default=0.8, help="Minimum ratio of numeric values required to keep a column (default: 0.8)")

    # Subcommand: train - Train models on a dataset
    tr = sub.add_parser("train", help="Train models for a phase")
    
    # Dataset and phase specification
    tr.add_argument("--phase", required=False, help="Phase: phaseA or phaseB (can be omitted if --dataset-id provided)")
    tr.add_argument("--dataset-id", default=None, help="Specific dataset ID to use (optional if --phase is provided)")
    tr.add_argument("--target-col", default=None, help="Override the target column name (usually auto-detected)")
    
    # Model selection and feature preprocessing
    tr.add_argument("--models", nargs="*", default=["Gradient Boosting"], help="Model types to train: 'Gradient Boosting', 'Random Forest', 'Logistic Regression' (default: Gradient Boosting)")
    tr.add_argument("--min-numeric-ratio", type=float, default=0.8, help="Minimum ratio of numeric values required to keep a column (default: 0.8)")

    # Cross-validation configuration
    tr.add_argument("--n-splits", type=int, default=5, help="Number of CV folds (default: 5)")
    tr.add_argument("--n-repeats", type=int, default=3, help="Number of CV repeats (default: 3)")
    tr.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility (default: 42)")

    # Individual hyperparameter flags (easier than raw JSON for most users)
    tr.add_argument("--max-depth", type=int, default=None, help="Max tree depth (for Random Forest and Gradient Boosting)")
    tr.add_argument("--n-estimators", type=int, default=None, help="Number of trees/estimators (for Random Forest and Gradient Boosting)")
    tr.add_argument("--learning-rate", type=float, default=None, help="Learning rate (for Gradient Boosting only)")
    tr.add_argument("--max-iter", type=int, default=None, help="Max iterations (for Logistic Regression only)")
    
    # Advanced: raw JSON hyperparameters (overrides individual flags if provided)
    tr.add_argument("--hyperparams", type=json.loads, default=None, help="Raw JSON hyperparams (overrides individual flags). Example: '{\"Gradient Boosting\": {\"max_depth\": 5}}'")

    # Subcommand: predict-single - Make a single prediction
    ps = sub.add_parser("predict-single", help="Run single prediction")
    ps.add_argument("model_id", help="Model ID (e.g., mdl_abc12345)")
    ps.add_argument("json", help="JSON string with feature values (e.g., '{\"volume\": 1000000}')")

    # Subcommand: predict-batch - Make predictions on entire dataset
    pb = sub.add_parser("predict-batch", help="Run batch prediction on a dataset")
    pb.add_argument("model_id", help="Model ID (e.g., mdl_abc12345)")
    pb.add_argument("dataset_id", help="Dataset ID to predict on (e.g., processed::phase_b_prepared)")

    return p


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI application.
    
    This function:
    1. Parses command-line arguments
    2. Routes to the appropriate subcommand handler
    3. Returns an exit code (0 = success, non-zero = error)
    
    Args:
        argv: List of command-line arguments (defaults to sys.argv[1:] if None).
              Useful for testing.
              
    Exit codes:
        0: Success
        1: No command provided (help shown)
        2: Invalid arguments / dataset not found
        3: Error during feature preparation
        4: Error during model training
        5: Error during single prediction
        6: Error during batch prediction
    """
    # Use provided argv or default to system arguments (skipping program name)
    argv = argv if argv is not None else sys.argv[1:]
    
    # Build the argument parser with all subcommands
    parser = build_parser()
    
    # Parse the command-line arguments
    args = parser.parse_args(argv)
    
    # If no command was provided, show help and exit
    if not args.cmd:
        parser.print_help()
        return 1

    # Route to the appropriate command handler based on args.cmd
    if args.cmd == "list":
        return cmd_list(args)
    if args.cmd == "preview":
        return cmd_preview(args)
    if args.cmd == "prepare":
        return cmd_prepare(args)
    if args.cmd == "train":
        return cmd_train(args)
    if args.cmd == "predict-single":
        return cmd_predict_single(args)
    if args.cmd == "predict-batch":
        return cmd_predict_batch(args)

    # Unknown command (shouldn't happen due to argparse, but safety check)
    print("Unknown command", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
