#!/usr/bin/env python3
"""
Prepare clean Phase A and Phase B training datasets with no leakage.
Creates CSV files and PDF documentation for model training.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Try to import reportlab for PDF generation, otherwise create simple text version
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    print("Warning: reportlab not available. Using CSV only.")


RANDOM_STATE = 42
MIN_NUMERIC_RATIO = 0.80

# Phase A configuration
PHASE_A_TARGET = "pm_direction_up"
PHASE_A_EXCLUDE = {"trade_date", "open_price", "close_price", "pm_return", "pm_direction_up"}

# Phase B configuration - WITH FULL LEAKAGE REMOVAL
PHASE_B_TARGET = "target_close_higher_prev"
PHASE_B_EXCLUDE = {
    "trade_date",
    "target_close_higher_prev",
    "end_price",
    "pm_direction_up",
    "pm_direction_up_synth",
    "up_price_final",
    "down_price_final",
    # Additional leakage removal (same-day target information)
    "pm_direction_up_final",
    "stock_return_daily",
    "close_price",
    "open_price",
    "pre_market_gap_pct",  # Daily market movement that correlates with daily close direction
}


def load_csv_from_processed(project_root: Path, csv_name: str) -> tuple:
    """Load CSV from processed data directory with normalization."""
    dataset_path = project_root / "data" / "processed" / csv_name
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset: {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    df.columns = [str(col).strip() for col in df.columns]
    return df, dataset_path


def build_feature_matrix(
    input_df: pd.DataFrame,
    target_col: str,
    exclude_cols: set,
    min_numeric_ratio: float = 0.80,
) -> tuple:
    """Build numeric feature matrix with target validation and leakage-safe exclusions."""
    working_df = input_df.copy()
    
    if target_col not in working_df.columns:
        raise ValueError(f"Target column not found: {target_col}")
    
    # Coerce target to numeric and keep only labeled rows
    working_df[target_col] = pd.to_numeric(working_df[target_col], errors="coerce")
    labeled_df = working_df[working_df[target_col].notna()].copy()
    
    if labeled_df.empty:
        raise ValueError("No labeled rows remain after target cleanup.")
    
    # Select numeric features
    selected_features = []
    drop_log_rows = []
    
    for column in labeled_df.columns:
        if column in exclude_cols:
            continue
        
        numeric_series = pd.to_numeric(labeled_df[column], errors="coerce")
        non_null_ratio = float(numeric_series.notna().mean())
        
        if non_null_ratio >= min_numeric_ratio:
            labeled_df[column] = numeric_series
            selected_features.append(column)
        else:
            drop_log_rows.append({
                "column": column,
                "reason": "low_numeric_ratio",
                "numeric_ratio": round(non_null_ratio, 4),
            })
    
    if not selected_features:
        raise ValueError("No usable numeric feature columns after filtering.")
    
    X = labeled_df[selected_features].copy()
    y = labeled_df[target_col].astype(int)
    drop_log = pd.DataFrame(drop_log_rows)
    
    return X, y, selected_features, labeled_df, drop_log


def prepare_phase_a(project_root: Path) -> dict:
    """Prepare Phase A training dataset (Polymarket direction prediction)."""
    print("\n" + "="*60)
    print("PHASE A: Polymarket Direction Prediction")
    print("="*60)
    
    # Load data
    df, dataset_path = load_csv_from_processed(project_root, "nvda_multimodal_final_model_ready.csv")
    print(f"✓ Loaded {len(df)} rows from {dataset_path.name}")
    
    # Build feature matrix
    X, y, feature_cols, labeled_df, drop_log = build_feature_matrix(
        input_df=df,
        target_col=PHASE_A_TARGET,
        exclude_cols=PHASE_A_EXCLUDE,
        min_numeric_ratio=MIN_NUMERIC_RATIO,
    )
    
    print(f"✓ Built feature matrix:")
    print(f"  - Labeled rows: {len(labeled_df)}")
    print(f"  - Features: {len(feature_cols)}")
    print(f"  - Excluded columns: {len(PHASE_A_EXCLUDE)}")
    
    # Create output with target column
    output_df = X.copy()
    output_df[PHASE_A_TARGET] = y
    
    # Save CSV
    output_path = project_root / "data" / "processed" / "nvda_phase_a_training_dataset.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"✓ Saved to {output_path}")
    
    # Compute statistics
    class_dist = y.value_counts().sort_index()
    print(f"✓ Target distribution:")
    print(f"  - Class 0 (DOWN): {class_dist.get(0, 0)} samples ({100*class_dist.get(0, 0)/len(y):.1f}%)")
    print(f"  - Class 1 (UP): {class_dist.get(1, 0)} samples ({100*class_dist.get(1, 0)/len(y):.1f}%)")
    
    # Missing value ratio
    missing_ratio = X.isnull().sum() / len(X)
    max_missing = missing_ratio.max()
    print(f"✓ Missing value coverage: max {max_missing*100:.1f}%")
    
    return {
        "phase": "A",
        "name": "Polymarket Direction Prediction",
        "target_col": PHASE_A_TARGET,
        "target_desc": "Binary target: 1 if Polymarket predicts UP, 0 if DOWN",
        "n_samples": len(y),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "class_0_count": int(class_dist.get(0, 0)),
        "class_1_count": int(class_dist.get(1, 0)),
        "excluded_cols": sorted(PHASE_A_EXCLUDE),
        "output_path": str(output_path),
        "output_df": output_df,
    }


def prepare_phase_b(project_root: Path) -> dict:
    """Prepare Phase B training dataset (Stock close direction prediction)."""
    print("\n" + "="*60)
    print("PHASE B: Stock Close Direction Prediction")
    print("="*60)
    
    # Load synthetic-extended data
    df, dataset_path = load_csv_from_processed(project_root, "nvda_multimodal_final_synthetic_pm.csv")
    print(f"✓ Loaded {len(df)} rows from {dataset_path.name}")
    
    # Prepare for target derivation
    stock_target_df = df.copy()
    stock_target_df["trade_date"] = pd.to_datetime(stock_target_df["trade_date"], errors="coerce")
    stock_target_df = stock_target_df.sort_values("trade_date").reset_index(drop=True)
    
    # Derive target: close higher than previous close
    end_price_series = pd.to_numeric(stock_target_df["end_price"], errors="coerce")
    prev_close = end_price_series.shift(1)
    stock_target_df[PHASE_B_TARGET] = (end_price_series > prev_close).astype(float)
    stock_target_df.loc[prev_close.isna(), PHASE_B_TARGET] = np.nan
    
    # Keep rows with valid target
    stock_target_df = stock_target_df[stock_target_df[PHASE_B_TARGET].notna()].copy()
    print(f"✓ Derived target: {len(stock_target_df)} rows with valid target")
    
    # Build feature matrix
    X, y, feature_cols, labeled_df, drop_log = build_feature_matrix(
        input_df=stock_target_df,
        target_col=PHASE_B_TARGET,
        exclude_cols=PHASE_B_EXCLUDE,
        min_numeric_ratio=MIN_NUMERIC_RATIO,
    )
    
    print(f"✓ Built feature matrix:")
    print(f"  - Labeled rows: {len(labeled_df)}")
    print(f"  - Features: {len(feature_cols)}")
    print(f"  - Excluded columns: {len(PHASE_B_EXCLUDE)}")
    
    # Create output with target column
    output_df = X.copy()
    output_df[PHASE_B_TARGET] = y
    
    # Save CSV
    output_path = project_root / "data" / "processed" / "nvda_phase_b_training_dataset.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"✓ Saved to {output_path}")
    
    # Compute statistics
    class_dist = y.value_counts().sort_index()
    print(f"✓ Target distribution:")
    print(f"  - Class 0 (LOWER): {class_dist.get(0, 0)} samples ({100*class_dist.get(0, 0)/len(y):.1f}%)")
    print(f"  - Class 1 (HIGHER): {class_dist.get(1, 0)} samples ({100*class_dist.get(1, 0)/len(y):.1f}%)")
    
    # Missing value ratio
    missing_ratio = X.isnull().sum() / len(X)
    max_missing = missing_ratio.max()
    print(f"✓ Missing value coverage: max {max_missing*100:.1f}%")
    
    return {
        "phase": "B",
        "name": "Stock Close Direction Prediction",
        "target_col": PHASE_B_TARGET,
        "target_desc": "Binary target: 1 if close > previous close, 0 otherwise",
        "n_samples": len(y),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "class_0_count": int(class_dist.get(0, 0)),
        "class_1_count": int(class_dist.get(1, 0)),
        "excluded_cols": sorted(PHASE_B_EXCLUDE),
        "output_path": str(output_path),
        "output_df": output_df,
    }


def validate_dataset(phase_info: dict) -> dict:
    """Validate dataset by training Gradient Boosting and computing metrics."""
    print(f"\n{'='*60}")
    print(f"VALIDATION: Training Gradient Boosting on Phase {phase_info['phase']}")
    print(f"{'='*60}")
    
    output_df = phase_info["output_df"]
    target_col = phase_info["target_col"]
    feature_cols = phase_info["feature_cols"]
    
    # Prepare X and y
    X = output_df[feature_cols].copy()
    y = output_df[target_col].astype(int)
    
    print(f"✓ Dataset: {len(y)} samples, {len(feature_cols)} features")
    print(f"✓ Class distribution: {dict(y.value_counts().sort_index())}")
    
    # Train with repeated stratified k-fold cross-validation
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ])
    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
    
    # Collect CV metrics
    accuracy_scores = []
    balanced_acc_scores = []
    f1_scores = []
    
    fold_num = 1
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        
        accuracy_scores.append(acc)
        balanced_acc_scores.append(bal_acc)
        f1_scores.append(f1)
        
        print(f"  Fold {fold_num}: accuracy={acc:.4f}, balanced_acc={bal_acc:.4f}, f1={f1:.4f}")
        fold_num += 1
    
    # Compute means and stds
    metrics = {
        "accuracy_mean": np.mean(accuracy_scores),
        "accuracy_std": np.std(accuracy_scores, ddof=1),
        "balanced_accuracy_mean": np.mean(balanced_acc_scores),
        "balanced_accuracy_std": np.std(balanced_acc_scores, ddof=1),
        "f1_weighted_mean": np.mean(f1_scores),
        "f1_weighted_std": np.std(f1_scores, ddof=1),
    }
    
    print(f"\n✓ Cross-validation results (15 folds):")
    print(f"  - Accuracy: {metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}")
    print(f"  - Balanced Accuracy: {metrics['balanced_accuracy_mean']:.4f} ± {metrics['balanced_accuracy_std']:.4f}")
    print(f"  - F1 Weighted: {metrics['f1_weighted_mean']:.4f} ± {metrics['f1_weighted_std']:.4f}")
    
    phase_info["metrics"] = metrics
    return phase_info


def create_text_report(phase_info: dict, output_path: Path):
    """Create a text-based report when PDF generation isn't available."""
    content = []
    content.append("=" * 80)
    content.append(f"PHASE {phase_info['phase']} TRAINING DATASET REPORT")
    content.append(f"{phase_info['name']}")
    content.append("=" * 80)
    content.append("")
    
    content.append("DATASET INFORMATION")
    content.append("-" * 80)
    content.append(f"Target Column: {phase_info['target_col']}")
    content.append(f"Description: {phase_info['target_desc']}")
    content.append(f"Total Samples: {phase_info['n_samples']}")
    content.append(f"Number of Features: {phase_info['n_features']}")
    content.append("")
    
    content.append("CLASS DISTRIBUTION")
    content.append("-" * 80)
    total = phase_info["class_0_count"] + phase_info["class_1_count"]
    content.append(f"Class 0: {phase_info['class_0_count']} samples ({100*phase_info['class_0_count']/total:.1f}%)")
    content.append(f"Class 1: {phase_info['class_1_count']} samples ({100*phase_info['class_1_count']/total:.1f}%)")
    content.append("")
    
    content.append("FEATURES")
    content.append("-" * 80)
    for i, feat in enumerate(phase_info["feature_cols"][:20], 1):
        content.append(f"{i}. {feat}")
    if len(phase_info["feature_cols"]) > 20:
        content.append(f"... and {len(phase_info['feature_cols']) - 20} more features")
    content.append("")
    
    content.append("EXCLUDED COLUMNS (Leakage Prevention)")
    content.append("-" * 80)
    for col in phase_info["excluded_cols"]:
        content.append(f"  - {col}")
    content.append("")
    
    content.append("VALIDATION METRICS (Gradient Boosting)")
    content.append("-" * 80)
    if "metrics" in phase_info:
        m = phase_info["metrics"]
        content.append(f"Accuracy: {m['accuracy_mean']:.4f} ± {m['accuracy_std']:.4f}")
        content.append(f"Balanced Accuracy: {m['balanced_accuracy_mean']:.4f} ± {m['balanced_accuracy_std']:.4f}")
        content.append(f"F1 Weighted: {m['f1_weighted_mean']:.4f} ± {m['f1_weighted_std']:.4f}")
    content.append("")
    
    content.append("FILE LOCATION")
    content.append("-" * 80)
    content.append(phase_info["output_path"])
    content.append("")
    
    content.append("=" * 80)
    
    with open(output_path, "w") as f:
        f.write("\n".join(content))
    
    print(f"\nReport saved to {output_path}")


def main():
    """Main entry point."""
    project_root = Path("/home/yousef/Projects/repos/PolySense")
    
    # Prepare Phase A
    phase_a = prepare_phase_a(project_root)
    phase_a = validate_dataset(phase_a)
    
    # Prepare Phase B
    phase_b = prepare_phase_b(project_root)
    phase_b = validate_dataset(phase_b)
    
    # Save text reports
    print("\n" + "="*60)
    print("GENERATING REPORTS")
    print("="*60)
    
    for phase_info in [phase_a, phase_b]:
        report_path = project_root / "data" / "processed" / f"nvda_phase_{phase_info['phase']}_report.txt"
        create_text_report(phase_info, report_path)
    
    # Save JSON metadata for reference
    metadata_path = project_root / "data" / "processed" / "training_datasets_metadata.json"
    metadata = {
        "phase_a": {k: v for k, v in phase_a.items() if k != "output_df"},
        "phase_b": {k: v for k, v in phase_b.items() if k != "output_df"},
        "created_at": pd.Timestamp.now().isoformat(),
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"✓ Metadata saved to {metadata_path}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Phase A: {phase_a['n_samples']} samples, {phase_a['n_features']} features")
    print(f"  - CSV: {phase_a['output_path']}")
    print(f"  - Report: {project_root / 'data' / 'processed' / 'nvda_phase_a_report.txt'}")
    print(f"  - Accuracy: {phase_a['metrics']['accuracy_mean']:.4f} ± {phase_a['metrics']['accuracy_std']:.4f}")
    print()
    print(f"✓ Phase B: {phase_b['n_samples']} samples, {phase_b['n_features']} features")
    print(f"  - CSV: {phase_b['output_path']}")
    print(f"  - Report: {project_root / 'data' / 'processed' / 'nvda_phase_b_report.txt'}")
    print(f"  - Accuracy: {phase_b['metrics']['accuracy_mean']:.4f} ± {phase_b['metrics']['accuracy_std']:.4f}")
    print("\n✓ All datasets prepared and validated successfully!")


if __name__ == "__main__":
    main()
