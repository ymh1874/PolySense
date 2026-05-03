#!/usr/bin/env python3
"""
Test that the reorganized datasets work correctly.
Train on train set, predict on test set.
"""
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np

project_root = Path("/home/yousef/Projects/repos/PolySense")
data_dir = project_root / "data" / "processed"

print("🧪 TESTING REORGANIZED DATASETS")
print("=" * 70)

# Test Phase A
print("\n📊 PHASE A: Polymarket Direction")
print("─" * 70)

# Load Phase A data
phase_a_train = pd.read_csv(data_dir / "nvda_phase_a_train.csv")
phase_a_test = pd.read_csv(data_dir / "nvda_phase_a_test.csv")

print(f"✓ Training set: {len(phase_a_train)} rows, {len(phase_a_train.columns)} columns")
print(f"✓ Test set: {len(phase_a_test)} rows, {len(phase_a_test.columns)} columns")

# Verify test set has no target
assert 'pm_direction_up' not in phase_a_test.columns, "❌ Test set contains target column!"
print(f"✓ Test set has no target column")

# Check for missing values
missing_train = phase_a_train.isnull().sum().sum()
missing_test = phase_a_test.isnull().sum().sum()
print(f"✓ Missing values in train: {missing_train}")
print(f"✓ Missing values in test: {missing_test}")

# Prepare data
X_train = phase_a_train.drop(columns=['pm_direction_up'])
y_train = phase_a_train['pm_direction_up'].astype(int)
X_test = phase_a_test

print(f"✓ Features: {len(X_train.columns)}")
print(f"✓ Target distribution: {dict(y_train.value_counts().sort_index())}")

# Train model
print(f"\n→ Training Gradient Boosting model...")
model_a = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('gb', GradientBoostingClassifier(random_state=42, n_estimators=100))
])
model_a.fit(X_train, y_train)
print(f"✓ Model trained successfully")

# Make predictions
y_pred_a = model_a.predict(X_test)
print(f"✓ Predictions made: {len(y_pred_a)} samples")
print(f"  Prediction distribution: {np.bincount(y_pred_a.astype(int))}")

# Get training accuracy (sanity check)
y_train_pred = model_a.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)
print(f"✓ Training accuracy: {train_acc:.4f}")

print(f"\n✅ PHASE A TEST PASSED")

# Test Phase B
print("\n" + "=" * 70)
print("📊 PHASE B: Stock Close Direction")
print("─" * 70)

# Load Phase B data
phase_b_train = pd.read_csv(data_dir / "nvda_phase_b_train.csv")
phase_b_test = pd.read_csv(data_dir / "nvda_phase_b_test.csv")

print(f"✓ Training set: {len(phase_b_train)} rows, {len(phase_b_train.columns)} columns")
print(f"✓ Test set: {len(phase_b_test)} rows, {len(phase_b_test.columns)} columns")

# Verify test set has no target
assert 'target_close_higher_prev' not in phase_b_test.columns, "❌ Test set contains target column!"
print(f"✓ Test set has no target column")

# Check for missing values
missing_train = phase_b_train.isnull().sum().sum()
missing_test = phase_b_test.isnull().sum().sum()
print(f"✓ Missing values in train: {missing_train}")
print(f"✓ Missing values in test: {missing_test}")

# Prepare data
X_train_b = phase_b_train.drop(columns=['target_close_higher_prev'])
y_train_b = phase_b_train['target_close_higher_prev'].astype(int)
X_test_b = phase_b_test

print(f"✓ Features: {len(X_train_b.columns)}")
print(f"✓ Target distribution: {dict(y_train_b.value_counts().sort_index())}")

# Train model
print(f"\n→ Training Gradient Boosting model...")
model_b = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('gb', GradientBoostingClassifier(random_state=42, n_estimators=100))
])
model_b.fit(X_train_b, y_train_b)
print(f"✓ Model trained successfully")

# Make predictions
y_pred_b = model_b.predict(X_test_b)
print(f"✓ Predictions made: {len(y_pred_b)} samples")
print(f"  Prediction distribution: {np.bincount(y_pred_b.astype(int))}")

# Get training accuracy (sanity check)
y_train_pred_b = model_b.predict(X_train_b)
train_acc_b = accuracy_score(y_train_b, y_train_pred_b)
print(f"✓ Training accuracy: {train_acc_b:.4f}")

print(f"\n✅ PHASE B TEST PASSED")

# Final summary
print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED - DATASETS ARE WORKING CORRECTLY")
print("=" * 70)
print("\nDataset Structure:")
print(f"  Phase A Train: {len(phase_a_train):3d} rows × {len(phase_a_train.columns):2d} columns (includes pm_direction_up)")
print(f"  Phase A Test:  {len(phase_a_test):3d} rows × {len(phase_a_test.columns):2d} columns (no target)")
print(f"  Phase B Train: {len(phase_b_train):3d} rows × {len(phase_b_train.columns):2d} columns (includes target_close_higher_prev)")
print(f"  Phase B Test:  {len(phase_b_test):3d} rows × {len(phase_b_test.columns):2d} columns (no target)")
print("\nPredictions:")
print(f"  Phase A: {len(y_pred_a)} predictions generated ✓")
print(f"  Phase B: {len(y_pred_b)} predictions generated ✓")
print("\n✓ Ready to use!")
