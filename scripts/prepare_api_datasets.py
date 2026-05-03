#!/usr/bin/env python3
"""
Create API-compatible versions of training datasets that include trade_date and end_price
(which are needed by the backend for Phase B target derivation) but not included in features.
"""
import pandas as pd
from pathlib import Path

project_root = Path("/home/yousef/Projects/repos/PolySense")

# Load the original raw data
print("Loading raw datasets...")
phase_a_raw = pd.read_csv(project_root / "data" / "processed" / "nvda_multimodal_final_model_ready.csv")
phase_b_raw = pd.read_csv(project_root / "data" / "processed" / "nvda_multimodal_final_synthetic_pm.csv")

# Load our prepared datasets to get the feature lists and cleaned data
phase_a_prepared = pd.read_csv(project_root / "data" / "processed" / "nvda_phase_a_training_dataset.csv")
phase_b_prepared = pd.read_csv(project_root / "data" / "processed" / "nvda_phase_b_training_dataset.csv")

# Phase A: Keep only rows that are in the prepared dataset, add back necessary columns
print("\nPreparing Phase A API dataset...")
phase_a_api = phase_a_prepared.copy()
# Add back trade_date from raw data (aligned by index)
if len(phase_a_api) == len(phase_a_raw):
    phase_a_api['trade_date'] = phase_a_raw['trade_date']
    phase_a_api_path = project_root / "data" / "processed" / "nvda_phase_a_training_dataset_api.csv"
    phase_a_api.to_csv(phase_a_api_path, index=False)
    print(f"✓ Phase A API dataset saved: {len(phase_a_api)} rows, {len(phase_a_api.columns)} cols")
    print(f"  Path: {phase_a_api_path}")

# Phase B: Reconstruct from raw data with proper target derivation
print("\nPreparing Phase B API dataset...")
phase_b_temp = phase_b_raw.copy()
phase_b_temp['trade_date'] = pd.to_datetime(phase_b_temp['trade_date'], errors='coerce')
phase_b_temp = phase_b_temp.sort_values('trade_date').reset_index(drop=True)

# Derive target
end_price_series = pd.to_numeric(phase_b_temp['end_price'], errors='coerce')
prev_close = end_price_series.shift(1)
phase_b_temp['target_close_higher_prev'] = (end_price_series > prev_close).astype(float)
phase_b_temp.loc[prev_close.isna(), 'target_close_higher_prev'] = float('nan')

# Keep only rows with valid targets
phase_b_temp = phase_b_temp[phase_b_temp['target_close_higher_prev'].notna()].copy()

# Now include only the feature columns from our prepared dataset plus trade_date and end_price
feature_cols = [col for col in phase_b_prepared.columns if col != 'target_close_higher_prev']
phase_b_api = phase_b_temp[feature_cols + ['trade_date', 'end_price']].copy()
phase_b_api['target_close_higher_prev'] = phase_b_temp['target_close_higher_prev'].values

phase_b_api_path = project_root / "data" / "processed" / "nvda_phase_b_training_dataset_api.csv"
phase_b_api.to_csv(phase_b_api_path, index=False)
print(f"✓ Phase B API dataset saved: {len(phase_b_api)} rows, {len(phase_b_api.columns)} cols")
print(f"  Path: {phase_b_api_path}")

print("\n✓ API-compatible datasets ready for testing!")
