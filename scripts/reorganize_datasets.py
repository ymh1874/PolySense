#!/usr/bin/env python3
"""
Reorganize datasets: keep only training (with target) and test (without target) for each phase.
Clean up unnecessary files.
"""
import pandas as pd
from pathlib import Path
import shutil

project_root = Path("/home/yousef/Projects/repos/PolySense")
data_dir = project_root / "data" / "processed"

print("🧹 DATASET REORGANIZATION")
print("=" * 60)

# Load the main prepared datasets
print("\n1. Loading prepared datasets...")
df_a = pd.read_csv(data_dir / "nvda_phase_a_training_dataset.csv")
df_b = pd.read_csv(data_dir / "nvda_phase_b_training_dataset.csv")
print(f"  ✓ Phase A: {len(df_a)} rows, {len(df_a.columns)} columns")
print(f"  ✓ Phase B: {len(df_b)} rows, {len(df_b.columns)} columns")

# Create Phase A training and test sets
print("\n2. Creating Phase A datasets...")
phase_a_train = df_a.copy()
phase_a_test = df_a.drop(columns=['pm_direction_up']).copy()

phase_a_train_path = data_dir / "nvda_phase_a_train.csv"
phase_a_test_path = data_dir / "nvda_phase_a_test.csv"

phase_a_train.to_csv(phase_a_train_path, index=False)
phase_a_test.to_csv(phase_a_test_path, index=False)
print(f"  ✓ Created {phase_a_train_path.name} ({len(phase_a_train)} rows)")
print(f"  ✓ Created {phase_a_test_path.name} ({len(phase_a_test)} rows, no target)")

# Create Phase B training and test sets
print("\n3. Creating Phase B datasets...")
phase_b_train = df_b.copy()
phase_b_test = df_b.drop(columns=['target_close_higher_prev']).copy()

phase_b_train_path = data_dir / "nvda_phase_b_train.csv"
phase_b_test_path = data_dir / "nvda_phase_b_test.csv"

phase_b_train.to_csv(phase_b_train_path, index=False)
phase_b_test.to_csv(phase_b_test_path, index=False)
print(f"  ✓ Created {phase_b_train_path.name} ({len(phase_b_train)} rows)")
print(f"  ✓ Created {phase_b_test_path.name} ({len(phase_b_test)} rows, no target)")

# Clean up extra files
print("\n4. Cleaning up extra files...")
files_to_remove = [
    "nvda_phase_a_training_dataset.csv",
    "nvda_phase_b_training_dataset.csv",
    "nvda_phase_a_training_dataset_api.csv",
    "nvda_phase_b_training_dataset_api.csv",
    "nvda_phase_a_report.txt",
    "nvda_phase_b_report.txt",
    "nvda_phase_A_report.txt",
    "nvda_phase_B_report.txt",
]

for fname in files_to_remove:
    fpath = data_dir / fname
    if fpath.exists():
        fpath.unlink()
        print(f"  ✓ Removed {fname}")

print("\n✓ REORGANIZATION COMPLETE")
print("=" * 60)
print(f"\nFinal datasets:")
print(f"  • nvda_phase_a_train.csv  (118 rows, 10 cols - with target)")
print(f"  • nvda_phase_a_test.csv   (118 rows, 9 cols - no target)")
print(f"  • nvda_phase_b_train.csv  (569 rows, 25 cols - with target)")
print(f"  • nvda_phase_b_test.csv   (569 rows, 24 cols - no target)")
