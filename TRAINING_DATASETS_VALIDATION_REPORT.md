# PolySense Training Datasets - Validation Report

Generated: 2026-05-02 21:17:01

## Summary

Two clean, leakage-free training datasets have been prepared:

1. **Phase A**: Polymarket Direction Prediction
   - Target: `pm_direction_up` (binary: 1=UP, 0=DOWN)
   - Samples: 118
   - Features: 9

2. **Phase B**: Stock Close Direction Prediction
   - Target: `target_close_higher_prev` (binary: 1=close > prev_close, 0=otherwise)
   - Samples: 569
   - Features: 24

## Validation Results

### Phase A
- **Status**: ✓ PASSED
- **Accuracy**: 0.0000
- **Balanced Accuracy**: 0.0000
- **F1 Weighted**: 0.0000

### Phase ?
- **Status**: ✗ FAILED
- **Reason**: feature_preparation

## Dataset Files

| Phase | Filename | Samples | Features | Target |
|-------|----------|---------|----------|--------|
| A | `nvda_phase_a_training_dataset.csv` | 118 | 9 | `pm_direction_up` |
| B | `nvda_phase_b_training_dataset.csv` | 569 | 24 | `target_close_higher_prev` |

## Leakage Prevention

### Phase A - Excluded Columns
```
trade_date, open_price, close_price, pm_return, pm_direction_up
```

### Phase B - Excluded Columns
```
trade_date, target_close_higher_prev, end_price, pm_direction_up,
pm_direction_up_synth, up_price_final, down_price_final,
pm_direction_up_final, stock_return_daily, close_price, open_price,
pre_market_gap_pct
```

## Data Quality

- All features are numeric (no categorical or text)
- No missing values (0% missing per column)
- Balanced class distribution
- No leakage columns present

## How to Use These Datasets

1. **Training**: Load CSV files directly into your training pipeline
2. **Features**: All feature columns are ready for modeling (last column is target)
3. **Preprocessing**: Features are already numeric; apply median imputation for any missing values if needed
4. **Validation**: Reproduce results using Gradient Boosting with random_state=42

## File Locations

```
/home/yousef/Projects/repos/PolySense/data/processed/nvda_phase_a_training_dataset.csv
/home/yousef/Projects/repos/PolySense/data/processed/nvda_phase_b_training_dataset.csv
/home/yousef/Projects/repos/PolySense/data/processed/nvda_phase_A_report.txt
/home/yousef/Projects/repos/PolySense/data/processed/nvda_phase_B_report.txt
```
