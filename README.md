# RandomForest v2 (Reproducible Pipeline)

## File
- `01_model_src/03_phase3/randomforest.py`

## What this version adds
- Time-aware split: train -> validation -> test (chronological)
- Leakage-safe preprocessing via `Pipeline` + `ColumnTransformer`
- Hyperparameter tuning with `RandomizedSearchCV` and `TimeSeriesSplit`
- Final holdout test metrics: RMSE, MAE, MAPE, R2, Bias
- Multi-horizon loop via `--shift-days` (default: `1,2,3`)
- Exported artifacts for traceability:
  - `output/all_runs_summary.json`
  - `output/shift_<N>/run_summary.json`
  - `output/shift_<N>/test_predictions.csv`

## Run
```bash
python randomforest.py --data-path ../../../dataset/data_4perday_cleaned.csv --output-dir ./output --shift-days 1,2,3
```

## Notes
- `shift-days` controls forecast horizon in row-steps.
- Example: `--shift-days 1,2,3` trains/evaluates three models in one run.
