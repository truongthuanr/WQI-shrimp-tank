# Phase 3 Model Description

## Script
- `randomforest.py`

## Purpose
- Train and evaluate Random Forest alkalinity forecast models with a reproducible, time-aware pipeline.
- Run multiple forecast shifts in one command (`shift_day` loop).

## Shift Definition
- `shift_day = N` means target is alkalinity value shifted by `N` rows forward within each `unit_id` timeline.
- Default run: `1,2,3`.

## Output Structure
- `output/all_runs_summary.json`: summary for all shifts.
- `output/shift_1/run_summary.json`, `output/shift_2/run_summary.json`, ...
- `output/shift_1/test_predictions.csv`, `output/shift_2/test_predictions.csv`, ...

## Naming Guidance (to avoid confusion)
- Use `*_baseline.py` for reproducible benchmark scripts.
- Use `*_exp_<topic>.py` for experiments.
- Keep one script for one responsibility (train/eval vs apply/inference).
- Add a 3-5 line header docstring in every new file: purpose, input, output, split strategy.
