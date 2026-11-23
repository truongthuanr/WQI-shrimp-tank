"""Evaluate the 10 specified predictors for alkalinity prediction.

Input: cleaned CSV used by the phase-3 pipeline.
Output: holdout metrics, row counts, and test predictions.
Split: chronological or seeded random train/validation/test (60/20/20).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from randomforest import (
    BEST_PARAMS,
    COL_ALKALINITY,
    COL_AREA,
    COL_CROP,
    COL_DATE,
    COL_FARM_TECH,
    COL_MODULE,
    COL_PH,
    COL_POND,
    COL_POND_TYPE,
    COL_SALINITY,
    COL_SEASON,
    COL_SHRIMP_AGE,
    COL_TEMP,
    COL_TRANSPARENCY,
    COL_WATER_LEVEL,
    Config,
    apply_zscore_filter,
    build_unit_id,
    make_pipeline,
    metrics_report,
    prepare_shift_target,
    split_time_aware,
)


CATEGORICAL_COLS = [COL_FARM_TECH, COL_POND_TYPE, COL_SEASON]
NUMERIC_COLS = [
    COL_SHRIMP_AGE,
    COL_TEMP,
    COL_PH,
    COL_SALINITY,
    COL_WATER_LEVEL,
    COL_AREA,
    COL_TRANSPARENCY,
]
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="../../../dataset/data_4perday_cleaned.csv")
    parser.add_argument("--output-dir", default="./output_exp_10_features")
    parser.add_argument("--shift-days", type=int, default=0)
    parser.add_argument("--zscore-limit", type=float, default=3.0)
    parser.add_argument("--split-method", choices=["time", "random"], default="time")
    args = parser.parse_args()
    if args.shift_days < 0:
        parser.error("--shift-days must be nonnegative")

    cfg = Config(data_path=args.data_path, output_dir=args.output_dir)
    df = pd.read_csv(args.data_path)
    needed = [COL_DATE, COL_CROP, COL_MODULE, COL_POND, COL_ALKALINITY] + FEATURE_COLS
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df[COL_DATE] = pd.to_datetime(df[COL_DATE], dayfirst=True, errors="coerce")
    for col in NUMERIC_COLS + [COL_ALKALINITY]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.loc[df[COL_WATER_LEVEL] == 0, COL_WATER_LEVEL] = np.nan
    df[COL_WATER_LEVEL] = df[COL_WATER_LEVEL].fillna(df[COL_WATER_LEVEL].median())
    df["unit_id"] = build_unit_id(df)
    n_raw = len(df)
    df = df[[COL_DATE, "unit_id", COL_ALKALINITY] + FEATURE_COLS]
    df = df.sort_values(["unit_id", COL_DATE]).dropna().reset_index(drop=True)
    n_after_clean = len(df)
    df, n_outliers = apply_zscore_filter(df, NUMERIC_COLS, args.zscore_limit)
    n_after_zscore = len(df)
    if args.shift_days:
        df, target_col, n_missing_target = prepare_shift_target(df, args.shift_days)
    else:
        target_col = COL_ALKALINITY
        n_missing_target = 0

    if args.split_method == "time":
        train, val, test = split_time_aware(df, cfg)
    else:
        trainval, test = train_test_split(
            df, test_size=cfg.test_ratio, random_state=cfg.random_state
        )
        train, val = train_test_split(
            trainval,
            test_size=cfg.val_ratio_within_trainval,
            random_state=cfg.random_state,
        )
    trainval = pd.concat([train, val], ignore_index=True)
    estimator = make_pipeline(CATEGORICAL_COLS, NUMERIC_COLS, cfg.random_state)
    estimator.set_params(**BEST_PARAMS)
    estimator.fit(trainval[FEATURE_COLS], trainval[target_col])
    predictions = estimator.predict(test[FEATURE_COLS])
    metrics = metrics_report(test[target_col].to_numpy(), predictions)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            COL_DATE: test[COL_DATE],
            "unit_id": test["unit_id"],
            "observed": test[target_col],
            "prediction": predictions,
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    summary = {
        "data_path": args.data_path,
        "target_column": target_col,
        "categorical_features": CATEGORICAL_COLS,
        "numeric_features": NUMERIC_COLS,
        "zscore_limit": args.zscore_limit,
        "split_method": args.split_method,
        "best_params": BEST_PARAMS,
        "rows": {
            "raw": n_raw,
            "after_clean": n_after_clean,
            "after_zscore": n_after_zscore,
            "excluded_outliers": n_outliers,
            "missing_shift_target": n_missing_target,
            "train": len(train),
            "validation": len(val),
            "test": len(test),
        },
        "test_metrics": metrics,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
