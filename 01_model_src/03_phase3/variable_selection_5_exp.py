"""Evaluate four models using the user's five specified predictors.

Input: cleaned phase-3 CSV; same-time alkalinity is the target (shift 0).
Output: per-model metrics and test predictions for review2.md.
Split: shared seeded random train/validation/test (60/20/20).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from randomforest import (
    COL_ALKALINITY,
    COL_CROP,
    COL_DATE,
    COL_FARM_TECH,
    COL_HARDNESS,
    COL_MODULE,
    COL_PH,
    COL_POND,
    COL_POND_TYPE,
    COL_SALINITY,
    COL_SHRIMP_AGE,
    COL_TDS,
    COL_TEMP,
    Config,
    apply_zscore_filter,
    build_unit_id,
    metrics_report,
)
from variable_selection_base_exp import fit_ann, fit_sklearn


FEATURES = [
    COL_HARDNESS,
    COL_POND_TYPE,
    COL_PH,
    COL_TDS,
    COL_SHRIMP_AGE,
]
NUMERIC = [COL_HARDNESS, COL_PH, COL_TDS, COL_SHRIMP_AGE]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="../../../dataset/data_4perday_cleaned.csv")
    parser.add_argument("--output-dir", default="./output_variable_selection_5")
    args = parser.parse_args()

    cfg = Config(data_path=args.data_path)
    df = pd.read_csv(args.data_path)
    needed = [COL_DATE, COL_CROP, COL_MODULE, COL_POND, COL_ALKALINITY] + FEATURES
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], dayfirst=True, errors="coerce")
    for col in NUMERIC + [COL_ALKALINITY]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["unit_id"] = build_unit_id(df)
    n_raw = len(df)
    df = df[[COL_DATE, "unit_id", COL_ALKALINITY] + FEATURES]
    df = df.sort_values(["unit_id", COL_DATE]).dropna().reset_index(drop=True)
    n_clean = len(df)
    df, n_outliers = apply_zscore_filter(df, NUMERIC, 3.0)

    trainval, test = train_test_split(
        df, test_size=cfg.test_ratio, random_state=cfg.random_state
    )
    train, val = train_test_split(
        trainval,
        test_size=cfg.val_ratio_within_trainval,
        random_state=cfg.random_state,
    )
    counts = {
        "raw": n_raw,
        "after_dropna": n_clean,
        "after_zscore": len(df),
        "excluded_outliers": n_outliers,
        "train": len(train),
        "validation": len(val),
        "trainval": len(trainval),
        "test": len(test),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for model_name in ("RF", "GBT", "ANN", "SVR"):
        if model_name == "ANN":
            predictions, fit_info = fit_ann(
                FEATURES, train, val, test, cfg.random_state
            )
        else:
            predictions, fit_info = fit_sklearn(
                model_name, FEATURES, train, val, trainval, test, cfg.random_state
            )
        metrics = metrics_report(test[COL_ALKALINITY].to_numpy(), predictions)
        model_dir = output_dir / model_name.lower()
        model_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            COL_DATE: test[COL_DATE],
            "unit_id": test["unit_id"],
            "observed": test[COL_ALKALINITY],
            "prediction": predictions,
        }).to_csv(model_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
        result = {
            "model": model_name,
            "features": FEATURES,
            "target": COL_ALKALINITY,
            "shift": 0,
            "split": "random 60/20/20",
            "seed": cfg.random_state,
            "zscore_limit": 3.0,
            "rows": counts,
            "fit": fit_info,
            "test_metrics": metrics,
        }
        (model_dir / "run_summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        results.append(result)
        print(
            f"{model_name}: n={len(test)} MAE={metrics['mae']:.4f} "
            f"RMSE={metrics['rmse']:.4f} R2={metrics['r2']:.4f}",
            flush=True,
        )

    (output_dir / "all_runs_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
