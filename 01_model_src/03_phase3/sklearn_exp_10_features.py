"""Evaluate GBT and SVR with the same 10 predictors and test rows as RF.

Input: cleaned CSV and saved RF test predictions.
Output: one summary and test-prediction CSV per model.
Split: seeded random train/validation/test (60/20/20), target at shift 0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from randomforest import (
    COL_ALKALINITY,
    COL_CROP,
    COL_DATE,
    COL_MODULE,
    COL_POND,
    COL_WATER_LEVEL,
    Config,
    apply_zscore_filter,
    build_unit_id,
    metrics_report,
)
from randomforest_exp_10_features import CATEGORICAL_COLS, FEATURE_COLS, NUMERIC_COLS


def load_and_split(data_path: str, cfg: Config):
    df = pd.read_csv(data_path)
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
    df = df[[COL_DATE, "unit_id", COL_ALKALINITY] + FEATURE_COLS]
    df = df.sort_values(["unit_id", COL_DATE]).dropna().reset_index(drop=True)
    df, _ = apply_zscore_filter(df, NUMERIC_COLS, 3.0)

    trainval, test = train_test_split(
        df, test_size=cfg.test_ratio, random_state=cfg.random_state
    )
    train, val = train_test_split(
        trainval,
        test_size=cfg.val_ratio_within_trainval,
        random_state=cfg.random_state,
    )
    return df, train, val, trainval, test


def assert_same_test(test: pd.DataFrame, rf_predictions_path: str) -> None:
    rf = pd.read_csv(rf_predictions_path, parse_dates=[COL_DATE])
    if len(rf) != len(test):
        raise ValueError("Test set size differs from the RF run")
    if not (
        rf[COL_DATE].reset_index(drop=True).equals(test[COL_DATE].reset_index(drop=True))
        and rf["unit_id"].reset_index(drop=True).equals(test["unit_id"].reset_index(drop=True))
        and np.allclose(rf["observed"], test[COL_ALKALINITY])
    ):
        raise ValueError("Test rows differ from the RF run")


def make_pipeline(model_name: str, seed: int) -> Pipeline:
    if model_name == "GBT":
        model = GradientBoostingRegressor(random_state=seed)
        numeric_transformer = "passthrough"
    else:
        model = SVR(kernel="rbf")
        numeric_transformer = StandardScaler()
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
            ("num", numeric_transformer, NUMERIC_COLS),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def parameter_grid(model_name: str):
    if model_name == "GBT":
        return ParameterGrid(
            {
                "model__n_estimators": [200, 400],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            }
        )
    return ParameterGrid(
        {
            "model__C": [10, 100, 300],
            "model__epsilon": [1, 5, 10],
            "model__gamma": ["scale"],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="../../../dataset/data_4perday_cleaned.csv")
    parser.add_argument("--output-dir", default="./output_sklearn_exp_10_features_random")
    parser.add_argument("--rf-predictions", default="./output_exp_10_features_random/test_predictions.csv")
    args = parser.parse_args()

    cfg = Config(data_path=args.data_path)
    df, train, val, trainval, test = load_and_split(args.data_path, cfg)
    assert_same_test(test, args.rf_predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in ("GBT", "SVR"):
        base = make_pipeline(model_name, cfg.random_state)
        best_params = None
        best_val_mae = float("inf")
        for params in parameter_grid(model_name):
            candidate = clone(base).set_params(**params)
            candidate.fit(train[FEATURE_COLS], train[COL_ALKALINITY])
            pred_val = candidate.predict(val[FEATURE_COLS])
            val_mae = float(np.mean(np.abs(val[COL_ALKALINITY].to_numpy() - pred_val)))
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_params = params

        final_model = clone(base).set_params(**best_params)
        final_model.fit(trainval[FEATURE_COLS], trainval[COL_ALKALINITY])
        predictions = final_model.predict(test[FEATURE_COLS])
        metrics = metrics_report(test[COL_ALKALINITY].to_numpy(), predictions)
        model_dir = output_dir / model_name.lower()
        model_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                COL_DATE: test[COL_DATE],
                "unit_id": test["unit_id"],
                "observed": test[COL_ALKALINITY],
                "prediction": predictions,
            }
        ).to_csv(model_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
        summary = {
            "model": model_name,
            "data_path": args.data_path,
            "target_column": COL_ALKALINITY,
            "shift_days": 0,
            "features": FEATURE_COLS,
            "split_method": "random",
            "random_state": cfg.random_state,
            "zscore_limit": 3.0,
            "rows": {
                "after_zscore": len(df),
                "train": len(train),
                "validation": len(val),
                "trainval": len(trainval),
                "test": len(test),
            },
            "test_rows_match_rf": True,
            "parameter_selection": "lowest validation MAE",
            "n_parameter_candidates": len(parameter_grid(model_name)),
            "best_params": best_params,
            "best_validation_mae": best_val_mae,
            "test_metrics": metrics,
        }
        (model_dir / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
