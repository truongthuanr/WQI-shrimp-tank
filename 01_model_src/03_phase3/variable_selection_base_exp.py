"""Evaluate the 8- and 9-variable models listed in variable-selection.md.

Input: cleaned phase-3 CSV; same-time alkalinity is the target (shift 0).
Output: per-model metrics, test predictions, and a combined summary.
Split: shared seeded random train/validation/test (60/20/20).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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
    BEST_PARAMS,
    COL_ALKALINITY,
    COL_AREA,
    COL_COLOR,
    COL_CROP,
    COL_DATE,
    COL_FARM_TECH,
    COL_HARDNESS,
    COL_MODULE,
    COL_PH,
    COL_POND,
    COL_POND_TYPE,
    COL_SALINITY,
    COL_SEED,
    COL_SHRIMP_AGE,
    COL_TDS,
    COL_TEMP,
    COL_TRANSPARENCY,
    Config,
    apply_zscore_filter,
    build_unit_id,
    metrics_report,
)


FEATURE_SETS = {
    ("ANN", 8): [COL_SEED, COL_AREA, COL_TDS, COL_HARDNESS, COL_PH,
                 COL_POND_TYPE, COL_FARM_TECH, COL_SHRIMP_AGE],
    ("ANN", 9): [COL_TEMP, COL_SEED, COL_AREA, COL_TDS, COL_HARDNESS, COL_PH,
                 COL_POND_TYPE, COL_FARM_TECH, COL_SHRIMP_AGE],
    ("RF", 8): [COL_COLOR, COL_AREA, COL_SALINITY, COL_POND_TYPE,
                COL_HARDNESS, COL_TDS, COL_PH, COL_SHRIMP_AGE],
    ("RF", 9): [COL_TEMP, COL_COLOR, COL_AREA, COL_SALINITY, COL_POND_TYPE,
                COL_HARDNESS, COL_TDS, COL_PH, COL_SHRIMP_AGE],
    ("GBT", 8): [COL_HARDNESS, COL_SALINITY, COL_TEMP, COL_TDS,
                 COL_POND_TYPE, COL_FARM_TECH, COL_PH, COL_SHRIMP_AGE],
    ("GBT", 9): [COL_TRANSPARENCY, COL_HARDNESS, COL_SALINITY, COL_TEMP,
                 COL_TDS, COL_POND_TYPE, COL_FARM_TECH, COL_PH, COL_SHRIMP_AGE],
}
# variable-selection.md has no SVR sets; the user chose matching RF inputs.
FEATURE_SETS[("SVR", 8)] = FEATURE_SETS[("RF", 8)]
FEATURE_SETS[("SVR", 9)] = FEATURE_SETS[("RF", 9)]
CATEGORICAL = {COL_SEED, COL_POND_TYPE, COL_FARM_TECH}
ALL_FEATURES = list(dict.fromkeys(col for cols in FEATURE_SETS.values() for col in cols))
ALL_NUMERIC = [col for col in ALL_FEATURES if col not in CATEGORICAL]


def prepare_data(data_path: str, cfg: Config):
    df = pd.read_csv(data_path)
    needed = [COL_DATE, COL_CROP, COL_MODULE, COL_POND, COL_ALKALINITY] + ALL_FEATURES
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], dayfirst=True, errors="coerce")
    for col in ALL_NUMERIC + [COL_ALKALINITY]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["unit_id"] = build_unit_id(df)
    n_raw = len(df)
    df = df[[COL_DATE, "unit_id", COL_ALKALINITY] + ALL_FEATURES]
    df = df.sort_values(["unit_id", COL_DATE]).dropna().reset_index(drop=True)
    n_clean = len(df)
    df, n_outliers = apply_zscore_filter(df, ALL_NUMERIC, 3.0)
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
    return train, val, trainval, test, counts


def preprocessor_for(model_name: str, features: list[str]) -> ColumnTransformer:
    categorical = [col for col in features if col in CATEGORICAL]
    numeric = [col for col in features if col not in CATEGORICAL]
    numeric_transform = StandardScaler() if model_name in {"ANN", "SVR"} else "passthrough"
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("num", numeric_transform, numeric),
        ]
    )


def grid_for(model_name: str):
    if model_name == "GBT":
        return ParameterGrid({
            "model__n_estimators": [200, 400],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        })
    return ParameterGrid({
        "model__C": [10, 100, 300],
        "model__epsilon": [1, 5, 10],
        "model__gamma": ["scale"],
    })


def fit_sklearn(model_name: str, features: list[str], train, val, trainval, test, seed: int):
    preprocess = preprocessor_for(model_name, features)
    if model_name == "RF":
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(random_state=seed)
        pipeline = Pipeline([("preprocess", preprocess), ("model", model)])
        pipeline.set_params(**BEST_PARAMS)
        pipeline.fit(trainval[features], trainval[COL_ALKALINITY])
        return pipeline.predict(test[features]), {"params": BEST_PARAMS}

    model = GradientBoostingRegressor(random_state=seed) if model_name == "GBT" else SVR(kernel="rbf")
    base = Pipeline([("preprocess", preprocess), ("model", model)])
    best_params, best_mae = None, float("inf")
    for params in grid_for(model_name):
        candidate = clone(base).set_params(**params)
        candidate.fit(train[features], train[COL_ALKALINITY])
        prediction = candidate.predict(val[features])
        mae = float(np.mean(np.abs(val[COL_ALKALINITY].to_numpy() - prediction)))
        if mae < best_mae:
            best_params, best_mae = params, mae
    final = clone(base).set_params(**best_params)
    final.fit(trainval[features], trainval[COL_ALKALINITY])
    return final.predict(test[features]), {
        "params": best_params,
        "selection": "lowest validation MAE",
        "validation_mae": best_mae,
    }


def fit_ann(features: list[str], train, val, test, seed: int):
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    preprocess = preprocessor_for("ANN", features)
    x_train = preprocess.fit_transform(train[features]).astype(np.float32)
    x_val = preprocess.transform(val[features]).astype(np.float32)
    x_test = preprocess.transform(test[features]).astype(np.float32)
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(train[[COL_ALKALINITY]]).astype(np.float32)
    y_val = y_scaler.transform(val[[COL_ALKALINITY]]).astype(np.float32)
    layers = [tf.keras.layers.Input(shape=(x_train.shape[1],))]
    for units in (72, 60, 32, 8):
        layers.append(tf.keras.layers.Dense(units, kernel_initializer="he_uniform", activation="relu"))
        layers.append(tf.keras.layers.Dropout(0.1))
    layers.append(tf.keras.layers.Dense(1))
    model = tf.keras.Sequential(layers)
    model.compile(loss="mae", optimizer="nadam")
    history = model.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=200, batch_size=32, verbose=0,
    )
    prediction = y_scaler.inverse_transform(model.predict(x_test, verbose=0)).ravel()
    return prediction, {
        "architecture": [72, 60, 32, 8],
        "dropout": 0.1,
        "optimizer": "nadam",
        "loss": "mae",
        "epochs": 200,
        "batch_size": 32,
        "last_validation_loss_scaled": float(history.history["val_loss"][-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="../../../dataset/data_4perday_cleaned.csv")
    parser.add_argument("--output-dir", default="./output_variable_selection_base")
    args = parser.parse_args()
    cfg = Config(data_path=args.data_path)
    train, val, trainval, test, counts = prepare_data(args.data_path, cfg)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for model_name, size in FEATURE_SETS:
        features = FEATURE_SETS[(model_name, size)]
        if model_name == "ANN":
            prediction, fit_info = fit_ann(features, train, val, test, cfg.random_state)
        else:
            prediction, fit_info = fit_sklearn(
                model_name, features, train, val, trainval, test, cfg.random_state
            )
        metrics = metrics_report(test[COL_ALKALINITY].to_numpy(), prediction)
        case_name = f"{model_name}-{size}"
        case_dir = out_dir / case_name.lower()
        case_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            COL_DATE: test[COL_DATE],
            "unit_id": test["unit_id"],
            "observed": test[COL_ALKALINITY],
            "prediction": prediction,
        }).to_csv(case_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
        result = {
            "case": case_name,
            "features": features,
            "target": COL_ALKALINITY,
            "shift": 0,
            "split": "random 60/20/20",
            "seed": cfg.random_state,
            "rows": counts,
            "fit": fit_info,
            "test_metrics": metrics,
        }
        (case_dir / "run_summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        results.append(result)
        print(f"{case_name}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}", flush=True)

    (out_dir / "all_runs_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
