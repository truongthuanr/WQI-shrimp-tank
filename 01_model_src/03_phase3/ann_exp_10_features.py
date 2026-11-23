"""Evaluate ANN on the same 10-feature, same-time target as the RF experiment.

Input: cleaned CSV used by the RF 10-feature experiment.
Output: holdout metrics, training history, and test predictions.
Split: seeded random train/validation/test, matching RF's 60/20/20 split.
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
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="../../../dataset/data_4perday_cleaned.csv")
    parser.add_argument("--output-dir", default="./output_ann_exp_10_features_random")
    parser.add_argument("--rf-predictions", default="./output_exp_10_features_random/test_predictions.csv")
    args = parser.parse_args()

    cfg = Config(data_path=args.data_path)
    tf.keras.utils.set_random_seed(cfg.random_state)
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

    rf_predictions = pd.read_csv(args.rf_predictions, parse_dates=[COL_DATE])
    rf_keys = rf_predictions[[COL_DATE, "unit_id", "observed"]].reset_index(drop=True)
    ann_keys = test[[COL_DATE, "unit_id", COL_ALKALINITY]].reset_index(drop=True)
    if not (
        rf_keys[COL_DATE].equals(ann_keys[COL_DATE])
        and rf_keys["unit_id"].equals(ann_keys["unit_id"])
        and np.allclose(rf_keys["observed"], ann_keys[COL_ALKALINITY])
    ):
        raise ValueError("ANN test rows do not match the saved RF test rows")

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
            ("num", StandardScaler(), NUMERIC_COLS),
        ]
    )
    x_train = preprocessor.fit_transform(train[FEATURE_COLS]).astype(np.float32)
    x_val = preprocessor.transform(val[FEATURE_COLS]).astype(np.float32)
    x_test = preprocessor.transform(test[FEATURE_COLS]).astype(np.float32)
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(train[[COL_ALKALINITY]]).astype(np.float32)
    y_val = y_scaler.transform(val[[COL_ALKALINITY]]).astype(np.float32)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(x_train.shape[1],)),
            tf.keras.layers.Dense(72, kernel_initializer="he_uniform", activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(60, kernel_initializer="he_uniform", activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(32, kernel_initializer="he_uniform", activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(8, kernel_initializer="he_uniform", activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(loss="mae", optimizer="nadam")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=200,
        batch_size=32,
        verbose=0,
    )
    prediction_scaled = model.predict(x_test, verbose=0)
    predictions = y_scaler.inverse_transform(prediction_scaled).ravel()
    metrics = metrics_report(test[COL_ALKALINITY].to_numpy(), predictions)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            COL_DATE: test[COL_DATE],
            "unit_id": test["unit_id"],
            "observed": test[COL_ALKALINITY],
            "prediction": predictions,
        }
    ).to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    summary = {
        "data_path": args.data_path,
        "target_column": COL_ALKALINITY,
        "shift_days": 0,
        "features": FEATURE_COLS,
        "split_method": "random",
        "random_state": cfg.random_state,
        "zscore_limit": 3.0,
        "architecture": [72, 60, 32, 8],
        "dropout": 0.1,
        "optimizer": "nadam",
        "loss": "mae",
        "epochs": 200,
        "batch_size": 32,
        "rows": {
            "after_zscore": len(df),
            "train": len(train),
            "validation": len(val),
            "test": len(test),
        },
        "test_rows_match_rf": True,
        "test_metrics": metrics,
        "last_training_loss_scaled": float(history.history["loss"][-1]),
        "last_validation_loss_scaled": float(history.history["val_loss"][-1]),
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
