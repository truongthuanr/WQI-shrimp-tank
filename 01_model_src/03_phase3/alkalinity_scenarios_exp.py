"""Evaluate measured versus simulated source alkalinity for Table 3.12.

Input: morning records from the cleaned phase-3 CSV.
Output: per-scenario metrics and test predictions for noon, afternoon, next day.
Split: shared seeded random 60/20/20 within each horizon; simulated alkalinity
is cross-fitted on training rows and predicted out of sample on test rows.
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    COL_SEED,
    COL_SHRIMP_AGE,
    COL_TEMP,
    COL_TRANSPARENCY,
    COL_WATER_LEVEL,
    Config,
    apply_zscore_filter,
    build_unit_id,
    metrics_report,
)
from variable_selection_base_exp import CATEGORICAL, FEATURE_SETS


TIME_COL = "Time"
STOCKING_COL = "Ngày thả"
ALK_INPUT = "alkalinity_input"
TARGET = "target_alkalinity"
STAGE1_FEATURES = [
    COL_SEASON, COL_POND_TYPE, COL_FARM_TECH, COL_SEED, STOCKING_COL,
    COL_AREA, COL_SHRIMP_AGE, COL_TEMP, COL_PH, COL_SALINITY,
    COL_WATER_LEVEL, COL_TRANSPARENCY,
]
STAGE1_CATEGORICAL = {COL_SEASON, COL_POND_TYPE, COL_FARM_TECH, COL_SEED}
BASE_SETS = {
    (model, size): features
    for (model, size), features in FEATURE_SETS.items()
    if model in {"ANN", "RF", "GBT"}
}
ALL_FEATURES = list(dict.fromkeys(
    feature for features in list(BASE_SETS.values()) + [STAGE1_FEATURES]
    for feature in features
))
ALL_NUMERIC = [col for col in ALL_FEATURES if col not in STAGE1_CATEGORICAL]


def read_source(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    needed = [COL_DATE, COL_CROP, COL_MODULE, COL_POND, TIME_COL, COL_ALKALINITY] + ALL_FEATURES
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], dayfirst=True, errors="coerce")
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    for col in ALL_NUMERIC + [COL_ALKALINITY]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.loc[df[COL_WATER_LEVEL] == 0, COL_WATER_LEVEL] = np.nan
    df[COL_WATER_LEVEL] = df[COL_WATER_LEVEL].fillna(df[COL_WATER_LEVEL].median())
    df["unit_id"] = build_unit_id(df)
    return df


def last_record(df: pd.DataFrame, lower: float, upper: float | None) -> pd.DataFrame:
    mask = df[TIME_COL].ge(lower)
    if upper is not None:
        mask &= df[TIME_COL].lt(upper)
    return (
        df.loc[mask].sort_values(["unit_id", COL_DATE, TIME_COL])
        .drop_duplicates(["unit_id", COL_DATE], keep="last")
    )


def horizon_data(df: pd.DataFrame, horizon: str) -> tuple[pd.DataFrame, dict]:
    morning = last_record(df, 0.0, 9.0)
    if horizon == "noon":
        target = last_record(df, 9.0, 15.0)
    elif horizon == "afternoon":
        target = last_record(df, 15.0, None)
    elif horizon == "next_day":
        target = morning.copy()
        target[COL_DATE] = target[COL_DATE] - pd.Timedelta(days=1)
    else:
        raise ValueError(f"Unknown horizon: {horizon}")
    target = target[["unit_id", COL_DATE, COL_ALKALINITY]].rename(
        columns={COL_ALKALINITY: TARGET}
    )
    source_cols = ["unit_id", COL_DATE, COL_ALKALINITY] + ALL_FEATURES
    paired = morning[source_cols].merge(
        target, on=["unit_id", COL_DATE], how="inner", validate="one_to_one"
    )
    n_paired = len(paired)
    paired = paired.dropna().reset_index(drop=True)
    n_clean = len(paired)
    paired, n_outliers = apply_zscore_filter(paired, ALL_NUMERIC, 3.0)
    counts = {
        "paired": n_paired, "after_dropna": n_clean,
        "after_zscore": len(paired), "excluded_outliers": n_outliers,
    }
    return paired, counts


def make_preprocessor(features: list[str], scaled: bool) -> ColumnTransformer:
    cats = [col for col in features if col in STAGE1_CATEGORICAL]
    nums = [col for col in features if col not in STAGE1_CATEGORICAL]
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
        ("num", StandardScaler() if scaled else "passthrough", nums),
    ])


def make_stage1(seed: int) -> Pipeline:
    pipeline = Pipeline([
        ("preprocess", make_preprocessor(STAGE1_FEATURES, scaled=False)),
        ("model", RandomForestRegressor(random_state=seed)),
    ])
    pipeline.set_params(**BEST_PARAMS)
    return pipeline


def simulated_source_alkalinity(trainval: pd.DataFrame, test: pd.DataFrame, seed: int):
    folds = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.full(len(trainval), np.nan)
    for fit_idx, pred_idx in folds.split(trainval):
        model = make_stage1(seed)
        model.fit(
            trainval.iloc[fit_idx][STAGE1_FEATURES],
            trainval.iloc[fit_idx][COL_ALKALINITY],
        )
        oof[pred_idx] = model.predict(trainval.iloc[pred_idx][STAGE1_FEATURES])
    if np.isnan(oof).any():
        raise RuntimeError("Stage-one cross-fitting left missing predictions")
    model = make_stage1(seed)
    model.fit(trainval[STAGE1_FEATURES], trainval[COL_ALKALINITY])
    test_prediction = model.predict(test[STAGE1_FEATURES])
    return oof, test_prediction, {
        "trainval_oof_mae": float(np.mean(np.abs(trainval[COL_ALKALINITY] - oof))),
        "test_mae": float(np.mean(np.abs(test[COL_ALKALINITY] - test_prediction))),
    }


def fit_predict(model_name: str, features: list[str], trainval: pd.DataFrame,
                test: pd.DataFrame, seed: int):
    if model_name == "ANN":
        import tensorflow as tf

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        train, val = train_test_split(
            trainval, test_size=0.25, random_state=seed
        )
        preprocessor = make_preprocessor(features, scaled=True)
        x_train = preprocessor.fit_transform(train[features]).astype(np.float32)
        x_val = preprocessor.transform(val[features]).astype(np.float32)
        x_test = preprocessor.transform(test[features]).astype(np.float32)
        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(train[[TARGET]]).astype(np.float32)
        y_val = y_scaler.transform(val[[TARGET]]).astype(np.float32)
        layers = [tf.keras.layers.Input(shape=(x_train.shape[1],))]
        for units in (72, 60, 32, 8):
            layers.extend([
                tf.keras.layers.Dense(units, kernel_initializer="he_uniform", activation="relu"),
                tf.keras.layers.Dropout(0.1),
            ])
        layers.append(tf.keras.layers.Dense(1))
        model = tf.keras.Sequential(layers)
        model.compile(loss="mae", optimizer="nadam")
        model.fit(x_train, y_train, validation_data=(x_val, y_val),
                  epochs=200, batch_size=32, verbose=0)
        return y_scaler.inverse_transform(model.predict(x_test, verbose=0)).ravel()

    if model_name == "RF":
        pipeline = Pipeline([
            ("preprocess", make_preprocessor(features, scaled=False)),
            ("model", RandomForestRegressor(random_state=seed)),
        ])
        pipeline.set_params(**BEST_PARAMS)
    else:
        pipeline = Pipeline([
            ("preprocess", make_preprocessor(features, scaled=False)),
            ("model", GradientBoostingRegressor(
                n_estimators=400, learning_rate=0.1, max_depth=3, random_state=seed
            )),
        ])
    pipeline.fit(trainval[features], trainval[TARGET])
    return pipeline.predict(test[features])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="../../../dataset/data_4perday_cleaned.csv")
    parser.add_argument("--output-dir", default="./output_alkalinity_scenarios")
    parser.add_argument("--horizons", default="noon,afternoon,next_day")
    args = parser.parse_args()
    horizons = [token.strip() for token in args.horizons.split(",") if token.strip()]
    if not horizons or any(h not in {"noon", "afternoon", "next_day"} for h in horizons):
        parser.error("Invalid --horizons; use noon,afternoon,next_day")

    cfg = Config(data_path=args.data_path)
    raw = read_source(args.data_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for horizon in horizons:
        data, counts = horizon_data(raw, horizon)
        trainval, test = train_test_split(
            data, test_size=cfg.test_ratio, random_state=cfg.random_state
        )
        counts.update({"trainval": len(trainval), "test": len(test)})
        simulated_trainval, simulated_test, stage1_metrics = simulated_source_alkalinity(
            trainval, test, cfg.random_state
        )
        for model_name in ("ANN", "RF", "GBT"):
            for size in (8, 9):
                base_features = BASE_SETS[(model_name, size)]
                features = base_features + [ALK_INPUT]
                for alk_type in ("measured", "simulated"):
                    case_trainval = trainval.copy()
                    case_test = test.copy()
                    if alk_type == "measured":
                        case_trainval[ALK_INPUT] = case_trainval[COL_ALKALINITY]
                        case_test[ALK_INPUT] = case_test[COL_ALKALINITY]
                    else:
                        case_trainval[ALK_INPUT] = simulated_trainval
                        case_test[ALK_INPUT] = simulated_test
                    predictions = fit_predict(
                        model_name, features, case_trainval, case_test, cfg.random_state
                    )
                    metrics = metrics_report(test[TARGET].to_numpy(), predictions)
                    case_id = f"{model_name.lower()}_{size}_{alk_type}"
                    case_dir = out_dir / horizon / case_id
                    case_dir.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame({
                        "unit_id": test["unit_id"],
                        "source_date": test[COL_DATE],
                        "source_alkalinity": test[COL_ALKALINITY],
                        "input_alkalinity": case_test[ALK_INPUT],
                        "observed_target": test[TARGET],
                        "prediction": predictions,
                    }).to_csv(case_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
                    result = {
                        "case": case_id,
                        "horizon": horizon,
                        "source_time": "last record before 09:00",
                        "target_time": {
                            "noon": "last same-day record from 09:00 to before 15:00",
                            "afternoon": "last same-day record at or after 15:00",
                            "next_day": "morning record exactly one calendar day later",
                        }[horizon],
                        "features": features,
                        "alkalinity_input": alk_type,
                        "simulated_alkalinity_method": "5-fold out-of-fold RF on training rows; RF trained on all trainval predicts test rows",
                        "stage1_features": STAGE1_FEATURES,
                        "stage1_metrics": stage1_metrics,
                        "seed": cfg.random_state,
                        "split": "random trainval/test 80/20; ANN uses 60/20/20",
                        "rows": counts,
                        "test_metrics": metrics,
                    }
                    (case_dir / "run_summary.json").write_text(
                        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    all_results.append(result)
                    print(
                        f"{horizon} {case_id}: n={len(test)} R2={metrics['r2']:.4f} "
                        f"RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f}",
                        flush=True,
                    )
    (out_dir / "all_runs_summary.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
