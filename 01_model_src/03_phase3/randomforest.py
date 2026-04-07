"""Random Forest v2 pipeline for alkalinity forecasting.

This script provides a reproducible, leakage-safe pipeline with:
- deterministic data cleaning
- time-aware splitting (train/validation/test)
- hyperparameter tuning with GridSearchCV/RandomizedSearchCV + TimeSeriesSplit
- final evaluation on a strict holdout test window
- export of metrics and model configuration

Naming convention:
- One run can train multiple forecast horizons using --shift-days.
- Each horizon is written to output/shift_<N>/.
- Aggregate overview is written to output/all_runs_summary.json.

Sample command:
python randomforest.py --data-path ../../../dataset/data_4perday_cleaned.csv --output-dir ./output --shift-days 1,2,3 --search-method grid --cv-folds 5 --zscore-limit 3
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Dataset column names (kept as unicode escapes to keep source ASCII-only).
COL_DATE = "Date"
COL_CROP = "V\u1ee5 nu\xf4i"
COL_MODULE = "module_name"
COL_POND = "ao"
COL_STOCKING_DAY = "Ng\xe0y th\u1ea3"
COL_TIME = "Time"
COL_TEMP = "Nhi\u1ec7t \u0111\u1ed9"
COL_PH = "pH"
COL_SALINITY = "\u0110\u1ed9 m\u1eb7n"
COL_TDS = "TDS"
COL_TURBIDITY = "\u0110\u1ed9 \u0111\u1ee5c"
COL_DO = "DO"
COL_COLOR = "\u0110\u1ed9 m\xe0u"
COL_TRANSPARENCY = "\u0110\u1ed9 trong"
COL_ALKALINITY = "\u0110\u1ed9 ki\u1ec1m"
COL_HARDNESS = "\u0110\u1ed9 c\u1ee9ng"
COL_POND_TYPE = "Lo\u1ea1i ao"
COL_FARM_TECH = "C\xf4ng ngh\u1ec7 nu\xf4i"
COL_AREA = "area"
COL_SEED = "Gi\u1ed1ng t\xf4m"
COL_SHRIMP_AGE = "Tu\u1ed5i t\xf4m"
COL_WATER_LEVEL = "M\u1ef1c n\u01b0\u1edbc"
COL_AMMONIA = "Amoni"
COL_NITRATE = "Nitrat"
COL_NITRITE = "Nitrit"
COL_SILICA = "Silica"
COL_SEASON = "Season"


@dataclass
class Config:
    data_path: str = "../../../dataset/data_4perday_cleaned.csv"
    output_dir: str = "./output"
    random_state: int = 42
    test_ratio: float = 0.2
    val_ratio_within_trainval: float = 0.25
    shift_days: str = "1,2,3"
    search_method: str = "grid"
    cv_folds: int = 5
    random_search_iter: int = 60
    zscore_limit: float = 3.0


def build_unit_id(df: pd.DataFrame) -> pd.Series:
    return (
        df[COL_CROP].astype(str).str.replace(" ", "", regex=False)
        + "-"
        + df[COL_MODULE].astype(str)
        + "-"
        + df[COL_POND].astype(str)
    )


def load_and_clean_data(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.data_path)

    required_cols = [
        COL_DATE,
        COL_CROP,
        COL_MODULE,
        COL_POND,
        COL_SEASON,
        COL_POND_TYPE,
        COL_FARM_TECH,
        COL_SEED,
        COL_SHRIMP_AGE,
        COL_WATER_LEVEL,
        COL_TEMP,
        COL_PH,
        COL_SALINITY,
        COL_TDS,
        COL_COLOR,
        COL_TRANSPARENCY,
        COL_HARDNESS,
        COL_ALKALINITY,
        COL_AMMONIA,
        COL_NITRATE,
        COL_NITRITE,
        COL_SILICA,
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input data: {missing}")

    df[COL_DATE] = pd.to_datetime(df[COL_DATE], dayfirst=True, errors="coerce")
    df[COL_SHRIMP_AGE] = pd.to_numeric(df[COL_SHRIMP_AGE], errors="coerce")
    df[COL_WATER_LEVEL] = pd.to_numeric(df[COL_WATER_LEVEL], errors="coerce")
    df.loc[df[COL_WATER_LEVEL] == 0, COL_WATER_LEVEL] = np.nan
    df[COL_WATER_LEVEL] = df[COL_WATER_LEVEL].fillna(df[COL_WATER_LEVEL].median())

    df["unit_id"] = build_unit_id(df)

    keep = [
        COL_DATE,
        "unit_id",
        COL_SEASON,
        COL_POND_TYPE,
        COL_FARM_TECH,
        COL_SEED,
        COL_SHRIMP_AGE,
        COL_WATER_LEVEL,
        COL_TEMP,
        COL_PH,
        COL_SALINITY,
        COL_TDS,
        COL_COLOR,
        COL_TRANSPARENCY,
        COL_HARDNESS,
        COL_ALKALINITY,
        COL_AMMONIA,
        COL_NITRATE,
        COL_NITRITE,
        COL_SILICA,
    ]

    df = df[keep].copy()
    df = df.sort_values(["unit_id", COL_DATE]).dropna().reset_index(drop=True)
    return df


def apply_zscore_filter(df: pd.DataFrame, numeric_cols: List[str], zscore_limit: float) -> pd.DataFrame:
    """Filter out rows with |z-score| >= zscore_limit on any numeric feature.

    Set zscore_limit <= 0 to disable this filter.
    """
    if zscore_limit <= 0:
        return df.copy()

    df_num = df[numeric_cols]
    std = df_num.std(ddof=0).replace(0, np.nan)
    z = (df_num - df_num.mean()) / std
    mask = z.abs().lt(zscore_limit).all(axis=1)
    mask = mask.fillna(False)
    return df.loc[mask].reset_index(drop=True)


def prepare_shift_target(df: pd.DataFrame, shift_day: int) -> Tuple[pd.DataFrame, str]:
    target_col = f"target_alkalinity_shift_{shift_day}"
    df_shift = df.copy()
    df_shift[target_col] = df_shift.groupby("unit_id")[COL_ALKALINITY].shift(-shift_day)
    df_shift = df_shift.dropna().reset_index(drop=True)
    return df_shift, target_col


def split_time_aware(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_values(COL_DATE).reset_index(drop=True)
    n = len(df_sorted)
    n_test = max(1, int(n * cfg.test_ratio))
    n_trainval = n - n_test
    n_val = max(1, int(n_trainval * cfg.val_ratio_within_trainval))

    train = df_sorted.iloc[: n_trainval - n_val].copy()
    val = df_sorted.iloc[n_trainval - n_val : n_trainval].copy()
    test = df_sorted.iloc[n_trainval:].copy()
    return train, val, test


def make_pipeline(categorical_cols: List[str], numeric_cols: List[str], random_state: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    model = RandomForestRegressor(random_state=random_state)

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def get_search_space() -> Dict[str, List]:
    return {
        "model__n_estimators": [200, 300, 500, 800],
        "model__max_depth": [10, 20, 30, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 5],
        "model__max_features": ["sqrt", "log2", None],
        "model__bootstrap": [True, False],
    }


def metrics_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    eps = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100
    bias = float(np.mean(y_pred - y_true))
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mape),
        "r2": float(r2_score(y_true, y_pred)),
        "bias": bias,
    }


def parse_shift_days(shift_days: str) -> List[int]:
    values = []
    for token in shift_days.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid shift value: '{token}'") from exc
        if value <= 0:
            raise ValueError("All shift values must be positive integers.")
        values.append(value)

    if not values:
        raise ValueError("No valid shift values found.")
    return sorted(set(values))


def run_single_shift(
    df_base: pd.DataFrame,
    cfg: Config,
    shift_day: int,
    categorical_cols: List[str],
    numeric_cols: List[str],
    out_dir: Path,
) -> Dict[str, object]:
    df_shift, target_col = prepare_shift_target(df_base, shift_day)
    train_df, val_df, test_df = split_time_aware(df_shift, cfg)
    trainval_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

    X_trainval = trainval_df[categorical_cols + numeric_cols]
    y_trainval = trainval_df[target_col].to_numpy()
    X_test = test_df[categorical_cols + numeric_cols]
    y_test = test_df[target_col].to_numpy()

    pipe = make_pipeline(categorical_cols, numeric_cols, cfg.random_state)
    tscv = TimeSeriesSplit(n_splits=cfg.cv_folds)

    search_space = get_search_space()
    if cfg.search_method == "grid":
        search = GridSearchCV(
            estimator=pipe,
            param_grid=search_space,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=1,
            refit=True,
        )
    else:
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=search_space,
            n_iter=cfg.random_search_iter,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            random_state=cfg.random_state,
            n_jobs=-1,
            verbose=1,
            refit=True,
        )
    search.fit(X_trainval, y_trainval)
    y_pred_test = search.best_estimator_.predict(X_test)

    shift_out_dir = out_dir / f"shift_{shift_day}"
    shift_out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "shift_day": shift_day,
        "target_column": target_col,
        "config": asdict(cfg),
        "search_method": cfg.search_method,
        "n_rows_after_shift": int(len(df_shift)),
        "split_sizes": {
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "best_params": search.best_params_,
        "cv_best_rmse": float(-search.best_score_),
        "test_metrics": metrics_report(y_test, y_pred_test),
    }

    (shift_out_dir / "run_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    pred_df = test_df[[COL_DATE, "unit_id", target_col]].copy()
    pred_df["prediction"] = y_pred_test
    pred_df.to_csv(shift_out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="RandomForest v2 reproducible pipeline")
    parser.add_argument("--data-path", type=str, default=Config.data_path)
    parser.add_argument("--output-dir", type=str, default=Config.output_dir)
    parser.add_argument(
        "--shift-days",
        type=str,
        default=Config.shift_days,
        help="Comma-separated forecast shifts, for example: 1,2,3",
    )
    parser.add_argument(
        "--search-method",
        type=str,
        default=Config.search_method,
        choices=["grid", "random"],
        help="Hyperparameter tuning strategy.",
    )
    parser.add_argument("--random-state", type=int, default=Config.random_state)
    parser.add_argument("--cv-folds", type=int, default=Config.cv_folds)
    parser.add_argument("--random-search-iter", type=int, default=Config.random_search_iter)
    parser.add_argument(
        "--zscore-limit",
        type=float,
        default=Config.zscore_limit,
        help="Absolute z-score threshold for numeric outlier filtering; <=0 disables filter.",
    )
    args = parser.parse_args()

    cfg = Config(
        data_path=args.data_path,
        output_dir=args.output_dir,
        shift_days=args.shift_days,
        search_method=args.search_method,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
        random_search_iter=args.random_search_iter,
        zscore_limit=args.zscore_limit,
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_base = load_and_clean_data(cfg)
    shift_days = parse_shift_days(cfg.shift_days)

    categorical_cols = [COL_SEASON, COL_POND_TYPE, COL_FARM_TECH, COL_SEED]
    numeric_cols = [
        COL_SHRIMP_AGE,
        COL_WATER_LEVEL,
        COL_TEMP,
        COL_PH,
        COL_SALINITY,
        COL_TDS,
        COL_COLOR,
        COL_TRANSPARENCY,
        COL_HARDNESS,
        COL_ALKALINITY,
        COL_AMMONIA,
        COL_NITRATE,
        COL_NITRITE,
        COL_SILICA,
    ]
    n_rows_before_zscore = len(df_base)
    df_base = apply_zscore_filter(df_base, numeric_cols=numeric_cols, zscore_limit=cfg.zscore_limit)
    n_rows_after_zscore = len(df_base)

    all_results = []
    for shift_day in shift_days:
        print(f"Running shift_day={shift_day}")
        all_results.append(
            run_single_shift(
                df_base=df_base,
                cfg=cfg,
                shift_day=shift_day,
                categorical_cols=categorical_cols,
                numeric_cols=numeric_cols,
                out_dir=out_dir,
            )
        )

    (out_dir / "all_runs_summary.json").write_text(
        json.dumps(
            {
                "global_config": asdict(cfg),
                "n_rows_before_zscore": int(n_rows_before_zscore),
                "n_rows_after_zscore": int(n_rows_after_zscore),
                "runs": all_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Done. Outputs:")
    print(f"- {out_dir / 'all_runs_summary.json'}")
    for shift_day in shift_days:
        print(f"- {out_dir / f'shift_{shift_day}' / 'run_summary.json'}")
        print(f"- {out_dir / f'shift_{shift_day}' / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
