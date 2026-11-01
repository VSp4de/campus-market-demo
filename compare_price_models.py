"""Compare price regressors on the deployed model's fixed Mercari test split.

Requires the optional xgboost package. The comparison does not replace the
deployed Ridge model or modify the source dataset.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from train_model import RANDOM_STATE, USD_TO_CNY, clean_data


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "model_comparison.json"


def score(actual: np.ndarray, predicted_log: np.ndarray) -> dict[str, float]:
    predicted = np.expm1(predicted_log).clip(min=0)
    return {
        "mae_cny": float(mean_absolute_error(actual, predicted)),
        "rmse_cny": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2_cny": float(r2_score(actual, predicted)),
    }


def main() -> None:
    frame = clean_data(pd.read_parquet(ROOT / "data" / "data.parquet"))
    frame["price_cny"] = np.expm1(frame["log_price"]).clip(lower=0) * USD_TO_CNY
    frame["log_price_cny"] = np.log1p(frame["price_cny"])
    train_idx, test_idx = train_test_split(
        np.arange(len(frame)), test_size=0.2, random_state=RANDOM_STATE
    )
    train = frame.iloc[train_idx]
    test = frame.iloc[test_idx]
    actual = test["price_cny"].to_numpy(dtype=float)
    target = train["log_price_cny"].to_numpy(dtype=float)

    bundle = joblib.load(ROOT / "models" / "price_model.joblib")
    deployed = bundle["model"]
    columns = bundle["feature_columns"]
    features = deployed.named_steps["features"]
    x_train = features.transform(train[columns])
    x_test = features.transform(test[columns])
    print(f"Samples: {len(train)} train, {len(test)} test; features: {x_train.shape[1]}", flush=True)

    category_medians = train.groupby("cat1")["price_cny"].median()
    baseline_price = test["cat1"].map(category_medians).fillna(train["price_cny"].median()).to_numpy(dtype=float)
    results = {
        "category_median": {
            "mae_cny": float(mean_absolute_error(actual, baseline_price)),
            "rmse_cny": float(np.sqrt(mean_squared_error(actual, baseline_price))),
            "r2_cny": float(r2_score(actual, baseline_price)),
        },
        "ridge": score(actual, deployed.named_steps["regressor"].predict(x_test)),
    }

    candidates = {
        "random_forest": RandomForestRegressor(
            n_estimators=40, max_depth=18, min_samples_leaf=5,
            max_features=0.2, n_jobs=4, random_state=RANDOM_STATE,
        ),
        "xgboost": XGBRegressor(
            n_estimators=180, max_depth=6, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.8, reg_lambda=5.0,
            tree_method="hist", n_jobs=6, random_state=RANDOM_STATE,
            objective="reg:squarederror",
        ),
    }
    for name, model in candidates.items():
        print(f"Training {name}...", flush=True)
        started = time.perf_counter()
        model.fit(x_train, target)
        results[name] = score(actual, model.predict(x_test))
        results[name]["train_seconds"] = round(time.perf_counter() - started, 1)
        print(name, json.dumps(results[name], ensure_ascii=False), flush=True)

    output = {
        "dataset": "Mercari 100,000 cleaned listings; prices converted to CNY",
        "split": {"random_state": RANDOM_STATE, "train_rows": len(train), "test_rows": len(test)},
        "features": "Same fitted TF-IDF, one-hot, and numeric transformer as deployed Ridge model",
        "feature_count": x_train.shape[1],
        "target": "ln(price_cny + 1); evaluated after inverse transform in CNY",
        "note": "Fixed illustrative hyperparameters for tree models; no test-set tuning. Ridge remains deployed.",
        "hyperparameters": {
            "random_forest": {
                "n_estimators": 40, "max_depth": 18, "min_samples_leaf": 5,
                "max_features": 0.2, "random_state": RANDOM_STATE,
            },
            "xgboost": {
                "n_estimators": 180, "max_depth": 6, "learning_rate": 0.08,
                "subsample": 0.85, "colsample_bytree": 0.8,
                "reg_lambda": 5.0, "tree_method": "hist", "random_state": RANDOM_STATE,
            },
        },
        "models": results,
    }
    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Saved: {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
