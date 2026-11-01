from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "data.parquet"
CONVERTED_DATA_PATH = ROOT / "data" / "data_cny.parquet"
MODEL_PATH = ROOT / "models" / "price_model.joblib"
METRICS_PATH = ROOT / "metrics.json"
RANDOM_STATE = 42
USD_TO_CNY = 6.6702
EXCHANGE_RATE_DATE = "2026-10-23"


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame = frame.dropna(subset=["log_price"])
    frame = frame[frame["log_price"].between(0, 8)]
    frame["name"] = frame["name"].fillna("").astype(str)
    frame["item_description"] = frame["item_description"].fillna("").astype(str)
    for column in ["cat1", "cat2", "cat3", "brand_name"]:
        frame[column] = frame[column].astype(object).where(frame[column].notna(), "Unknown").astype(str)
    frame["item_condition_id"] = pd.to_numeric(
        frame["item_condition_id"], errors="coerce"
    ).fillna(3)
    frame["shipping"] = pd.to_numeric(frame["shipping"], errors="coerce").fillna(0)
    return frame


def build_features() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "name_text",
                TfidfVectorizer(
                    max_features=20_000,
                    ngram_range=(1, 2),
                    min_df=2,
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
                "name",
            ),
            (
                "description_text",
                TfidfVectorizer(
                    max_features=30_000,
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.98,
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
                "item_description",
            ),
            (
                "categories_and_brand",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                ["cat1", "cat2", "cat3", "brand_name"],
            ),
            (
                "condition_and_shipping",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                ["item_condition_id", "shipping"],
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )


def build_model(alpha: float) -> Pipeline:
    return Pipeline([("features", build_features()), ("regressor", Ridge(alpha=alpha, solver="lsqr"))])


def price_band_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, dict[str, float]]:
    bands = [(0, 50, "0-50"), (50, 100, "50-100"), (100, 300, "100-300"), (300, float("inf"), "300+")]
    result = {}
    for lower, upper, label in bands:
        mask = (actual >= lower) & (actual < upper)
        if not np.any(mask):
            continue
        result[label] = {
            "count": int(mask.sum()),
            "mae_cny": float(mean_absolute_error(actual[mask], predicted[mask])),
            "median_relative_error_pct": float(
                np.median(np.abs(predicted[mask] - actual[mask]) / np.maximum(actual[mask], 1.0)) * 100
            ),
        }
    return result


def price_band_key(value: float) -> str:
    if value < 50:
        return "0-50"
    if value < 100:
        return "50-100"
    if value < 300:
        return "100-300"
    return "300+"


def _interval_by_mask(residual: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    result: dict[str, object] = {}
    labels = ["0-50", "50-100", "100-300", "300+"]
    for label in labels:
        label_mask = mask == label
        if np.any(label_mask):
            result[label] = {
                "count": int(label_mask.sum()),
                "residual_p10_cny": float(np.percentile(residual[label_mask], 10)),
                "residual_p90_cny": float(np.percentile(residual[label_mask], 90)),
            }
    return result


def residual_intervals(actual: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    residual = actual - predicted
    actual_bands = np.array([price_band_key(value) for value in actual])
    predicted_bands = np.array([price_band_key(value) for value in predicted])
    return {
        "p10_cny": float(np.percentile(residual, 10)),
        "p90_cny": float(np.percentile(residual, 90)),
        "by_actual_price_band": _interval_by_mask(residual, actual_bands),
        # 估价时只能看到预测价格，因此区间应按预测价格段选择。
        "by_predicted_price_band": _interval_by_mask(residual, predicted_bands),
        "calibration_split": "training data validation split",
        "coverage_target_pct": 80,
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    original_frame = pd.read_parquet(DATA_PATH)
    frame = clean_data(original_frame)
    frame["price_cny"] = np.expm1(frame["log_price"]).clip(lower=0) * USD_TO_CNY
    frame["log_price_cny"] = np.log1p(frame["price_cny"])
    converted_frame = original_frame.copy()
    converted_frame["price_cny"] = np.expm1(converted_frame["log_price"]).clip(lower=0) * USD_TO_CNY
    converted_frame["log_price_cny"] = np.log1p(converted_frame["price_cny"])
    CONVERTED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    converted_frame.to_parquet(CONVERTED_DATA_PATH, index=False)
    columns = [
        "name",
        "item_description",
        "cat1",
        "cat2",
        "cat3",
        "brand_name",
        "item_condition_id",
        "shipping",
    ]
    train_x, test_x, train_y, test_y = train_test_split(
        frame[columns],
        frame["log_price_cny"].astype(float),
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    # Select regularization on a validation split drawn only from the training data.
    inner_train_idx, validation_idx = train_test_split(
        np.arange(len(train_x)), test_size=0.2, random_state=RANDOM_STATE
    )
    feature_builder = build_features()
    inner_train_x = feature_builder.fit_transform(train_x.iloc[inner_train_idx])
    validation_x = feature_builder.transform(train_x.iloc[validation_idx])
    inner_train_y = train_y.iloc[inner_train_idx].to_numpy()
    validation_y = train_y.iloc[validation_idx].to_numpy()
    alpha_scores = []
    for alpha in [0.3, 1.0, 3.0, 10.0]:
        candidate = Ridge(alpha=alpha, solver="lsqr")
        candidate.fit(inner_train_x, inner_train_y)
        validation_price = np.expm1(candidate.predict(validation_x)).clip(min=0)
        actual_validation_price = np.expm1(validation_y).clip(min=0)
        alpha_scores.append((
            float(mean_absolute_error(actual_validation_price, validation_price)),
            alpha,
        ))
    validation_mae, best_alpha = min(alpha_scores)

    calibration_model = Ridge(alpha=best_alpha, solver="lsqr")
    calibration_model.fit(inner_train_x, inner_train_y)
    validation_predicted_cny = np.expm1(calibration_model.predict(validation_x)).clip(min=0)
    validation_actual_cny = np.expm1(validation_y).clip(min=0)
    interval_metrics = residual_intervals(validation_actual_cny, validation_predicted_cny)

    model = build_model(alpha=best_alpha)
    model.fit(train_x, train_y)
    predicted_log = model.predict(test_x)
    actual_cny = np.expm1(test_y.to_numpy()).clip(min=0)
    predicted_cny = np.expm1(predicted_log).clip(min=0)
    metrics = {
        "rows_total": int(len(frame)),
        "rows_train": int(len(train_x)),
        "rows_test": int(len(test_x)),
        "mae_cny": float(mean_absolute_error(actual_cny, predicted_cny)),
        "rmse_cny": float(np.sqrt(mean_squared_error(actual_cny, predicted_cny))),
        "r2_cny": float(r2_score(actual_cny, predicted_cny)),
        "price_band_metrics": price_band_metrics(actual_cny, predicted_cny),
        "model": "Ridge regression",
        "ridge_alpha": best_alpha,
        "validation_mae_cny": float(validation_mae),
        "alpha_validation_scores": {str(alpha): float(mae) for mae, alpha in alpha_scores},
        "target": "log_price_cny = ln(price_cny + 1)",
        "currency": "CNY",
        "usd_to_cny": USD_TO_CNY,
        "exchange_rate_date": EXCHANGE_RATE_DATE,
        "exchange_rate_source": "Frankfurter / ECB reference rates",
        "dataset_note": "Mercari US listings converted from USD to CNY; results are not campus transaction prices.",
        "residual_intervals": interval_metrics,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": columns}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
