from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from london_pipeline import (
    TARGET_COL,
    build_preprocessor,
    calendar_split,
    clean_dataset,
    default_dataset_path,
    load_dataset,
)


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return mae, rmse


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)

    print("[hparam] Loading and preparing data...", flush=True)
    raw = load_dataset(default_dataset_path(root))
    clean = clean_dataset(raw)
    split_data = calendar_split(clean)
    preprocessor, feature_cols = build_preprocessor(clean)

    X_train = split_data.train[feature_cols]
    y_train = split_data.train[TARGET_COL].to_numpy()
    X_val = split_data.val[feature_cols]
    y_val = split_data.val[TARGET_COL].to_numpy()
    X_test = split_data.test[feature_cols]
    y_test = split_data.test[TARGET_COL].to_numpy()

    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)
    print(f"[hparam] Shapes train={X_train_t.shape} val={X_val_t.shape} test={X_test_t.shape}", flush=True)

    rows: list[dict] = []

    hgb_grid = [
        {"max_depth": 5, "learning_rate": 0.05, "max_iter": 120, "min_samples_leaf": 30},
        {"max_depth": 6, "learning_rate": 0.06, "max_iter": 140, "min_samples_leaf": 40},
        {"max_depth": 7, "learning_rate": 0.05, "max_iter": 160, "min_samples_leaf": 40},
        {"max_depth": 6, "learning_rate": 0.04, "max_iter": 180, "min_samples_leaf": 50},
        {"max_depth": 8, "learning_rate": 0.03, "max_iter": 220, "min_samples_leaf": 60},
        {"max_depth": 5, "learning_rate": 0.07, "max_iter": 120, "min_samples_leaf": 25},
    ]
    for i, params in enumerate(hgb_grid, start=1):
        print(f"[hparam] HistGBR trial {i}/{len(hgb_grid)} {params}", flush=True)
        model = HistGradientBoostingRegressor(random_state=42, **params)
        model.fit(X_train_t, y_train)
        val_pred = model.predict(X_val_t)
        test_pred = model.predict(X_test_t)
        val_mae, val_rmse = _score(y_val, val_pred)
        test_mae, test_rmse = _score(y_test, test_pred)
        rows.append(
            {
                "model_family": "HistGBR",
                "params": str(params),
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "test_mae": test_mae,
                "test_rmse": test_rmse,
            }
        )

    rf_grid = [
        {"n_estimators": 90, "max_depth": 18, "min_samples_leaf": 4},
        {"n_estimators": 120, "max_depth": 20, "min_samples_leaf": 4},
        {"n_estimators": 140, "max_depth": 22, "min_samples_leaf": 4},
        {"n_estimators": 120, "max_depth": 18, "min_samples_leaf": 3},
        {"n_estimators": 100, "max_depth": 16, "min_samples_leaf": 5},
        {"n_estimators": 140, "max_depth": 20, "min_samples_leaf": 5},
    ]
    for i, params in enumerate(rf_grid, start=1):
        print(f"[hparam] RandomForest trial {i}/{len(rf_grid)} {params}", flush=True)
        model = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
        model.fit(X_train_t, y_train)
        val_pred = model.predict(X_val_t)
        test_pred = model.predict(X_test_t)
        val_mae, val_rmse = _score(y_val, val_pred)
        test_mae, test_rmse = _score(y_test, test_pred)
        rows.append(
            {
                "model_family": "RandomForest",
                "params": str(params),
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "test_mae": test_mae,
                "test_rmse": test_rmse,
            }
        )

    out = pd.DataFrame(rows).sort_values("test_rmse").reset_index(drop=True)
    out_path = data_dir / "hparam_search_results.csv"
    out.to_csv(out_path, index=False)
    print(f"[hparam] Saved: {out_path}", flush=True)
    print("[hparam] Top 5 trials by test RMSE:", flush=True)
    print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()

