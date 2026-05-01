from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from london_pipeline import TARGET_COL, calendar_split, clean_dataset, default_dataset_path, load_dataset


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return mae, rmse


def _assisted_features(clean: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    drop_cols = {
        TARGET_COL,
        "history_date",
        "fullAddress",
        "postcode",
        "outcode",
        "country",
        "history_percentageChange",
        "history_numericChange",
        "saleEstimate_valueChange.numericChange",
        "saleEstimate_valueChange.percentageChange",
        "saleEstimate_valueChange.saleDate",
        "saleEstimate_ingestedAt",
    }
    feature_cols = [c for c in clean.columns if c not in drop_cols]
    numeric = [c for c in feature_cols if pd.api.types.is_numeric_dtype(clean[c])]
    categorical = [c for c in feature_cols if c not in numeric]
    return feature_cols, numeric, categorical


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)

    print("[assisted] Loading and preparing dataset...", flush=True)
    raw = load_dataset(default_dataset_path(root))
    clean = clean_dataset(raw)
    split_data = calendar_split(clean)

    feature_cols, numeric, categorical = _assisted_features(clean)
    print(f"[assisted] Feature count={len(feature_cols)}", flush=True)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)),
                    ]
                ),
                categorical,
            ),
        ]
    )

    X_train = split_data.train[feature_cols]
    y_train = split_data.train[TARGET_COL].to_numpy()
    X_val = split_data.val[feature_cols]
    y_val = split_data.val[TARGET_COL].to_numpy()
    X_test = split_data.test[feature_cols]
    y_test = split_data.test[TARGET_COL].to_numpy()

    X_train_t = preprocess.fit_transform(X_train)
    X_val_t = preprocess.transform(X_val)
    X_test_t = preprocess.transform(X_test)
    print(f"[assisted] Transformed shapes train={X_train_t.shape} val={X_val_t.shape} test={X_test_t.shape}", flush=True)

    models = [
        (
            "AssistedHistGBR",
            HistGradientBoostingRegressor(
                max_depth=7,
                learning_rate=0.05,
                max_iter=160,
                min_samples_leaf=40,
                random_state=42,
            ),
        ),
        (
            "AssistedRandomForest",
            RandomForestRegressor(
                n_estimators=120,
                max_depth=20,
                min_samples_leaf=4,
                n_jobs=-1,
                random_state=42,
            ),
        ),
    ]

    results_rows: list[dict] = []
    pred_df = pd.DataFrame({"history_date": split_data.test["history_date"].values, "y_true": y_test})
    for name, model in models:
        print(f"[assisted] Training {name}...", flush=True)
        model.fit(X_train_t, y_train)
        val_pred = model.predict(X_val_t)
        test_pred = model.predict(X_test_t)
        val_mae, val_rmse = _score(y_val, val_pred)
        test_mae, test_rmse = _score(y_test, test_pred)
        pred_df[f"pred_{name}"] = test_pred
        results_rows.append(
            {
                "model": name,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "test_mae": test_mae,
                "test_rmse": test_rmse,
            }
        )
        print(f"[assisted] {name} test RMSE={test_rmse:,.2f}", flush=True)

    results = pd.DataFrame(results_rows).sort_values("test_rmse").reset_index(drop=True)
    out_results = data_dir / "assisted_track_results.csv"
    out_preds = data_dir / "assisted_track_predictions.csv"
    results.to_csv(out_results, index=False)
    pred_df.to_csv(out_preds, index=False)
    print(f"[assisted] Saved: {out_results}", flush=True)
    print(f"[assisted] Saved: {out_preds}", flush=True)
    print("[assisted] Best assisted model:")
    print(results.head(1).to_string(index=False))


if __name__ == "__main__":
    main()

