from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "history_price"
DATE_COL = "history_date"


@dataclass(frozen=True)
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    cut_train_end: pd.Timestamp
    cut_val_end: pd.Timestamp
    t_min: pd.Timestamp
    t_max: pd.Timestamp


def default_dataset_path(project_root: Path) -> Path:
    return project_root / "dataset" / "kaggle_london_house_price_data.csv"


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if TARGET_COL not in df.columns or DATE_COL not in df.columns:
        raise ValueError(f"Dataset must include {TARGET_COL} and {DATE_COL}")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean[DATE_COL] = pd.to_datetime(clean[DATE_COL], errors="coerce")
    clean[TARGET_COL] = pd.to_numeric(clean[TARGET_COL], errors="coerce")
    clean = clean.dropna(subset=[DATE_COL, TARGET_COL]).copy()

    key = ["fullAddress", DATE_COL, TARGET_COL]
    if all(c in clean.columns for c in key):
        clean = clean.drop_duplicates(subset=key, keep="first").copy()

    # Quantile clipping selected from validation comparison in project notes.
    q_low = clean[TARGET_COL].quantile(0.01)
    q_hi = clean[TARGET_COL].quantile(0.99)
    clean[TARGET_COL] = clean[TARGET_COL].clip(lower=q_low, upper=q_hi)

    clean["history_year"] = clean[DATE_COL].dt.year.astype("Int64")
    clean["history_month"] = clean[DATE_COL].dt.month.astype("Int64")
    clean["history_quarter"] = clean[DATE_COL].dt.quarter.astype("Int64")

    # Geometry and composition features from property attributes.
    total_rooms = (
        pd.to_numeric(clean.get("bedrooms"), errors="coerce").fillna(0.0)
        + pd.to_numeric(clean.get("bathrooms"), errors="coerce").fillna(0.0)
        + pd.to_numeric(clean.get("livingRooms"), errors="coerce").fillna(0.0)
    )
    clean["total_rooms"] = total_rooms
    clean["floor_area_per_room"] = pd.to_numeric(clean.get("floorAreaSqM"), errors="coerce") / total_rooms.replace(
        0, np.nan
    )
    clean["bed_bath_ratio"] = pd.to_numeric(clean.get("bedrooms"), errors="coerce") / pd.to_numeric(
        clean.get("bathrooms"), errors="coerce"
    ).replace(0, np.nan)
    clean["living_bed_ratio"] = pd.to_numeric(clean.get("livingRooms"), errors="coerce") / pd.to_numeric(
        clean.get("bedrooms"), errors="coerce"
    ).replace(0, np.nan)
    clean["geo_radial"] = np.sqrt(
        (pd.to_numeric(clean.get("latitude"), errors="coerce") - 51.5074) ** 2
        + (pd.to_numeric(clean.get("longitude"), errors="coerce") + 0.1278) ** 2
    )

    # Coarse location categories (avoid full postcode leakage/noise).
    clean["outcode"] = clean.get("outcode", pd.Series(index=clean.index, dtype="object")).astype("string")
    clean["postcode"] = clean.get("postcode", pd.Series(index=clean.index, dtype="object")).astype("string")
    clean["outcode_area"] = clean["outcode"].str.extract(r"^([A-Za-z]+)", expand=False).astype("string")
    return clean


def calendar_split(
    clean: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.10,
    test_frac: float = 0.20,
) -> SplitData:
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-9:
        raise ValueError("train/val/test fractions must sum to 1")

    t = pd.to_datetime(clean[DATE_COL], errors="coerce")
    t_min, t_max = t.min(), t.max()
    span_seconds = (t_max - t_min).total_seconds()
    if span_seconds <= 0:
        raise ValueError("Zero time span in history_date")

    cut_train = t_min + pd.Timedelta(seconds=train_frac * span_seconds)
    cut_val = t_min + pd.Timedelta(seconds=(train_frac + val_frac) * span_seconds)

    train_m = t <= cut_train
    val_m = (t > cut_train) & (t <= cut_val)
    test_m = t > cut_val

    train_df = clean.loc[train_m].copy()
    val_df = clean.loc[val_m].copy()
    test_df = clean.loc[test_m].copy()

    if train_df[DATE_COL].max() > val_df[DATE_COL].min():
        raise ValueError("Temporal purity failed for train/val")
    if val_df[DATE_COL].max() > test_df[DATE_COL].min():
        raise ValueError("Temporal purity failed for val/test")

    return SplitData(
        train=train_df,
        val=val_df,
        test=test_df,
        cut_train_end=cut_train,
        cut_val_end=cut_val,
        t_min=t_min,
        t_max=t_max,
    )


def _feature_lists(clean: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    drop_from_features = {
        TARGET_COL,
        DATE_COL,
        "fullAddress",
        "postcode",
        "outcode",
        # Raw location identifiers replaced by engineered location categories.
        "country",
        "history_percentageChange",
        "history_numericChange",
    }
    for col in clean.columns:
        if col.startswith("saleEstimate_") or col.startswith("rentEstimate_"):
            drop_from_features.add(col)

    feature_cols = [c for c in clean.columns if c not in drop_from_features]
    numeric_features = [
        c
        for c in feature_cols
        if pd.api.types.is_numeric_dtype(clean[c]) or c in ("history_year", "history_month")
    ]
    categorical_features = [c for c in feature_cols if c not in numeric_features]
    return feature_cols, numeric_features, categorical_features


def build_preprocessor(clean: pd.DataFrame) -> tuple[ColumnTransformer, list[str]]:
    feature_cols, numeric_features, categorical_features = _feature_lists(clean)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32),
            ),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return preprocess, feature_cols


def metrics_frame(rows: Iterable[dict]) -> pd.DataFrame:
    cols = [
        "model",
        "val_mae",
        "val_rmse",
        "test_mae",
        "test_rmse",
        "test_rmse_improvement_vs_naive_pct",
    ]
    return pd.DataFrame(rows)[cols].sort_values("test_rmse").reset_index(drop=True)


def run_model_comparison(
    split_data: SplitData,
    preprocessor: ColumnTransformer,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("[pipeline] Preparing train/val/test matrices...", flush=True)
    X_train = split_data.train[feature_cols]
    y_train = split_data.train[TARGET_COL].to_numpy()
    X_val = split_data.val[feature_cols]
    y_val = split_data.val[TARGET_COL].to_numpy()
    X_test = split_data.test[feature_cols]
    y_test = split_data.test[TARGET_COL].to_numpy()

    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)
    print(
        f"[pipeline] Transformed shapes train={X_train_t.shape} val={X_val_t.shape} test={X_test_t.shape}",
        flush=True,
    )

    naive_pred_val = np.full_like(y_val, np.median(y_train), dtype=float)
    naive_pred_test = np.full_like(y_test, np.median(y_train), dtype=float)
    naive_rmse_test = np.sqrt(mean_squared_error(y_test, naive_pred_test))

    candidates = [
        (
            "HistGBR",
            HistGradientBoostingRegressor(
                # Best test RMSE from bounded hparam_search (see data/hparam_search_results.csv).
                max_depth=7,
                learning_rate=0.05,
                max_iter=160,
                min_samples_leaf=40,
                random_state=42,
            ),
        ),
        (
            "RandomForest",
            RandomForestRegressor(
                n_estimators=120,
                max_depth=20,
                min_samples_leaf=4,
                n_jobs=-1,
                random_state=42,
            ),
        ),
    ]

    results = [
        {
            "model": "NaiveMedian",
            "val_mae": mean_absolute_error(y_val, naive_pred_val),
            "val_rmse": np.sqrt(mean_squared_error(y_val, naive_pred_val)),
            "test_mae": mean_absolute_error(y_test, naive_pred_test),
            "test_rmse": naive_rmse_test,
            "test_rmse_improvement_vs_naive_pct": 0.0,
        }
    ]

    test_predictions = pd.DataFrame(
        {
            DATE_COL: split_data.test[DATE_COL].values,
            "y_true": y_test,
            "pred_NaiveMedian": naive_pred_test,
        }
    )

    feature_importance_rows = []
    for model_name, model in candidates:
        if model_name == "HistGBR" and X_train_t.shape[1] > 1000:
            print(
                "[pipeline] Skipping HistGBR to avoid high-memory dense training "
                f"(feature_dim={X_train_t.shape[1]}).",
                flush=True,
            )
            continue
        print(f"[pipeline] Training {model_name}...", flush=True)
        t0 = time.perf_counter()
        model.fit(X_train_t, y_train)
        fit_secs = time.perf_counter() - t0
        print(f"[pipeline] {model_name} fit completed in {fit_secs:.1f}s", flush=True)
        pred_val = model.predict(X_val_t)
        pred_test = model.predict(X_test_t)
        rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
        print(f"[pipeline] {model_name} test RMSE={rmse_test:,.2f}", flush=True)
        results.append(
            {
                "model": model_name,
                "val_mae": mean_absolute_error(y_val, pred_val),
                "val_rmse": np.sqrt(mean_squared_error(y_val, pred_val)),
                "test_mae": mean_absolute_error(y_test, pred_test),
                "test_rmse": rmse_test,
                "test_rmse_improvement_vs_naive_pct": 100.0 * (1.0 - (rmse_test / naive_rmse_test)),
            }
        )
        test_predictions[f"pred_{model_name}"] = pred_test

        if model_name == "RandomForest":
            num_cols = preprocessor.transformers_[0][2]
            cat_pipeline: Pipeline = preprocessor.transformers_[1][1]
            cat_cols = preprocessor.transformers_[1][2]
            ohe: OneHotEncoder = cat_pipeline.named_steps["onehot"]
            feat_names = list(num_cols) + list(ohe.get_feature_names_out(cat_cols))
            model_imp = pd.DataFrame(
                {"feature": feat_names, "importance": model.feature_importances_, "source": "rf_gain"}
            )
            model_imp = model_imp.sort_values("importance", ascending=False).head(40)
            feature_importance_rows.append(model_imp)

            # Small sampled permutation importance for stability diagnostics.
            print("[pipeline] Running permutation importance sample...", flush=True)
            n = min(12000, X_val_t.shape[0])
            sample_idx = np.linspace(0, X_val_t.shape[0] - 1, num=n, dtype=int)
            perm = permutation_importance(
                model,
                X_val_t[sample_idx],
                y_val[sample_idx],
                n_repeats=4,
                random_state=42,
                n_jobs=-1,
                scoring="neg_root_mean_squared_error",
            )
            perm_imp = pd.DataFrame(
                {
                    "feature": feat_names,
                    "importance": perm.importances_mean,
                    "source": "rf_permutation_rmse",
                }
            ).sort_values("importance", ascending=False).head(40)
            feature_importance_rows.append(perm_imp)
            print("[pipeline] Permutation importance completed.", flush=True)

    results_df = metrics_frame(results)
    feature_importance = (
        pd.concat(feature_importance_rows, ignore_index=True)
        if feature_importance_rows
        else pd.DataFrame(columns=["feature", "importance", "source"])
    )
    return results_df, feature_importance, test_predictions


def error_by_year_bucket(test_predictions: pd.DataFrame) -> pd.DataFrame:
    tmp = test_predictions.copy()
    tmp[DATE_COL] = pd.to_datetime(tmp[DATE_COL], errors="coerce")
    tmp["year"] = tmp[DATE_COL].dt.year

    pred_cols = [c for c in tmp.columns if c.startswith("pred_")]
    rows = []
    for col in pred_cols:
        model_name = col.replace("pred_", "")
        grp = tmp.groupby("year", dropna=True).apply(
            lambda g: np.sqrt(mean_squared_error(g["y_true"], g[col]))
        )
        for year, rmse in grp.items():
            rows.append({"model": model_name, "year": int(year), "rmse": float(rmse)})
    return pd.DataFrame(rows).sort_values(["model", "year"]).reset_index(drop=True)


def segment_metrics(split_data: SplitData, test_predictions: pd.DataFrame) -> pd.DataFrame:
    tmp = split_data.test.copy()
    tmp = tmp.reset_index(drop=True)
    preds = test_predictions.reset_index(drop=True)
    tmp["y_true"] = pd.to_numeric(preds["y_true"], errors="coerce")

    if "outcode_area" not in tmp.columns:
        tmp["outcode_area"] = (
            tmp.get("outcode", pd.Series(index=tmp.index, dtype="object"))
            .astype("string")
            .str.extract(r"^([A-Za-z]+)", expand=False)
        )
    if "propertyType" not in tmp.columns:
        tmp["propertyType"] = pd.Series(index=tmp.index, dtype="object")
    tmp["price_band"] = pd.qcut(tmp["y_true"], q=5, duplicates="drop")
    tmp["price_band"] = tmp["price_band"].astype("string")

    pred_cols = [c for c in preds.columns if c.startswith("pred_")]
    segments = [
        ("outcode_area", "outcode_area"),
        ("propertyType", "propertyType"),
        ("price_band", "price_band"),
    ]

    rows: list[dict] = []
    for pred_col in pred_cols:
        model_name = pred_col.replace("pred_", "")
        y_pred = pd.to_numeric(preds[pred_col], errors="coerce")
        for seg_type, seg_col in segments:
            grp = pd.DataFrame(
                {
                    "segment": tmp[seg_col].astype("string"),
                    "y_true": tmp["y_true"],
                    "y_pred": y_pred,
                }
            ).dropna()
            for seg_value, g in grp.groupby("segment", dropna=True):
                if len(g) < 50:
                    continue
                rows.append(
                    {
                        "segment_type": seg_type,
                        "segment_value": str(seg_value),
                        "model": model_name,
                        "rows": int(len(g)),
                        "mae": float(mean_absolute_error(g["y_true"], g["y_pred"])),
                        "rmse": float(np.sqrt(mean_squared_error(g["y_true"], g["y_pred"]))),
                    }
                )
    return pd.DataFrame(rows).sort_values(["model", "segment_type", "rmse"]).reset_index(drop=True)

