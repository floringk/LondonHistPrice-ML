from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from london_pipeline import (
    DATE_COL,
    TARGET_COL,
    _make_log_target_mlp,
    build_preprocessor,
    clean_dataset,
    default_dataset_path,
    load_dataset,
)


def build_time_folds(df: pd.DataFrame, n_folds: int = 4) -> list[tuple[pd.Series, pd.Series, str]]:
    t = pd.to_datetime(df[DATE_COL], errors="coerce")
    t_min, t_max = t.min(), t.max()
    span = (t_max - t_min).total_seconds()
    fold_width = span / (n_folds + 1)
    folds = []

    for i in range(1, n_folds + 1):
        val_start = t_min + pd.Timedelta(seconds=i * fold_width)
        val_end = t_min + pd.Timedelta(seconds=(i + 1) * fold_width)
        train_m = t < val_start
        val_m = (t >= val_start) & (t < val_end)
        label = f"fold_{i}_{val_start.date()}_{val_end.date()}"
        folds.append((train_m, val_m, label))
    return folds


def _rf() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=350,
        max_depth=24,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )


def _densify(X) -> np.ndarray:
    if issparse(X):
        return X.toarray().astype(np.float32, copy=False)
    return np.asarray(X, dtype=np.float32)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)

    csv_path = default_dataset_path(root)
    print(f"[walk-forward] Loading dataset from: {csv_path}", flush=True)
    raw = load_dataset(csv_path)
    print(f"[walk-forward] Loaded rows={len(raw):,}", flush=True)
    clean = clean_dataset(raw)
    print(f"[walk-forward] Clean rows={len(clean):,}", flush=True)
    preprocess, feature_cols = build_preprocessor(clean)
    print(f"[walk-forward] Feature count={len(feature_cols)}", flush=True)

    families: tuple[str, ...] = ("RandomForest", "MLP")

    rows: list[dict] = []
    folds = build_time_folds(clean, n_folds=4)
    print(
        f"[walk-forward] Total folds={len(folds)} families={list(families)}",
        flush=True,
    )
    for i, (train_m, val_m, fold_name) in enumerate(folds, start=1):
        train_df = clean.loc[train_m].copy()
        val_df = clean.loc[val_m].copy()
        print(
            f"[walk-forward] Fold {i}/{len(folds)} {fold_name}: "
            f"train={len(train_df):,} val={len(val_df):,}",
            flush=True,
        )
        if len(train_df) < 1000 or len(val_df) < 1000:
            print(f"[walk-forward] Skipping {fold_name} (too small)", flush=True)
            continue

        X_train = train_df[feature_cols]
        y_train = train_df[TARGET_COL].to_numpy()
        X_val = val_df[feature_cols]
        y_val = val_df[TARGET_COL].to_numpy()

        X_train_t = preprocess.fit_transform(X_train)
        X_val_t = preprocess.transform(X_val)

        for family in families:
            try:
                if family == "RandomForest":
                    model = _rf()
                    model.fit(X_train_t, y_train)
                    pred = model.predict(X_val_t)
                elif family == "MLP":
                    model = _make_log_target_mlp((128, 64))
                    model.fit(_densify(X_train_t), y_train)
                    pred = model.predict(_densify(X_val_t))
                else:
                    continue
                rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
                print(
                    f"[walk-forward] {fold_name} {family} RMSE={rmse:,.2f}",
                    flush=True,
                )
                rows.append(
                    {
                        "fold": fold_name,
                        "model_family": family,
                        "train_rows": int(len(train_df)),
                        "val_rows": int(len(val_df)),
                        "rmse": rmse,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[walk-forward] {fold_name} {family} skipped: {exc}", flush=True)

    wf_df = pd.DataFrame(rows)
    if wf_df.empty:
        raise RuntimeError("No valid walk-forward folds produced.")
    # Per-family aggregates as additional columns so existing readers still see the columns
    # mean/std/CV; values now repeat the per-family aggregate for every row of that family.
    aggregates = (
        wf_df.groupby("model_family")["rmse"]
        .agg(rmse_mean="mean", rmse_std=lambda s: float(s.std(ddof=0)))
        .reset_index()
    )
    aggregates["cv_pct"] = 100.0 * aggregates["rmse_std"] / aggregates["rmse_mean"]
    wf_df = wf_df.merge(aggregates, on="model_family", how="left")

    output = data_dir / "walk_forward_results.csv"
    wf_df.to_csv(output, index=False)
    print(f"[walk-forward] Saved: {output}", flush=True)
    print(wf_df.to_string(index=False))


if __name__ == "__main__":
    main()
