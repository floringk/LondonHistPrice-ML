from __future__ import annotations

import json
from pathlib import Path

from london_pipeline import (
    build_preprocessor,
    calendar_split,
    clean_dataset,
    default_dataset_path,
    error_by_year_bucket,
    segment_metrics,
    load_dataset,
    run_model_comparison,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)

    csv_path = default_dataset_path(root)
    print(f"[run] Loading dataset from: {csv_path}", flush=True)
    raw = load_dataset(csv_path)
    print(f"[run] Loaded rows={len(raw):,} cols={len(raw.columns)}", flush=True)

    print("[run] Cleaning dataset...", flush=True)
    clean = clean_dataset(raw)
    print(f"[run] Clean rows={len(clean):,} cols={len(clean.columns)}", flush=True)

    print("[run] Building calendar split...", flush=True)
    split_data = calendar_split(clean)
    print(
        (
            "[run] Split rows "
            f"train={len(split_data.train):,} "
            f"val={len(split_data.val):,} "
            f"test={len(split_data.test):,}"
        ),
        flush=True,
    )
    print(
        f"[run] Date range {split_data.t_min} -> {split_data.t_max}; "
        f"cuts {split_data.cut_train_end} | {split_data.cut_val_end}",
        flush=True,
    )

    print("[run] Building preprocessor...", flush=True)
    preprocess, feature_cols = build_preprocessor(clean)
    print(f"[run] Feature count={len(feature_cols)}", flush=True)

    print("[run] Running model comparison...", flush=True)
    results_df, feature_imp_df, test_pred_df = run_model_comparison(
        split_data=split_data,
        preprocessor=preprocess,
        feature_cols=feature_cols,
    )
    print("[run] Computing year-bucket diagnostics...", flush=True)
    year_rmse_df = error_by_year_bucket(test_pred_df)
    segment_df = segment_metrics(split_data, test_pred_df)

    results_path = data_dir / "model_results_summary.csv"
    imp_path = data_dir / "feature_importance.csv"
    pred_path = data_dir / "test_predictions.csv"
    year_path = data_dir / "test_rmse_by_year.csv"
    segment_path = data_dir / "segment_metrics.csv"
    registry_path = data_dir / "experiment_registry.json"

    results_df.to_csv(results_path, index=False)
    print(f"[run] Saved: {results_path}", flush=True)
    feature_imp_df.to_csv(imp_path, index=False)
    print(f"[run] Saved: {imp_path}", flush=True)
    test_pred_df.to_csv(pred_path, index=False)
    print(f"[run] Saved: {pred_path}", flush=True)
    year_rmse_df.to_csv(year_path, index=False)
    print(f"[run] Saved: {year_path}", flush=True)
    segment_df.to_csv(segment_path, index=False)
    print(f"[run] Saved: {segment_path}", flush=True)

    registry = {
        "run_name": "london_phase2_baseline",
        "target": "history_price",
        "date_axis": "history_date",
        "split_policy": "calendar_fair_70_10_20_by_time_span",
        "primary_metric": "test_rmse",
        "artifacts": {
            "model_results_summary": str(results_path),
            "feature_importance": str(imp_path),
            "test_predictions": str(pred_path),
            "test_rmse_by_year": str(year_path),
            "segment_metrics": str(segment_path),
        },
    }
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    print(f"[run] Saved: {registry_path}", flush=True)
    print("\nBest test model:")
    print(results_df.head(1).to_string(index=False))


if __name__ == "__main__":
    main()

