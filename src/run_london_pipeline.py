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
from regression_diagnostics import plot_loss_curve, write_regression_diagnostics


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
    results_df, feature_imp_df, test_pred_df, loss_curves = run_model_comparison(
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
    diag_paths = write_regression_diagnostics(data_dir, test_pred_df, results_df)
    for key, path in diag_paths.items():
        print(f"[run] diagnostics {key}: {path}", flush=True)
    loss_curve_paths: dict[str, tuple[Path, Path]] = {}
    for name, values in loss_curves.items():
        if not values:
            continue
        csv_path, png_path = plot_loss_curve(name, values, data_dir)
        loss_curve_paths[name] = (csv_path, png_path)
        print(f"[run] loss curve {name}: {png_path}", flush=True)
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
            "regression_pred_vs_actual": str(data_dir / "regression_pred_vs_actual.png"),
            "regression_residuals": str(data_dir / "regression_residuals.png"),
            "price_bin_confusion_csv": str(data_dir / "price_bin_confusion.csv"),
            "price_bin_confusion_png": str(data_dir / "price_bin_confusion.png"),
            "price_bin_edges": str(data_dir / "price_bin_edges.csv"),
            "price_bin_classification_report": str(
                data_dir / "price_bin_classification_report.csv"
            ),
            "price_bin_classification_summary": str(
                data_dir / "price_bin_classification_summary.csv"
            ),
            "mlp_loss_curves": {
                name: {"csv": str(csv_p), "png": str(png_p)}
                for name, (csv_p, png_p) in loss_curve_paths.items()
            },
        },
    }
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    print(f"[run] Saved: {registry_path}", flush=True)
    print("\nBest test model:")
    print(results_df.head(1).to_string(index=False))


if __name__ == "__main__":
    main()

