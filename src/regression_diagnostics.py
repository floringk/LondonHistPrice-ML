from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import KBinsDiscretizer


def _best_model_name(results_df: pd.DataFrame | None) -> str:
    if results_df is None or results_df.empty:
        return "HistGBR"
    return str(results_df.sort_values("test_rmse", ascending=True).iloc[0]["model"])


def write_regression_diagnostics(
    data_dir: Path,
    test_predictions: pd.DataFrame,
    results_df: pd.DataFrame | None,
    max_scatter_points: int = 12000,
    n_bins: int = 5,
) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    best = _best_model_name(results_df)
    pred_col = f"pred_{best}"
    if pred_col not in test_predictions.columns:
        raise KeyError(f"Missing column {pred_col} in test_predictions")

    y = test_predictions["y_true"].to_numpy(dtype=float)
    yhat = test_predictions[pred_col].to_numpy(dtype=float)
    resid = y - yhat

    rng = np.random.default_rng(42)
    n = len(y)
    if n > max_scatter_points:
        idx = rng.choice(n, size=max_scatter_points, replace=False)
        ys, yhs = y[idx], yhat[idx]
    else:
        ys, yhs = y, yhat

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(ys, yhs, s=4, alpha=0.35, color="#1f77b4")
    lim_lo = float(np.nanmin([ys.min(), yhs.min()]))
    lim_hi = float(np.nanmax([ys.max(), yhs.max()]))
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="gray", linestyle="--", linewidth=1, label="y = ŷ")
    ax.set_xlabel("Actual price (GBP)")
    ax.set_ylabel("Predicted price (GBP)")
    ax.set_title(f"Test: actual vs predicted ({best})")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    p1 = data_dir / "regression_pred_vs_actual.png"
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    out["pred_vs_actual"] = p1

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].hist(resid, bins=60, color="#2ca02c", alpha=0.85, edgecolor="white")
    axes[0].set_title("Residuals (y − ŷ)")
    axes[0].set_xlabel("GBP")
    axes[0].grid(True, alpha=0.25)
    axes[1].scatter(yhat, resid, s=3, alpha=0.2, color="#d62728")
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Predicted (GBP)")
    axes[1].set_ylabel("Residual (GBP)")
    axes[1].set_title(f"Residuals vs fitted ({best})")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    p2 = data_dir / "regression_residuals.png"
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    out["residuals"] = p2

    try:
        kbd = KBinsDiscretizer(
            n_bins=n_bins,
            encode="ordinal",
            strategy="quantile",
            quantile_method="averaged_inverted_cdf",
        )
    except TypeError:
        kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    yt = kbd.fit_transform(y.reshape(-1, 1)).ravel().astype(int)
    yp = kbd.transform(yhat.reshape(-1, 1)).ravel().astype(int)
    labels = list(range(n_bins))
    cm = confusion_matrix(yt, yp, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_bin_{i}" for i in labels],
        columns=[f"pred_bin_{j}" for j in labels],
    )
    p3 = data_dir / "price_bin_confusion.csv"
    cm_df.to_csv(p3)
    edges = kbd.bin_edges_[0]
    meta = pd.DataFrame({"bin_index": labels, "edge_left": edges[:-1], "edge_right": edges[1:]})
    meta_path = data_dir / "price_bin_edges.csv"
    meta.to_csv(meta_path, index=False)
    out["confusion_csv"] = p3
    out["bin_edges"] = meta_path

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xlabel("Predicted price bin")
    ax.set_ylabel("True price bin")
    ax.set_title(f"Confusion on price bins ({best})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    p4 = data_dir / "price_bin_confusion.png"
    fig.savefig(p4, dpi=140)
    plt.close(fig)
    out["confusion_png"] = p4

    # Classification view of the price-bin task: precision/recall/F1 per bin + overall accuracy.
    report_dict = classification_report(
        yt, yp, labels=labels, output_dict=True, zero_division=0
    )
    per_bin_rows: list[dict] = []
    for lab in labels:
        key = str(lab)
        info = report_dict.get(key, {})
        per_bin_rows.append(
            {
                "bin": lab,
                "precision": float(info.get("precision", 0.0)),
                "recall": float(info.get("recall", 0.0)),
                "f1": float(info.get("f1-score", 0.0)),
                "support": int(info.get("support", 0)),
            }
        )
    per_bin_df = pd.DataFrame(per_bin_rows)
    p5 = data_dir / "price_bin_classification_report.csv"
    per_bin_df.to_csv(p5, index=False)
    out["classification_report"] = p5

    summary_df = pd.DataFrame(
        [
            {
                "model": best,
                "n_bins": n_bins,
                "accuracy": float(accuracy_score(yt, yp)),
                "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(yt, yp, average="weighted", zero_division=0)),
                "support": int(len(yt)),
            }
        ]
    )
    p6 = data_dir / "price_bin_classification_summary.csv"
    summary_df.to_csv(p6, index=False)
    out["classification_summary"] = p6

    return out


def plot_loss_curve(name: str, loss_values: list[float], out_dir: Path) -> tuple[Path, Path]:
    """Save MLP/torch training loss curve as CSV + PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}_loss_curve.csv"
    png_path = out_dir / f"{name}_loss_curve.png"
    df = pd.DataFrame({"iteration": np.arange(1, len(loss_values) + 1), "loss": loss_values})
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(df["iteration"], df["loss"], color="#1f77b4", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training loss")
    ax.set_title(f"{name} training loss")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    return csv_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Regression diagnostics from test_predictions.csv")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Directory with test_predictions.csv and model_results_summary.csv",
    )
    args = parser.parse_args()
    data_dir: Path = args.data_dir
    pred_path = data_dir / "test_predictions.csv"
    res_path = data_dir / "model_results_summary.csv"
    if not pred_path.is_file():
        raise SystemExit(f"Missing {pred_path}")
    test_predictions = pd.read_csv(pred_path)
    results_df = pd.read_csv(res_path) if res_path.is_file() else None
    paths = write_regression_diagnostics(data_dir, test_predictions, results_df)
    for k, v in paths.items():
        print(f"[diagnostics] wrote {k}: {v}", flush=True)


if __name__ == "__main__":
    main()
