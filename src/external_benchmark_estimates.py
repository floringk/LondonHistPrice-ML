from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from london_pipeline import TARGET_COL, calendar_split, clean_dataset, default_dataset_path, load_dataset


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    return float(mean_absolute_error(y_true, y_pred)), float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)

    print("[benchmark] Loading and preparing dataset...", flush=True)
    raw = load_dataset(default_dataset_path(root))
    clean = clean_dataset(raw)
    split_data = calendar_split(clean)

    candidate_cols = [
        "saleEstimate_currentPrice",
        "saleEstimate_lowerPrice",
        "saleEstimate_upperPrice",
    ]
    available = [c for c in candidate_cols if c in split_data.test.columns]
    if not available:
        raise RuntimeError("No saleEstimate_* columns available for external benchmark.")

    y_test = pd.to_numeric(split_data.test[TARGET_COL], errors="coerce")
    rows = []
    for col in available:
        pred = pd.to_numeric(split_data.test[col], errors="coerce")
        mask = y_test.notna() & pred.notna()
        if mask.sum() < 100:
            continue
        mae, rmse = _score(y_test[mask].to_numpy(), pred[mask].to_numpy())
        rows.append(
            {
                "benchmark": col,
                "test_rows_used": int(mask.sum()),
                "test_mae": mae,
                "test_rmse": rmse,
            }
        )
        print(f"[benchmark] {col}: rows={int(mask.sum()):,} RMSE={rmse:,.2f}", flush=True)

    if not rows:
        raise RuntimeError("No valid benchmark rows after non-null filtering.")

    out = pd.DataFrame(rows).sort_values("test_rmse").reset_index(drop=True)
    output_path = data_dir / "external_estimate_benchmark.csv"
    out.to_csv(output_path, index=False)
    print(f"[benchmark] Saved: {output_path}", flush=True)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

