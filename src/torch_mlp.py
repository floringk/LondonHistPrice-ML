"""PyTorch MLP baseline on the same calendar split as `london_pipeline`.

Why a separate script:
- The NN course audience expects an explicit framework loop with early stopping and
  a saved loss curve.
- Keeps the heavy `torch` import optional: the main `run_model_comparison` flow does
  not require PyTorch.

Usage:
    python src/torch_mlp.py            # writes data/torch_mlp_*, appends TorchMLP row
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.metrics import mean_absolute_error, mean_squared_error

from london_pipeline import (
    DATE_COL,
    TARGET_COL,
    _extra_regression_metrics,
    build_preprocessor,
    calendar_split,
    clean_dataset,
    default_dataset_path,
    load_dataset,
)
from regression_diagnostics import plot_loss_curve

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required for src/torch_mlp.py. Install with: pip install torch"
    ) from exc


def _densify(X) -> np.ndarray:
    if issparse(X):
        return X.toarray().astype(np.float32, copy=False)
    return np.asarray(X, dtype=np.float32)


class MLPNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: Sequence[int] = (256, 128),
        dropout: float = 0.2,
        output_bias_init: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        final = nn.Linear(prev, 1)
        # Initialise final-layer bias to the training log-mean and zero weights so
        # epoch 0 predicts the mean target; prevents `expm1` blow-up on random init.
        with torch.no_grad():
            final.weight.zero_()
            final.bias.fill_(output_bias_init)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x).squeeze(-1)


def _set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _predict_dense(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch: int = 4096,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> np.ndarray:
    model.eval()
    out = np.empty(X.shape[0], dtype=np.float32)
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            sl = slice(i, min(i + batch, X.shape[0]))
            tx = torch.from_numpy(X[sl]).to(device)
            out[sl] = model(tx).detach().cpu().numpy()
    if clip_min is not None or clip_max is not None:
        out = np.clip(out, clip_min, clip_max)
    return out


def _append_row(results_path: Path, row: dict) -> None:
    """Append (or replace) a row in model_results_summary.csv keeping schema + sort order."""
    if not results_path.is_file():
        df = pd.DataFrame([row])
    else:
        df = pd.read_csv(results_path)
        df = df[df["model"] != row["model"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    if "test_rmse" in df.columns:
        df = df.sort_values("test_rmse").reset_index(drop=True)
    df.to_csv(results_path, index=False)


def _append_pred(pred_path: Path, dates: np.ndarray, y_true: np.ndarray, pred_test: np.ndarray, name: str) -> None:
    if pred_path.is_file():
        df = pd.read_csv(pred_path)
        if len(df) != len(y_true):
            # Stale file from a different split — overwrite to avoid silent mismatch.
            df = pd.DataFrame({DATE_COL: dates, "y_true": y_true})
        df[f"pred_{name}"] = pred_test
    else:
        df = pd.DataFrame({DATE_COL: dates, "y_true": y_true, f"pred_{name}": pred_test})
    df.to_csv(pred_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch MLP baseline on London prices")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden", type=str, default="256,128")
    args = parser.parse_args()

    _set_seed(42)
    hidden = tuple(int(h) for h in args.hidden.split(",") if h.strip())

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)

    print("[torch-mlp] Loading data...", flush=True)
    raw = load_dataset(default_dataset_path(root))
    clean = clean_dataset(raw)
    split = calendar_split(clean)
    preprocess, feature_cols = build_preprocessor(clean)

    X_train = _densify(preprocess.fit_transform(split.train[feature_cols]))
    X_val = _densify(preprocess.transform(split.val[feature_cols]))
    X_test = _densify(preprocess.transform(split.test[feature_cols]))
    y_train = split.train[TARGET_COL].to_numpy(dtype=np.float64)
    y_val = split.val[TARGET_COL].to_numpy(dtype=np.float64)
    y_test = split.test[TARGET_COL].to_numpy(dtype=np.float64)

    print(
        f"[torch-mlp] Shapes train={X_train.shape} val={X_val.shape} test={X_test.shape}",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[torch-mlp] Device={device} hidden={hidden}", flush=True)

    y_train_log = np.log1p(y_train).astype(np.float32)
    # Clip predictions to training log range to prevent expm1 blow-up from extreme
    # initial random weights or extrapolation outside the training distribution.
    log_min = float(np.min(y_train_log))
    log_max = float(np.max(y_train_log))
    log_clip_pad = 0.5  # Allow some slack beyond observed train range.
    log_clip_min = log_min - log_clip_pad
    log_clip_max = log_max + log_clip_pad
    print(
        f"[torch-mlp] Log target clip range=[{log_clip_min:.3f}, {log_clip_max:.3f}] "
        f"(GBP=[{float(np.expm1(log_clip_min)):,.0f}, {float(np.expm1(log_clip_max)):,.0f}])",
        flush=True,
    )
    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train_log)
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False
    )

    model = MLPNet(
        in_dim=X_train.shape[1],
        hidden=hidden,
        dropout=args.dropout,
        output_bias_init=float(np.mean(y_train_log)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()

    history_rows: list[dict] = []
    best_val_rmse = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    no_improve = 0
    train_loss_curve: list[float] = []

    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        train_loss_curve.append(train_loss)

        # Validation in log space converted back to GBP for the early-stop signal.
        pred_val_log = _predict_dense(
            model, X_val, device, clip_min=log_clip_min, clip_max=log_clip_max
        )
        pred_val_gbp = np.expm1(pred_val_log)
        val_rmse = float(np.sqrt(mean_squared_error(y_val, pred_val_gbp)))
        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_rmse": val_rmse})
        print(
            f"[torch-mlp] epoch={epoch:03d} train_loss={train_loss:.4f} val_rmse={val_rmse:,.2f}",
            flush=True,
        )

        if val_rmse < best_val_rmse - 1e-6:
            best_val_rmse = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(
                    f"[torch-mlp] Early stop at epoch {epoch} (no improvement in {args.patience}).",
                    flush=True,
                )
                break

    elapsed = time.perf_counter() - t0
    print(f"[torch-mlp] Training done in {elapsed:.1f}s; best val RMSE={best_val_rmse:,.2f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    pred_val = np.expm1(
        _predict_dense(model, X_val, device, clip_min=log_clip_min, clip_max=log_clip_max)
    )
    pred_test = np.expm1(
        _predict_dense(model, X_test, device, clip_min=log_clip_min, clip_max=log_clip_max)
    )

    rmse_test = float(np.sqrt(mean_squared_error(y_test, pred_test)))
    ex_val = _extra_regression_metrics(y_val, pred_val)
    ex_test = _extra_regression_metrics(y_test, pred_test)

    # Need naive RMSE on test to fill the uplift column consistently with the others.
    naive_test_pred = np.full_like(y_test, float(np.median(y_train)), dtype=float)
    naive_rmse_test = float(np.sqrt(mean_squared_error(y_test, naive_test_pred)))
    uplift = 100.0 * (1.0 - rmse_test / naive_rmse_test) if naive_rmse_test > 0 else 0.0

    row = {
        "model": "TorchMLP",
        "val_mae": float(mean_absolute_error(y_val, pred_val)),
        "val_rmse": float(np.sqrt(mean_squared_error(y_val, pred_val))),
        "val_r2": ex_val["r2"],
        "val_mape": ex_val["mape"],
        "val_within_10pct_rate": ex_val["within_10pct_rate"],
        "test_mae": float(mean_absolute_error(y_test, pred_test)),
        "test_rmse": rmse_test,
        "test_r2": ex_test["r2"],
        "test_mape": ex_test["mape"],
        "test_within_10pct_rate": ex_test["within_10pct_rate"],
        "test_rmse_improvement_vs_naive_pct": uplift,
    }

    history_df = pd.DataFrame(history_rows)
    history_path = data_dir / "torch_mlp_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"[torch-mlp] Saved: {history_path}", flush=True)

    csv_path, png_path = plot_loss_curve("torch_mlp", train_loss_curve, data_dir)
    print(f"[torch-mlp] Saved: {png_path}", flush=True)

    results_path = data_dir / "model_results_summary.csv"
    _append_row(results_path, row)
    print(f"[torch-mlp] Appended TorchMLP row to: {results_path}", flush=True)

    pred_path = data_dir / "test_predictions.csv"
    _append_pred(pred_path, split.test[DATE_COL].to_numpy(), y_test, pred_test, "TorchMLP")
    print(f"[torch-mlp] Appended pred_TorchMLP column to: {pred_path}", flush=True)

    print("\nTorchMLP test metrics:")
    for k, v in row.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
