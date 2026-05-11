# London housing price prediction (ML Model)

Private research repo: temporally split regression models predicting **`history_price`** from London transaction history, with a leakage-safe **mainline** track and an optional **assisted** track that uses vendor **`saleEstimate_*`** features.

## Problem and target

- **Target:** `history_price`  
- **Time axis:** `history_date`  
- **Evaluation:** Calendar-fair split — train / validation / test use **70% / 10% / 20% of elapsed calendar time** between min and max dates (not 70% of rows).

Full methodology: see [`methodology.md`](methodology.md). **Deployment, dual-track policy, and teammate bundle:** see [`GOVERNANCE.md`](GOVERNANCE.md).

## Frozen metrics (committed)

After a full run, copy small summary files from `data/` into [`results/`](results/) with:

```powershell
python "src/main.py" --report-only --sync-results
```

This refreshes [`results/RUN_STAMP.txt`](results/RUN_STAMP.txt) and the CSV/JSON report mirrors. Large working outputs stay in [`data/`](data/) (gitignored).

## Setup

1. Python **3.11+** recommended (matches typical sklearn stacks).
2. Create a virtual environment and install dependencies:

```powershell
cd "C:\Projects\ML Model"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Download data — see [`dataset/README.md`](dataset/README.md). Expected path:

`dataset/kaggle_london_house_price_data.csv`

## How to run (PowerShell)

Run each step from the project root. Use `;` to chain commands (not `&&` on older PowerShell).

**Full evaluation bundle and frozen `results/` mirror:**

```powershell
python "src/run_london_pipeline.py"
python "src/assisted_track.py"
python "src/walk_forward_validation.py"
python "src/external_benchmark_estimates.py"
python "src/main.py" --report-only --sync-results
```

**One-shot (mainline + walk-forward + benchmark + assisted + PyTorch MLP + report, then sync to `results/`):**

```powershell
python "src/main.py" --all --sync-results
```

Flags for `src/main.py`: `--baseline`, `--walk-forward`, `--benchmark`, `--assisted`, `--torch-mlp`, `--report-only`, `--all`, `--sync-results`.

`--torch-mlp` requires PyTorch; for the CPU wheel on Windows:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Where the code lives

| Path | Role |
|------|------|
| [`src/london_pipeline.py`](src/london_pipeline.py) | Load/clean, calendar split, preprocessing, mainline model comparison, segment metrics |
| [`src/run_london_pipeline.py`](src/run_london_pipeline.py) | CLI: mainline run and writes to `data/` |
| [`src/assisted_track.py`](src/assisted_track.py) | Assisted track with `saleEstimate_*` predictors |
| [`src/walk_forward_validation.py`](src/walk_forward_validation.py) | Rolling time folds, RF + MLP stability per family |
| [`src/external_benchmark_estimates.py`](src/external_benchmark_estimates.py) | Benchmark RMSE vs `saleEstimate_*` columns |
| [`src/torch_mlp.py`](src/torch_mlp.py) | PyTorch MLP baseline (log-target, early stopping); appends `TorchMLP` row |
| [`src/regression_diagnostics.py`](src/regression_diagnostics.py) | Plots + price-bin confusion + classification report; loss-curve helper |
| [`src/hparam_search.py`](src/hparam_search.py) | Bounded hyperparameter grid |
| [`src/main.py`](src/main.py) | Orchestration + consolidated `run_report.md`, gates, decision CSVs; `--sync-results` copies summaries to `results/` |
| [`notebooks/01_london_phase1.ipynb`](notebooks/01_london_phase1.ipynb) | Original exploratory notebook |
| [`GOVERNANCE.md`](GOVERNANCE.md) | Dual-track deployment policy, segment blockers, private-repo migration |
| [`rollout/SEGMENT_BLOCKER_RESOLUTION.md`](rollout/SEGMENT_BLOCKER_RESOLUTION.md) | Pre-rollout checklist for segment gate failures |

Long-form write-up: [`task.md`](task.md).

**Course / team:** what to cite for literature vs slides, slide bullets, and an IMRaD report with filled tables — [`docs/TEAM_HANDOUT.md`](docs/TEAM_HANDOUT.md) and [`docs/REPORT_IMRAD.md`](docs/REPORT_IMRAD.md).

**Paper-style unified report (English, abstract + tables + captioned figures):** [`docs/RESULTS_AND_DISCUSSION.md`](docs/RESULTS_AND_DISCUSSION.md). Romanian feedback replies (long and short form): [`docs/Dedicatie_Radu.md`](docs/Dedicatie_Radu.md), [`docs/FEEDBACK_REPLIES_SCURT_RO.md`](docs/FEEDBACK_REPLIES_SCURT_RO.md).

## Accuracy metrics

The task is **regression on continuous GBP** prices, so classification accuracy doesn't apply directly to the target. Five complementary lenses are reported (frozen numbers from the latest snapshot in `results/`):

**Regression-friendly accuracy surrogates (best model: `HistGBR`, test set 2018-10 → 2024-09):**

| Metric | Value | Reading |
|---|---:|---|
| Test RMSE (GBP) | **393,489** | Root-mean-square error in pounds |
| Test MAE (GBP) | 216,083 | Mean absolute error in pounds |
| Test R² | **0.720** | ~72% of variance explained |
| Test MAPE | 33.8% | Mean relative error |
| Within-10% rate | **24.3%** | Share of rows with \|y−ŷ\|/\|y\| < 10% (on \|y\|>100 GBP) |
| RMSE uplift vs naive median | **57.6%** | Reduction in RMSE vs predicting train median everywhere |

**Classification-style accuracy on price bins (HistGBR, 5 quantile bins of `y_true`):**

| Metric | Value | Source |
|---|---:|---|
| **Price-bin accuracy** | **49.51%** | `results/price_bin_classification_summary.csv` |
| Macro F1 | 0.495 | same |
| Weighted F1 | 0.497 | same |

Per-bin precision/recall/F1/support: `results/price_bin_classification_report.csv`. Confusion matrix CSV + PNG: `results/price_bin_confusion.csv` / `.png`.

**Walk-forward stability (4 calendar folds, two model families):**

| Family | Mean RMSE | RMSE std | CV% |
|---|---:|---:|---:|
| RandomForest | 347,500 | 84,775 | 24.4% |
| MLP (sklearn medium) | 356,658 | 53,909 | **15.1%** |

MLP has **lower CV than RF** (more stable across folds); per-fold detail in `results/walk_forward_results.csv`.

**Where each metric comes from:**

- `data/model_results_summary.csv` (synced to `results/`) — MAE / RMSE / R² / MAPE / within-10% per model on val and test.
- `data/price_bin_classification_summary.csv` — accuracy + F1 on the derived 5-bin task.
- `data/walk_forward_results.csv` — per-fold RMSE per `model_family`.
- `data/segment_metrics.csv` — MAE / RMSE per `outcode_area`, `propertyType`, `price_band` (high-price tail and sparse types dominate RMSE; see `results/run_report.md` blockers).
- `data/test_rmse_by_year.csv` — RMSE per calendar year on test.
- Figures: `regression_pred_vs_actual.png`, `regression_residuals.png`, `price_bin_confusion.png`, `MLP_*_loss_curve.png`, `torch_mlp_loss_curve.png`.

Re-running after dependency upgrades may shift metrics slightly; freeze milestone numbers in `results/` via `python src/main.py --report-only --sync-results`.

## Repository hygiene

- **`data/`** — ignored (volatile pipeline output).  
- **`dataset/*.csv`** — ignored (large); use `dataset/README.md`.  
- **`.conda/`**, **`anaconda_projects/`** — ignored.  
- **`results/`** — committed summaries for this milestone.

Legacy Colab EDA scripts (`o_altă_copie_*.py`) can stay on your machine for reference; they are gitignored and not part of the pipeline.

## GitHub (private)

To move from a **public** repo to **private**, use GitHub **Settings → General → Danger zone → Change repository visibility** (keeps history), or delete and recreate a private repo (see [`GOVERNANCE.md`](GOVERNANCE.md) § 6).

After cloning, add data, run the pipeline, commit refreshed `results/` after `--sync-results`.

```powershell
git remote add origin https://github.com/<org>/<repo>.git
git branch -M main
git push -u origin main
```

Use HTTPS with a personal access token or SSH as per your org policy.
