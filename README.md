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

**One-shot (mainline + walk-forward + benchmark + assisted + report, then sync to `results/`):**

```powershell
python "src/main.py" --all --sync-results
```

Flags for `src/main.py`: `--baseline`, `--walk-forward`, `--benchmark`, `--assisted`, `--report-only`, `--all`, `--sync-results`.

## Where the code lives

| Path | Role |
|------|------|
| [`src/london_pipeline.py`](src/london_pipeline.py) | Load/clean, calendar split, preprocessing, mainline model comparison, segment metrics |
| [`src/run_london_pipeline.py`](src/run_london_pipeline.py) | CLI: mainline run and writes to `data/` |
| [`src/assisted_track.py`](src/assisted_track.py) | Assisted track with `saleEstimate_*` predictors |
| [`src/walk_forward_validation.py`](src/walk_forward_validation.py) | Rolling time folds, RF stability |
| [`src/external_benchmark_estimates.py`](src/external_benchmark_estimates.py) | Benchmark RMSE vs `saleEstimate_*` columns |
| [`src/hparam_search.py`](src/hparam_search.py) | Bounded hyperparameter grid |
| [`src/main.py`](src/main.py) | Orchestration + consolidated `run_report.md`, gates, decision CSVs; `--sync-results` copies summaries to `results/` |
| [`notebooks/01_london_phase1.ipynb`](notebooks/01_london_phase1.ipynb) | Original exploratory notebook |
| [`GOVERNANCE.md`](GOVERNANCE.md) | Dual-track deployment policy, segment blockers, private-repo migration |
| [`rollout/SEGMENT_BLOCKER_RESOLUTION.md`](rollout/SEGMENT_BLOCKER_RESOLUTION.md) | Pre-rollout checklist for segment gate failures |

Long-form write-up: [`task.md`](task.md).

## Accuracy metrics

- **MAE / RMSE** on validation and test (pounds).  
- **Primary headline:** test RMSE on the calendar holdout.  
- **Walk-forward:** RMSE per fold; mean and CV in `walk_forward_results.csv`.  
- **Segments:** `segment_metrics.csv` (high-price tail and sparse types may dominate RMSE).

Re-running after dependency upgrades may shift metrics slightly; freeze milestone numbers in `results/`.

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
