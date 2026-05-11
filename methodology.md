# Methodology — London historical house price prediction

This document is the **canonical methodology reference** for the repository. For setup and commands, see [README.md](README.md). For a narrative project log and extended tables, see [task.md](task.md).

**Frozen metric snapshots** committed for reproducibility live under [`results/`](results/) (especially [`results/run_report.md`](results/run_report.md) and CSV summaries). Re-running pipelines writes to ignored `data/` and may produce slightly different numbers if dependencies change.

---

## 1. Scope and research questions

### 1.1 Prediction problem

- **Target:** `history_price` (historical transaction price in the dataset’s currency units).
- **Temporal index:** `history_date` — used only to order observations and define train/validation/test boundaries.

The modelling task is **supervised regression**: learn a mapping from property and context features to price, evaluated **as if predicting forward in time** (later periods are never used to train earlier-period models in the primary split).

### 1.2 Research questions (aligned with [task.md](task.md))

1. Which dataset supports credible **forward-in-time** evaluation? (London chosen over Moscow due to usable dates and chronological structure.)
2. Can models trained on earlier calendar periods **generalise** to a held-out recent period?
3. How do **preprocessing choices** (duplicates, outlier handling) affect holdout performance?

---

## 2. Principles from the literature (summary)

| Principle | Design choice in this repo |
|-----------|----------------------------|
| Time-aware evaluation avoids look-ahead bias (e.g. Bergmeir & Benitez, 2012; Cerqueira et al., 2020) | **Calendar-fair** split on `history_date` + **walk-forward** checks in [`src/walk_forward_validation.py`](src/walk_forward_validation.py) |
| Leakage inflates apparent accuracy (Kaufman et al., 2012) | Preprocessing **fit on train only**; **mainline** excludes `saleEstimate_*` / `rentEstimate_*` from predictors |
| Tree ensembles suit mixed tabular data (Hastie et al., 2009; Breiman, 2001) | `HistGradientBoostingRegressor` and `RandomForestRegressor` vs naive median |
| MAE and RMSE complement each other (Willmott & Matsuura, 2005) | Both reported on validation and test |

Full citation mapping appears in [task.md § Literature review](task.md#literature-review).

---

## 3. Data source and cleaning

### 3.1 Source

- Expected path: `dataset/kaggle_london_house_price_data.csv` (see [`dataset/README.md`](dataset/README.md) for download).
- Original scale (pre-cleaning): on the order of **418,201 rows × 28 columns**; calendar span **1995-01-02** to **2024-09-27** for `history_date` ([task.md](task.md)).

### 3.2 Cleaning pipeline

Implemented in [`clean_dataset`](src/london_pipeline.py) (`src/london_pipeline.py`):

1. Parse `history_date` and `history_price`; drop rows missing either.
2. **Deduplication:** exact duplicates on `(fullAddress, history_date, history_price)`, **keep first** (counts documented in [task.md](task.md)).
3. **Target clipping:** quantile clipping at **1% / 99%** on `history_price` (policy chosen after comparing with IQR clipping during project exploration — see task narrative).
4. **Feature engineering (non-leaky):** calendar parts (`history_year`, `history_month`, `history_quarter`), room aggregates and ratios, coarse `outcode_area` from `outcode`, etc.

Vendor-specific datetime/text columns that are not used as stable predictors at inference are excluded from assisted-feature construction where noted in [`src/assisted_track.py`](src/assisted_track.py) (`saleEstimate_ingestedAt`, structured sale-change fields, etc.).

---

## 4. Experimental design: temporal split

### 4.1 Calendar-fair split (main holdout)

Implemented in [`calendar_split`](src/london_pipeline.py):

Let \(t_{\min} = \min(\texttt{history\_date})\), \(t_{\max} = \max(\texttt{history\_date})\), and \(\Delta = t_{\max} - t_{\min}\).

- **Training:** all rows with \(t \leq t_{\min} + 0.70\,\Delta\)
- **Validation:** \(t_{\min} + 0.70\,\Delta < t \leq t_{\min} + 0.80\,\Delta\)
- **Test:** \(t > t_{\min} + 0.80\,\Delta\)

So splits use **70% / 10% / 20% of elapsed calendar time**, **not** 70% / 10% / 20% of row counts. Recent years may contribute more rows — reported train/val/test **row shares** are empirical consequences of the density over time.

Assertions enforce **strict temporal ordering**: maximum train date \(\leq\) minimum validation date; maximum validation date \(\leq\) minimum test date.

### 4.2 Preprocessing leakage control

[`build_preprocessor`](src/london_pipeline.py) fits **only on the training partition**; validation and test receive **transform** only. Imputation statistics and encoders never see future labels or future-feature distributions beyond train.

---

## 5. Feature policies (two tracks)

### 5.1 Mainline (leakage-safe deployment candidate)

Defined implicitly by [`_feature_lists`](src/london_pipeline.py): excludes target, date, raw identifiers, history-change columns, and **any** column whose name starts with `saleEstimate_` or `rentEstimate_`.

Encoding:

- Numeric pipeline: median imputation + `StandardScaler`.
- Categorical pipeline: most-frequent imputation + `OneHotEncoder` with **`sparse_output=True`** and float32 where configured — to control memory on wide one-hot spaces.

### 5.2 Assisted track (governance experiment)

[`src/assisted_track.py`](src/assisted_track.py) builds a **separate** feature matrix that **includes** numeric vendor estimate columns (`saleEstimate_*` among others), subject to explicit drops of ingestion timestamps and nested sale-change columns that are poor inference-time analogues.

**Interpretation:** assisted models quantify **upper-bound-style** accuracy **if** those vendor signals are acceptable at deployment time. They are **not** merged into the mainline definition of “our features only.”

### 5.3 External benchmark (non-trained comparators)

[`src/external_benchmark_estimates.py`](src/external_benchmark_estimates.py) scores raw vendor columns (e.g. `saleEstimate_lowerPrice`) against `history_price` on the **same test partition** — **comparator RMSE**, not a trained mainline feature.

---

## 6. Models and selection strategy

### 6.1 Compared learners

| Role | Implementation |
|------|----------------|
| Naive baseline | Predict test with **median** of training targets |
| Mainline tree ML | `HistGradientBoostingRegressor`, `RandomForestRegressor` ([`run_model_comparison`](src/london_pipeline.py)) |
| Mainline linear ML | `Ridge`, `ElasticNet` (same preprocessor, sparse OHE input) |
| Mainline neural baselines | `MLP_small/medium/large` (sklearn `MLPRegressor` capacity scan, log-target via `TransformedTargetRegressor`), `TorchMLP` ([`src/torch_mlp.py`](src/torch_mlp.py)) |
| Assisted ML | `AssistedHistGBR`, `AssistedRandomForest` ([`src/assisted_track.py`](src/assisted_track.py)) |

**HistGBR** hyperparameters for mainline follow tuned settings documented in code (see bounded search in [`src/hparam_search.py`](src/hparam_search.py)).

### 6.2 Selection criterion

Primary headline for model ranking on the holdout: **lowest test RMSE** among candidates, with validation metrics used for sanity checking and tuning experiments. Naive uplift is reported as **percentage reduction in test RMSE** relative to the naive median baseline (`model_results_summary.csv`).

### 6.3 Neural network baselines

Two NN baselines are reported on the **same calendar split** and **same preprocessor** as the trees, so the comparison is fair. After empirical iteration, the final configurations used in the committed snapshot are:

**sklearn `MLPRegressor` capacity scan** (`MLP_small`, `MLP_medium`, `MLP_large`)
- Architectures: `(64,)`, `(128, 64)`, `(256, 128, 64)`.
- Activation `relu`, solver `adam` (`learning_rate_init=1e-3`), L2 `alpha=1e-4`.
- Early stopping: `validation_fraction=0.15`, `n_iter_no_change=15`, `tol=1e-4`, `max_iter=200`, `batch_size=256`.
- **Target scaler:** `TransformedTargetRegressor(transformer=StandardScaler())`. Rationale: a `log1p`/`expm1` wrap looks attractive for the heavy GBP tail but interacts badly with sklearn's internal early-stop signal — checkpoints selected on log-space R² systematically under-predict the shifted test era, and `expm1` then amplifies that bias multiplicatively (preliminary experiments hit test RMSE > 1 M GBP). A linear `StandardScaler` inverse is **bias-preserving**: errors in z-score space translate to errors in GBP at a fixed ratio, so the early-stop signal is honest.
- Fixed `random_state=42`.

**PyTorch `MLPNet`** ([`src/torch_mlp.py`](src/torch_mlp.py))
- Architecture: `Linear(d, 256) → ReLU → Dropout(0.2) → Linear(256, 128) → ReLU → Dropout(0.2) → Linear(128, 1)`.
- Optimiser `AdamW(lr=1e-3, weight_decay=1e-4)`, loss `SmoothL1Loss` (Huber-style robust loss), `batch_size=1024`, up to 80 epochs.
- **Early stopping in GBP space**: validation RMSE is computed *after* the `expm1` inverse on every epoch and the best-by-GBP-RMSE checkpoint is restored (patience 10). This is the key reason the PyTorch MLP can use a log target safely while the sklearn MLP cannot — the early-stop signal lives in price units, not log units.
- **Log target with two safety guards:**
  1. The final `Linear(128, 1)` is initialised with `weight = 0` and `bias = mean(log1p(y_train))`, so epoch 0 predicts the training log-mean. Without this, random init can produce log values around 24, and `expm1(24) ≈ 26 billion` GBP — observed empirically on the first run.
  2. Log predictions are clipped to `[log_min(y_train) − 0.5, log_max(y_train) + 0.5]` *before* `expm1`, bounding test-time predictions to a sensible GBP envelope and preventing tail explosions.
- Fixed seed `42`.

```mermaid
flowchart LR
  input["Input: 52 features (numeric scaled + OHE)"] --> h1[Linear 256 -> ReLU -> Dropout 0.2]
  h1 --> h2[Linear 128 -> ReLU -> Dropout 0.2]
  h2 --> out["Linear -> 1 (log price)"]
  out --> clip["clip to [log_min-0.5, log_max+0.5]"]
  clip --> inv[expm1]
  inv --> y["y_hat in GBP"]
```

Why NNs are still reported, even though trees usually win on this data:

- **Course alignment:** the cohort expects an explicit NN with a loss curve, an architecture, and a training loop.
- **Theoretical context:** universal approximation (Hornik et al., 1989) guarantees representability but **not** sample efficiency on heterogeneous tabular features.
- **Empirical literature:** tree ensembles tend to outperform deep nets on medium-sized heterogeneous tabular datasets (Shwartz-Ziv & Armon, 2022; Grinsztajn et al., 2022). The observed ordering is consistent with that literature.
- **Walk-forward provides a positive NN finding:** the sklearn MLP has *lower* coefficient of variation across calendar folds than RandomForest (15.1% vs 24.4%) and wins 2 of 4 folds, so on shorter shift windows the NN is competitive.

Loss curves are persisted as `data/MLP_*_loss_curve.{csv,png}` and `data/torch_mlp_loss_curve.png` (mirrored under [`results/`](results/) after `--sync-results`). PyTorch per-epoch history (`epoch`, `train_loss`, `val_rmse`) is at `data/torch_mlp_history.csv`.

### 6.4 Linear baselines (Ridge, ElasticNet)

`Ridge(alpha=1.0)` and `ElasticNet(alpha=0.001, l1_ratio=0.2)` consume the same sparse preprocessor output. They give the table an honest linear reference so the boosting-vs-NN-vs-linear ordering is visible in `model_results_summary.csv`.

---

## 7. Metrics (definitions)

For test observations with true values \(y_i\) and predictions \(\hat{y}_i\), \(i = 1,\ldots,n\):

\[
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|
\]

\[
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2}
\]

Both are in **the same currency units** as `history_price`. RMSE penalises large errors more heavily than MAE (important for **tail prices**).

### 7.1 Regression-friendly “accuracy” surrogates

Because the target is continuous, classification **accuracy** does not apply directly. The pipeline therefore reports:

- **\(R^2\)** — fraction of variance explained.
- **MAPE** — mean of \(\lvert (y - \hat y)/y\rvert\) on rows with \(\lvert y \rvert > 100\) GBP (avoids the divide-by-near-zero pathology).
- **Within-10% rate** — share of rows with relative error below \(10\%\) (same support filter as MAPE).

### 7.2 Derived classification view on price bins

For audiences that explicitly want an accuracy percentage, [`src/regression_diagnostics.py`](src/regression_diagnostics.py) discretises both `y_true` and the predictions of the best model into **five quantile bins** (of `y_true`) and writes:

- `price_bin_confusion.csv` / `.png` — the 5×5 confusion matrix.
- `price_bin_classification_report.csv` — per-bin precision/recall/F1/support.
- `price_bin_classification_summary.csv` — overall **accuracy**, **macro F1**, **weighted F1**.

This is a derived proxy task; the regression metrics above remain the primary signal.

---

## 8. Extended evaluation protocol

### 8.1 Walk-forward stability

[`src/walk_forward_validation.py`](src/walk_forward_validation.py) trains **both a RandomForest and an MLP (`MLP_medium`, log-target)** on expanding/rolling past windows and reports RMSE on forward windows for each family. `walk_forward_results.csv` has columns `fold, model_family, train_rows, val_rows, rmse, rmse_mean, rmse_std, cv_pct`; the consolidated report ([`results/run_report.md`](results/run_report.md)) prints mean RMSE and CV% **per family**. The release gate (Section 9) is evaluated on the **RandomForest** family for continuity with prior snapshots.

### 8.2 Segment-level error

[`segment_metrics`](src/london_pipeline.py) aggregates MAE/RMSE by **outcode area**, **property type**, and **price band** (quantile bins of true price on the test side), with a minimum count threshold per segment to avoid empty groups.

### 8.3 Year-bucket error

[`error_by_year_bucket`](src/london_pipeline.py) — RMSE by calendar year on the test prediction table.

### 8.4 RandomForest explainability (mainline)

Feature importances and a **subsampled permutation importance** on validation (see `run_london_pipeline` output) — outputs to `data/feature_importance.csv` when run (not all columns copied to `results/` by default).

---

## 9. Governance report and release gates

[`src/main.py`](src/main.py) consolidates CSV artifacts under `data/` into **`run_report.md`** and writes:

- [`model_decision_summary.csv`](results/model_decision_summary.csv) — primary vs benchmark RMSE, assisted RMSE, recommendation strings.
- [`track_comparison_summary.csv`](results/track_comparison_summary.csv) — cross-track MAE/RMSE deltas.
- [`segment_blocker_actions.csv`](results/segment_blocker_actions.csv) — worst high-support segments with suggested owner/mitigation.

**Illustrative automated gates** (see frozen outcome in [`results/run_report.md`](results/run_report.md)):

- Artifact presence / pipeline completes without crash.
- Mainline test RMSE \(\leq\) historical regression threshold (documented in code/report generation).
- Walk-forward CV \(\leq\) configured ceiling (e.g. 26%).
- **Segment gate:** for the **best mainline model name**, only segments with **`rows >= 200`** can trigger failure; failure if RMSE \(> 1.8 \times\) overall mainline test RMSE.

These gates are **heuristic release checks**, not legal warranties.

---

## 10. Frozen results snapshot (committed)

The following mirror the last committed [`results/RUN_STAMP.txt`](results/RUN_STAMP.txt) run. **Re-runs may differ slightly.**

| Item | Source file | Headline (illustrative) |
|------|-------------|-------------------------|
| Mainline best | `results/model_results_summary.csv` | HistGBR test RMSE **393,489.27** |
| Walk-forward | `results/walk_forward_results.csv` | CV ~**24.40%** |
| External benchmark | `results/external_estimate_benchmark.csv` | Best single column: `saleEstimate_lowerPrice`, RMSE **350,168.21** |
| Assisted best | `results/assisted_track_results.csv` | AssistedHistGBR test RMSE **304,436.22** |
| Consolidated narrative | [`results/run_report.md`](results/run_report.md) | Includes gates, blockers, governance recommendation |

Full tables: CSV files in [`results/`](results/).

---

## 11. Artifacts map (reproducibility)

| Script | Primary outputs (working dir `data/`, gitignored) |
|--------|---------------------------------------------------|
| [`src/run_london_pipeline.py`](src/run_london_pipeline.py) | `model_results_summary.csv`, predictions, segment metrics, feature importance, registry JSON |
| [`src/assisted_track.py`](src/assisted_track.py) | `assisted_track_results.csv`, `assisted_track_predictions.csv` |
| [`src/walk_forward_validation.py`](src/walk_forward_validation.py) | `walk_forward_results.csv` |
| [`src/external_benchmark_estimates.py`](src/external_benchmark_estimates.py) | `external_estimate_benchmark.csv` |
| [`src/hparam_search.py`](src/hparam_search.py) | `hparam_search_results.csv` |
| [`src/main.py`](src/main.py) | `run_report.md`, `model_decision_summary.csv`, `track_comparison_summary.csv`, `segment_blocker_actions.csv` |

**Commit policy:** after a full evaluation, run `python "src/main.py" --report-only --sync-results` to mirror summaries into [`results/`](results/) (see [README.md](README.md), [`GOVERNANCE.md`](GOVERNANCE.md)).

**Full command bundle** to reproduce end-to-end:

```powershell
python "src/run_london_pipeline.py"
python "src/assisted_track.py"
python "src/walk_forward_validation.py"
python "src/external_benchmark_estimates.py"
python "src/main.py" --report-only --sync-results
```

Or: `python "src/main.py" --all --sync-results` (runs baseline, walk-forward, benchmark, assisted, report, then copies frozen summaries to `results/`).

Dependencies: [`requirements.txt`](requirements.txt). Python **3.11+** recommended.

---

## 12. Limitations and future work

1. **Scale of errors:** MAE/RMSE in raw currency are hard to interpret without normalising by typical price; consider **log-target** or **percentage errors** in future work.
2. **Tail segments:** high price bands and rare property types drive large RMSE; segment gate may **fail** until mitigated (features, weighting, or business sign-off) — see [`results/segment_blocker_actions.csv`](results/segment_blocker_actions.csv).
3. **Assisted track** depends on **vendor availability** and policy; it must not be conflated with mainline for **leakage-safe** claims.
4. **Reproducibility:** fixed `random_state` where set; full bitwise reproducibility across OS/library versions is not guaranteed.
5. **Walk-forward** uses a fixed RandomForest configuration for speed/stability; it is a **stability** instrument, not the same as the final tuned HistGBR in every fold.

---

## 13. References (abbreviated)

See [task.md § Literature review](task.md#literature-review) for the full mapping table and citations: Bergmeir & Benítez (2012), Cerqueira et al. (2020), Kaufman et al. (2012), Hastie et al. (2009), Breiman (2001), Willmott & Matsuura (2005).

Additional references introduced for the NN baselines (Section 6.3):

- Hornik, K., Stinchcombe, M., White, H. (1989). *Multilayer feedforward networks are universal approximators.* **Neural Networks**, 2(5), 359–366.
- Shwartz-Ziv, R., Armon, A. (2022). *Tabular data: Deep learning is not all you need.* **Information Fusion**, 81, 84–90.
- Grinsztajn, L., Oyallon, E., Varoquaux, G. (2022). *Why do tree-based models still outperform deep learning on typical tabular data?* **NeurIPS 2022 Datasets and Benchmarks Track**.

---

## Document history

- **Purpose:** single reference for methodology chapters, ethics/governance discussion, and reproducibility audits.
- **Canonical numeric snapshot:** [`results/run_report.md`](results/run_report.md) + companion CSVs in [`results/`](results/).
