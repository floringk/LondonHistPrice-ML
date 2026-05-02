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
| Mainline ML | `HistGradientBoostingRegressor`, `RandomForestRegressor` ([`run_model_comparison`](src/london_pipeline.py)) |
| Assisted ML | `AssistedHistGBR`, `AssistedRandomForest` ([`src/assisted_track.py`](src/assisted_track.py)) |

**HistGBR** hyperparameters for mainline follow tuned settings documented in code (see bounded search in [`src/hparam_search.py`](src/hparam_search.py)).

### 6.2 Selection criterion

Primary headline for model ranking on the holdout: **lowest test RMSE** among candidates, with validation metrics used for sanity checking and tuning experiments. Naive uplift is reported as **percentage reduction in test RMSE** relative to the naive median baseline (`model_results_summary.csv`).

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

---

## 8. Extended evaluation protocol

### 8.1 Walk-forward stability

[`src/walk_forward_validation.py`](src/walk_forward_validation.py) trains a **RandomForest** on expanding/rolling past windows and reports RMSE on forward windows. Summarise **mean RMSE**, **std**, and **coefficient of variation (CV%)** across folds in `walk_forward_results.csv` (see [`results/walk_forward_results.csv`](results/walk_forward_results.csv)).

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

---

## Document history

- **Purpose:** single reference for methodology chapters, ethics/governance discussion, and reproducibility audits.
- **Canonical numeric snapshot:** [`results/run_report.md`](results/run_report.md) + companion CSVs in [`results/`](results/).
