# Report structure (IMRaD) — example tables from frozen `results/`

Use this as the skeleton for a paper or technical report. **Numbers below** come from the committed snapshot in [`results/`](../results/) (same period as [`results/RUN_STAMP.txt`](../results/RUN_STAMP.txt)). Round display values as your style guide requires; keep full precision in appendix if needed.

**Citation hint:** “Metrics computed from repository artifacts as of [git commit / date]; primary tables: `results/model_results_summary.csv`, `results/run_report.md`.”

---

## Title & authors

Suggest: *London housing prices: temporal evaluation of gradient boosting and random forests with a vendor-assisted comparison track*

---

## Abstract (100–200 words)

Fill with: prediction target (`history_price`), UK London transactions, **calendar-based** holdout, mainline models excluding vendor sale estimates, headline test RMSE/MAE and naive uplift, one line on walk-forward CV, one line on assisted/benchmark for context, one limitation (segments or currency-scale metrics).

---

## 1. Introduction

### 1.1 Problem

- Supervised regression: predict `history_price` from structural and location-related features.
- Temporal framing: evaluation must reflect **forward-in-time** performance, not random mixing of years.

### 1.2 Contributions (adapt to assignment)

- Chronological split and train-only preprocessing pipeline (implementation: `src/london_pipeline.py`).
- Comparison of naive baseline, RandomForest, HistGBR on a shared holdout.
- Walk-forward stability checks (`src/walk_forward_validation.py`).
- Separately reported **assisted** track using vendor estimates (`src/assisted_track.py`) and raw-column **benchmark** RMSE (`src/external_benchmark_estimates.py`) for context—not merged into mainline training.

---

## 2. Methods

### 2.1 Data

- Source path (after download): `dataset/kaggle_london_house_price_data.csv` (see `dataset/README.md`).
- Target: `history_price`; time index: `history_date`.
- Cleaning: duplicates key `fullAddress + history_date + history_price` (keep first); target winsorization via quantile clipping in pipeline code; engineered calendar and aggregate features as in `clean_dataset` (`src/london_pipeline.py`).

### 2.2 Train / validation / test split

- **Calendar-fair:** first **70%**, next **10%**, final **20%** of **elapsed time** between min and max `history_date` (not 70% of rows).
- Preprocessing **fit on train only** (`build_preprocessor`).

### 2.3 Mainline feature policy

- Exclude columns starting with `saleEstimate_` and `rentEstimate_` from mainline predictors (leakage / policy).
- Numeric: imputation + scaling; categorical: one-hot with unknown handling.

### 2.4 Models

- **NaiveMedian:** predict test set with median of training targets.
- **RandomForestRegressor** and **HistGradientBoostingRegressor** with project hyperparameters (see code / `hparam_search` notes).

### 2.5 Metrics

- **MAE**, **RMSE** on validation and test (same currency as prices).
- **Walk-forward:** rolling folds; report mean RMSE and coefficient of variation across folds.

### 2.6 Assisted track and benchmark (optional subsection)

- **Assisted:** same split; includes numeric vendor estimate columns in features — report separately.
- **Benchmark:** RMSE of raw `saleEstimate_*` columns vs true price on test rows — comparator, not a trained mainline feature.

---

## 3. Results

### 3.1 Mainline model comparison (holdout)

**Table 1.** Validation and test performance (mainline track). Source: [`results/model_results_summary.csv`](../results/model_results_summary.csv).

| Model        | Val MAE (GBP) | Val RMSE (GBP) | Test MAE (GBP) | Test RMSE (GBP) | Test RMSE improvement vs naive (%) |
|-------------|----------------:|----------------:|----------------:|----------------:|-------------------------------------:|
| HistGBR     | 202,361.51     | 404,783.94     | 216,082.64     | 393,489.27     | 57.58 |
| RandomForest| 202,493.47     | 414,854.27     | 215,149.40     | 396,947.22     | 57.21 |
| NaiveMedian | 558,363.56     | 965,708.34     | 561,898.98     | 927,645.60     | 0.00 |

**Takeaway for text:** HistGBR achieves the lowest **test RMSE** among compared mainline models; both ML models greatly outperform the naive baseline.

### 3.2 Walk-forward stability

**Table 2.** Fold-level RMSE (RandomForest protocol). Source: [`results/walk_forward_results.csv`](../results/walk_forward_results.csv).

| Fold | Train rows | Val rows | RMSE (GBP) |
|------|------------|---------:|-----------:|
| 1 (2000-12–2006-11 window) | 44,642 | 50,806 | 224,290.76 |
| 2 (2006-11–2012-11 window) | 95,448 | 36,453 | 341,327.72 |
| 3 (2012-11–2018-10 window) | 131,901 | 53,297 | 462,977.05 |
| 4 (2018-10–2024-09 window) | 185,198 | 130,470 | 361,405.18 |

**Summary:** Mean RMSE **347,500.18**; std **84,774.58**; CV **24.40%** (see [`results/run_report.md`](../results/run_report.md)).

### 3.3 External benchmark (single-column comparators)

**Table 3.** Test MAE/RMSE for selected `saleEstimate_*` columns. Source: [`results/external_estimate_benchmark.csv`](../results/external_estimate_benchmark.csv).

| Benchmark column           | Test rows used | Test MAE (GBP) | Test RMSE (GBP) |
|---------------------------|----------------:|---------------:|----------------:|
| saleEstimate_lowerPrice   | 130,196        | 105,702.66    | 350,168.21 |
| saleEstimate_currentPrice | 130,196        | 129,807.75    | 416,009.91 |
| saleEstimate_upperPrice   | 130,196        | 182,243.06    | 516,301.64 |

### 3.4 Assisted vs mainline vs benchmark (cross-track)

**Table 4.** Aggregated RMSE/MAE comparison. Source: [`results/track_comparison_summary.csv`](../results/track_comparison_summary.csv).

| Track / comparator | Test RMSE (GBP) | Test MAE (GBP) |
|--------------------|----------------:|---------------:|
| Mainline (best)    | 393,489.27      | 216,082.64 |
| Assisted (best)    | 304,436.22      | 158,399.33 |
| External benchmark (best column) | 350,168.21 | 105,702.66 |

**Narrative caution:** benchmark MAE is **not** directly comparable to full multivariate models in interpretation (single column vs rich feature set); use as **context**.

### 3.5 Automated governance summary (optional)

From [`results/model_decision_summary.csv`](../results/model_decision_summary.csv):

| Field | Value |
|-------|-------|
| Recommendation | open_assisted_track |
| Governance recommendation | deploy_assisted |
| Delta (primary RMSE − benchmark RMSE) | +43,321.06 |

Explain in prose: “deploy_assisted” here means **governance hint from RMSE ordering**, not automatic production—see [`GOVERNANCE.md`](../GOVERNANCE.md).

### 3.6 Release gates & segment blockers

From [`results/run_report.md`](../results/run_report.md):

- Crash / artifact gate: **PASS**
- Primary RMSE threshold: **PASS**
- Walk-forward CV ≤ 26%: **PASS**
- High-support segment RMSE gate: **FAIL** — segments listed under **Blockers** in `run_report.md` (expensive tail, selected property types).

Use this honestly in **Discussion / limitations**.

---

## 4. Discussion

Suggested paragraphs:

1. **Interpretation:** Mainline HistGBR gives strong uplift vs naive; errors remain large in GBP because prices span a wide range—optional future work: log or percentage metrics.
2. **Vendor estimates:** Assisted track and single-column benchmarks show that vendor channels carry signal; mainline excludes them to avoid leakage and deployment dependency—align conclusions with course ethics / brief.
3. **Segments:** Failed segment gate highlights unequal error across price bands and types; propose mitigation (features, stratified objectives) or scoped deployment.
4. **Reproducibility:** Fixed seeds where applicable; document Python/sklearn versions (`requirements.txt`).

---

## 5. Conclusion

- Summarize best **mainline** outcome (HistGBR test RMSE / MAE, uplift vs naive).
- One line on walk-forward CV.
- One line on assisted/benchmark **only if** your assignment asks for full comparison.
- Restate limitations.

---

## References

Pull citations from [`task.md`](../task.md) literature section; align bibliography style with course (APA/IEEE/Harvard).

---

## Appendix (optional)

- Commands to reproduce: [`README.md`](../README.md).
- Full methodology: [`methodology.md`](../methodology.md).
- Feature lists and code paths: `src/london_pipeline.py`.
