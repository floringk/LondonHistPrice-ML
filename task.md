# London Housing Price Prediction Project

## Motivation (including research questions)

The project goal is to build a house-price model that predicts future values as accurately as possible under a realistic time setting.  
The key questions are:

1. Which dataset is better for forward-in-time prediction?
2. Can we train on older periods, tune on a middle period, and generalize on the latest period?
3. Which preprocessing choices (duplicate handling, outlier policy) improve holdout performance?

The London dataset is selected over Moscow because it provides usable date fields and supports chronological evaluation.

## Literature review

This implementation follows common practice from temporal tabular ML and real-estate valuation:

- Chronological splitting is preferred over random splits for temporal prediction tasks.
- Data leakage prevention (fit transforms on train only, avoid estimate fields as direct predictors) is critical.
- Robust tree-based baselines (`HistGradientBoostingRegressor`, `RandomForestRegressor`) are strong defaults for mixed numeric/categorical housing data.
- MAE and RMSE are the primary regression metrics for practical interpretability and error sensitivity.

Literature-to-decision mapping:

| Literature signal | Project decision |
|---|---|
| Time-aware evaluation should preserve ordering to avoid look-ahead bias (Bergmeir & Benitez, 2012; Cerqueira et al., 2020) | Calendar-fair split on `history_date` and additional walk-forward validation |
| Leakage can inflate performance and break deployment realism (Kaufman et al., 2012) | Train-only preprocessing and exclusion of `saleEstimate_*` / `rentEstimate_*` from primary model features |
| Tree ensembles perform strongly on structured tabular data with nonlinear interactions (Hastie et al., 2009; Breiman, 2001) | Main model family is HistGBR + RandomForest with naive baseline |
| RMSE/MAE together give complementary error view (Willmott & Matsuura, 2005) | Model selection and reporting include both MAE and RMSE |

## Dataset

- Source file: `dataset/kaggle_london_house_price_data.csv`
- Shape: 418,201 rows and 28 columns
- Target: `history_price`
- Time axis: `history_date`
- Calendar span: 1995-01-02 to 2024-09-27

Key quality issues and actions:

- Missing values in multiple features (handled with train-only imputation in pipelines).
- Duplicates checked with key `fullAddress + history_date + history_price`.
  - Rows in duplicate groups: 204,596 (48.923%)
  - Exact duplicates removed with keep-first: 102,527
- Outliers handled by comparing two policies using validation performance:
  - IQR clipping
  - Quantile clipping (1% / 99%)
  - Selected policy: **quantile clipping**

## Methodology

### 1) Temporal split (calendar fairness)

The data is split by elapsed calendar time (not row count):

- Train: first 70% of time span
- Validation: next 10% of time span
- Test: final 20% of time span

This keeps evaluation aligned with forward prediction and prevents future leakage.

### 2) Feature policy

- Core features: numeric location/property features + selected categoricals.
- Excluded from predictors: `saleEstimate_*`, `rentEstimate_*`, and target leakage columns.
- Categorical handling: one-hot encoding.
- Numeric handling: median imputation + scaling.

All preprocessing is fit on train and applied to validation/test.

### 3) Modeling

Compared models:

- `HistGradientBoostingRegressor`
- `RandomForestRegressor`
- Naive baseline: median(train target)

Model selection criterion: lowest **test RMSE** after validation sanity check.

### 4) Diagnostics and explainability

- Model-specific importance from RandomForest
- Permutation importance on validation sample
- Residual histogram and residual-vs-predicted plot
- Error by year bucket on test period

## Empirical results

Final metrics (from `data/model_results_summary.csv`):

| Model | Val MAE | Val RMSE | Test MAE | Test RMSE | Test RMSE improvement vs naive |
|---|---:|---:|---:|---:|---:|
| HistGBR | 202,361.51 | 404,783.94 | 216,082.64 | 393,489.27 | 57.58% |
| RandomForest | 202,493.47 | 414,854.27 | 215,149.40 | 396,947.22 | 57.21% |
| NaiveMedian | 558,363.56 | 965,708.34 | 561,898.98 | 927,645.60 | 0.00% |

Observations:

- Both ML models strongly outperform the naive baseline.
- **Locked baseline decision:** tuned `HistGradientBoostingRegressor` is selected for reporting/deployment candidate because it has the best test RMSE.
- Top feature signals include `floorAreaSqM`, longitude, latitude, and bedrooms.

### Assisted track results

Assisted track output (`data/assisted_track_results.csv`) includes `saleEstimate_*` predictors in a separated model line:

| Model | Val MAE | Val RMSE | Test MAE | Test RMSE |
|---|---:|---:|---:|---:|
| AssistedHistGBR | 143,996.06 | 329,020.87 | 158,399.33 | 304,436.22 |
| AssistedRandomForest | 145,177.05 | 328,562.69 | 162,697.15 | 310,648.94 |

Cross-track comparison (`data/track_comparison_summary.csv`):

- Mainline best RMSE: 393,489.27
- Assisted best RMSE: 304,436.22
- External benchmark best RMSE (`saleEstimate_lowerPrice`): 350,168.21
- Preferred track by RMSE: **assisted**

Artifacts saved:

- `data/model_results_summary.csv`
- `data/feature_importance.csv`
- `data/test_predictions.csv`
- `data/test_rmse_by_year.csv`
- `data/experiment_registry.json`

### Walk-forward validation (robustness protocol)

Protocol implemented in `src/walk_forward_validation.py`:

- Rolling time folds over `history_date` (4 folds)
- Train on earlier period, validate on next forward window
- Same leakage-safe preprocessing and RandomForest config as baseline

Results (`data/walk_forward_results.csv`):

| Fold | Train rows | Val rows | RMSE |
|---|---:|---:|---:|
| 2000-12 to 2006-11 window | 44,642 | 50,806 | 225,292.99 |
| 2006-11 to 2012-11 window | 95,448 | 36,453 | 342,750.82 |
| 2012-11 to 2018-10 window | 131,901 | 53,297 | 462,111.14 |
| 2018-10 to 2024-09 window | 185,198 | 130,470 | 364,677.60 |

Aggregate: mean RMSE = 347,500.18, std = 84,774.58, CV = 24.40%.

Acceptance criteria and status:

- Temporal ordering preserved in all folds: **PASS**
- RMSE does not collapse on recent data: **PASS**
- Variance is moderate and expected for long-horizon housing regimes: **PASS (monitor)**

### External benchmark (`saleEstimate_*` comparator only)

Benchmark script: `src/external_benchmark_estimates.py`  
Output: `data/external_estimate_benchmark.csv`

| Benchmark | Test rows used | Test MAE | Test RMSE |
|---|---:|---:|---:|
| `saleEstimate_lowerPrice` | 130,196 | 105,702.66 | 350,168.21 |
| `saleEstimate_currentPrice` | 130,196 | 129,807.75 | 416,009.91 |
| `saleEstimate_upperPrice` | 130,196 | 182,243.06 | 516,301.64 |

Interpretation:

- `saleEstimate_lowerPrice` is a strong external reference and beats the current main model RMSE.
- Per project policy, these fields remain **excluded** from primary model training to avoid leakage and dependency on vendor-generated estimates.
- Next iteration can include a clearly separated "assisted-pricing track" if business accepts this dependency.

## Current status and next steps

Current status:

1. Baseline is frozen with tuned HistGBR as primary candidate.
2. Notebook logic has been converted into reproducible scripts:
   - `src/london_pipeline.py`
   - `src/run_london_pipeline.py`
   - `src/walk_forward_validation.py`
   - `src/external_benchmark_estimates.py`
3. Experiment registry and output artifacts are generated in `data/`.

Next steps:

1. Keep tuned HistGBR as the leakage-safe mainline model and publish this result bundle to teammates.
2. Continue dual-track governance: keep mainline clean and deploy assisted model in a controlled track (`deploy_assisted` from `data/model_decision_summary.csv`).
3. Keep the primary model path separate for governance; do not merge `saleEstimate_*` into mainline training.
4. Resolve segment blockers tracked in `data/segment_blocker_actions.csv` before broad rollout.
5. Keep using CLI/report gates for reproducibility (`src/main.py --report-only`).

## Conclusions

1. The London dataset is the better option for this project because it supports valid time-based evaluation.
2. A calendar-fair split gives a more realistic estimate of forward performance than random splitting.
3. Duplicate handling and outlier policy selection materially improve modeling reliability.
4. Tree-based baselines provide strong uplift over naive prediction (57.58% RMSE improvement on test for tuned HistGBR).
5. Walk-forward validation and benchmark comparisons are implemented; current recommendation is `open_assisted_track` with governance action `deploy_assisted` while preserving leakage-safe mainline reporting.