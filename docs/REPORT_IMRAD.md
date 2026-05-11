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
- **MLPRegressor** (optional NN-style baseline): small feed-forward network trained on a **random dense subsample** of the transformed training matrix (same preprocessor as trees); full validation/test predictions via batched dense conversion when the design matrix is sparse. Skipped automatically if one-hot dimensionality exceeds a safety threshold.

### 2.5 Metrics

- **MAE**, **RMSE** on validation and test (same currency as prices).
- **R²**, **MAPE**, and **within-10% rate** (share of rows with \(|y-\hat y|/|y|<0.10\) among rows with \(|y|>100\) GBP) — regression-friendly complements when a single “accuracy” number is requested.
- **Walk-forward:** rolling folds; report mean RMSE and coefficient of variation across folds.

### 2.6 Diagnostics (regression)

- **Not** a classification confusion matrix on the continuous target. After each baseline run, the pipeline writes **prediction vs actual** and **residual** figures, plus a **price-bin confusion matrix** (quantile bins on test `y_true`, same bins applied to predictions) — useful for presentation; interpret as a derived discrete view, not the primary optimisation objective.

### 2.7 Assisted track and benchmark (optional subsection)

- **Assisted:** same split; includes numeric vendor estimate columns in features — report separately.
- **Benchmark:** RMSE of raw `saleEstimate_*` columns vs true price on test rows — comparator, not a trained mainline feature.

---

## 3. Results

### 3.1 Mainline model comparison (holdout)

**Table 1.** Validation and test performance, mainline track, ranked by test RMSE (lower is better). All models share the calendar split and the same preprocessor; "within-10%" is the share of test rows with `|y - y_hat| / |y| < 0.10` among rows where `|y| > 100` GBP. Source: [`results/model_results_summary.csv`](../results/model_results_summary.csv). Run stamp: [`results/RUN_STAMP.txt`](../results/RUN_STAMP.txt).

| Model        | Family | Val RMSE (GBP) | Test MAE (GBP) | Test RMSE (GBP) | Test R² | Test MAPE | Within 10% | Test RMSE vs naive (%) |
|--------------|--------|----------------:|----------------:|----------------:|--------:|----------:|----------:|----------------------:|
| **HistGBR**  | Tree boosting     | 404,783.94 | 216,082.64 | **393,489.27** | 0.720 | 0.338 | 0.243 | **57.58** |
| RandomForest | Tree ensemble     | 414,854.27 | 215,149.40 | 396,947.22 | 0.715 | 0.329 | 0.232 | 57.21 |
| Ridge        | Linear (L2)       | 603,695.51 | 375,218.61 | 562,967.66 | 0.426 | 0.745 | 0.109 | 39.31 |
| ElasticNet   | Linear (L1+L2)    | 604,209.00 | 374,925.88 | 563,113.40 | 0.426 | 0.744 | 0.109 | 39.30 |
| MLP_medium   | NN (sklearn)      | 448,223.72 | 444,363.38 | 645,104.30 | 0.247 | 0.852 | 0.109 | 30.46 |
| MLP_large    | NN (sklearn)      | 480,800.68 | 477,584.44 | 721,999.16 | 0.056 | 0.887 | 0.114 | 22.17 |
| TorchMLP     | NN (PyTorch)      | 428,184.98 | 470,268.29 | 729,943.31 | 0.035 | 0.796 | 0.106 | 21.31 |
| MLP_small    | NN (sklearn)      | 468,588.16 | 619,645.38 | 838,729.34 | −0.274 | 1.173 | 0.059 | 9.59 |
| NaiveMedian  | Reference         | 965,708.34 | 561,898.98 | 927,645.60 | −0.558 | 0.570 | 0.035 | 0.00 |

**Takeaway for text:** HistGBR achieves the lowest test RMSE among compared mainline models. The boosting / random forest pair sits about 30% below the best linear baseline on test RMSE, which is consistent with the boosting > linear gap reported on heterogeneous tabular data (Grinsztajn et al., 2022). NN baselines are positioned between linear and naive on the held-out test set — see §3.7 for why this is *expected* and how walk-forward (§3.2) reveals a positive NN finding.

### 3.2 Walk-forward stability

**Table 2.** Per-fold test RMSE for two model families on the same expanding-window folds. Source: [`results/walk_forward_results.csv`](../results/walk_forward_results.csv).

| Fold (validation window) | Train rows | Val rows | RandomForest RMSE (GBP) | MLP RMSE (GBP) | Lower (winner) |
|--------------------------|-----------:|---------:|------------------------:|---------------:|:---------------|
| 1 (2000-12 → 2006-11)    |     44,642 |   50,806 |              224,290.76 |     293,533.19 | RandomForest  |
| 2 (2006-11 → 2012-11)    |     95,448 |   36,453 |              341,327.72 |     312,845.88 | **MLP**       |
| 3 (2012-11 → 2018-10)    |    131,901 |   53,297 |              462,977.05 |     411,278.64 | **MLP**       |
| 4 (2018-10 → 2024-09)    |    185,198 |  130,470 |              361,405.18 |     408,975.44 | RandomForest  |

**Table 3.** Walk-forward summary across folds (lower is better; lower CV % = more stable).

| Family       | Mean RMSE (GBP) | Std RMSE (GBP) | CV %   | Folds won |
|--------------|----------------:|---------------:|-------:|----------:|
| RandomForest |      347,500.18 |      84,774.58 |  24.40 |     2 / 4 |
| **MLP**      |  **356,658.29** |  **53,909.12** | **15.12** | **2 / 4** |

**Takeaway:** MLP loses by only 2.6% on mean RMSE but is **38% more stable** across calendar shifts (15.1% CV vs 24.4%). This is the strongest positive NN finding in this study: under realistic temporal shift the neural network is *more reliable* even if it isn't the single-shot RMSE leader.

### 3.3 External benchmark (single-column comparators)

**Table 4.** Test MAE/RMSE for selected `saleEstimate_*` columns evaluated directly against `history_price` on the test rows where the column is populated. Source: [`results/external_estimate_benchmark.csv`](../results/external_estimate_benchmark.csv).

| Benchmark column           | Test rows used | Test MAE (GBP) | Test RMSE (GBP) |
|---------------------------|----------------:|---------------:|----------------:|
| saleEstimate_lowerPrice   | 130,196        | 105,702.66    | 350,168.21 |
| saleEstimate_currentPrice | 130,196        | 129,807.75    | 416,009.91 |
| saleEstimate_upperPrice   | 130,196        | 182,243.06    | 516,301.64 |

### 3.4 Assisted vs mainline vs benchmark (cross-track)

**Table 5.** Best per track (mainline, assisted, benchmark) on the calendar test set. Source: [`results/track_comparison_summary.csv`](../results/track_comparison_summary.csv).

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

### 3.5b Regression diagnostic figures

**Figure 6.** HistGBR predictions vs actual log-GBP price on the test set; diagonal `y = x` reference line — [`results/regression_pred_vs_actual.png`](../results/regression_pred_vs_actual.png). Heavy clustering on the diagonal in the 200k–800k GBP band; visible scatter widens above 1.5 M GBP, consistent with the segment gate in §3.6.

**Figure 7.** HistGBR residuals: histogram (top) and residuals vs fitted (bottom) — [`results/regression_residuals.png`](../results/regression_residuals.png). The residual histogram is centred near zero but right-skewed (large positive residuals where the model under-predicts expensive transactions); the fan in residuals-vs-fitted is heteroscedastic, again pointing to the expensive tail as the dominant error source.

### 3.6 Release gates & segment blockers

From [`results/run_report.md`](../results/run_report.md):

- Crash / artifact gate: **PASS**
- Primary RMSE threshold: **PASS**
- Walk-forward CV ≤ 26%: **PASS**
- High-support segment RMSE gate: **FAIL** — segments listed under **Blockers** in `run_report.md` (expensive tail, selected property types).

Use this honestly in **Discussion / limitations**.

### 3.7 Neural network results

Two NN baselines run on the **same calendar split** and **same preprocessor** as trees (configurations: see [`methodology.md`](../methodology.md) §6.3):

- **sklearn capacity scan:** rows `MLP_small (64,)`, `MLP_medium (128, 64)`, `MLP_large (256, 128, 64)` in [`results/model_results_summary.csv`](../results/model_results_summary.csv). `TransformedTargetRegressor(StandardScaler())`, Adam, L2, early stopping on validation score (patience 15).
- **PyTorch baseline:** row `TorchMLP` from [`src/torch_mlp.py`](../src/torch_mlp.py). `Linear(256) → ReLU → Dropout → Linear(128) → ReLU → Dropout → Linear(1)` on log target, AdamW + SmoothL1Loss, early stopping on val RMSE in GBP space (patience 10), bias-initialised to training log-mean and log-clipped for numerical safety.

**Figure 1.** sklearn `MLP_small` training-loss curve — [`results/MLP_small_loss_curve.png`](../results/MLP_small_loss_curve.png).
**Figure 2.** sklearn `MLP_medium` training-loss curve — [`results/MLP_medium_loss_curve.png`](../results/MLP_medium_loss_curve.png).
**Figure 3.** sklearn `MLP_large` training-loss curve — [`results/MLP_large_loss_curve.png`](../results/MLP_large_loss_curve.png).
**Figure 4.** PyTorch `TorchMLP` validation RMSE per epoch with early-stop checkpoint at epoch 21 — [`results/torch_mlp_loss_curve.png`](../results/torch_mlp_loss_curve.png). Per-epoch table: [`results/torch_mlp_history.csv`](../results/torch_mlp_history.csv).

**Takeaway:** All four NN configurations are reported in the same ranked table (§3.1) so that boosting / linear / NN are visible end-to-end. On the single 2018–2024 holdout, boosting leads, consistent with the published finding that tree ensembles win on medium-sized heterogeneous tabular data (Shwartz-Ziv & Armon, 2022; Grinsztajn et al., 2022). The NN's positive evidence comes from walk-forward (§3.2): MLP has **38% lower coefficient of variation** across folds than RandomForest and wins folds 2 and 3.

### 3.8 Linear baselines

Rows `Ridge` and `ElasticNet` (added in [`src/london_pipeline.py`](../src/london_pipeline.py)) use the same sparse preprocessor as the trees. They give the report a clean linear reference so the boosting / linear / NN ordering is visible in one table. Test RMSE ≈ 563k GBP; about 30% worse than HistGBR's 393k. This is the empirical "trees > linear" gap the assignment expected to see articulated.

### 3.9 Bin classification view (for the "accuracy %" question)

The best regression model's predictions are also discretised into **5 quantile bins** of `y_true`; the resulting 5×5 confusion matrix is shown in **Figure 5** ([`results/price_bin_confusion.png`](../results/price_bin_confusion.png)).

**Table 6.** Derived classification metrics on the bin-classification proxy task. Sources: [`results/price_bin_classification_summary.csv`](../results/price_bin_classification_summary.csv) (aggregate), [`results/price_bin_classification_report.csv`](../results/price_bin_classification_report.csv) (per-bin).

| Metric | Value |
|--------|------:|
| **Overall accuracy** (HistGBR, 5 quantile bins) | **49.5%** |
| Macro F1 | 0.495 |
| Weighted F1 | 0.497 |
| Random-guess baseline | 20.0% |

| Bin | Price range (GBP)        | Precision | Recall | F1   | Support |
|----:|--------------------------|----------:|-------:|-----:|--------:|
| 0   | 42,995 → 370,000         | 0.544     | 0.703  | 0.613 | 25,953 |
| 1   | 370,000 → 487,000        | 0.338     | 0.382  | 0.359 | 26,212 |
| 2   | 487,000 → 650,000        | 0.338     | 0.324  | 0.331 | 25,352 |
| 3   | 650,000 → 1,000,000      | 0.464     | 0.381  | 0.419 | 26,344 |
| 4   | 1,000,000 → 4,400,000    | **0.845** | 0.680  | 0.753 | 26,615 |

**Takeaway:** 49.5% is **~2.5× random**. The U-shape across bins is informative: the model is very good at separating cheap (bin 0) and expensive (bin 4) properties from the rest, and weakest in the dense middle band (bins 1–2). Most misclassifications are **off-by-one bin** — see Figure 5 — confirming the regression error pattern translates to "close but not exact" on the bin axis. This is a **derived proxy task** that translates regression performance into a single accuracy percentage; the primary signal remains §3.1.

---

## 4. Discussion

Suggested paragraphs:

1. **Interpretation:** Mainline HistGBR gives strong uplift vs naive; errors remain large in GBP because prices span a wide range—optional future work: log or percentage metrics.
2. **Vendor estimates:** Assisted track and single-column benchmarks show that vendor channels carry signal; mainline excludes them to avoid leakage and deployment dependency—align conclusions with course ethics / brief.
3. **Segments:** Failed segment gate highlights unequal error across price bands and types; propose mitigation (features, stratified objectives) or scoped deployment.
4. **Reproducibility:** Fixed seeds where applicable; document Python/sklearn versions (`requirements.txt`).

---

## 5. Conclusion

- The best mainline model on the calendar-fair test holdout is **HistGBR**: test RMSE **393,489 GBP**, test MAE **216,083 GBP**, R² **0.72**, a **57.6%** improvement in test RMSE over the naive median baseline.
- Walk-forward (Table 3): RandomForest mean RMSE **347,500 GBP** (CV 24.4%); MLP mean RMSE **356,658 GBP** (CV **15.1%**) — the NN is the *more stable* family across calendar shifts even though boosting wins the single-shot test.
- The discrete proxy view (Table 6) gives an **accuracy of 49.5%** across 5 quantile price bins (≈2.5× random), with most errors off-by-one bin.
- **Limitations.** The expensive-property tail (> 1.5 M GBP) drives the largest residuals (Figure 7) and trips the segment gate in §3.6; vendor estimates carry signal but are kept out of the mainline track to avoid leakage and deployment dependency.

---

## References

- Breiman, L. (2001). Random Forests. *Machine Learning, 45*(1), 5–32.
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics, 29*(5), 1189–1232.
- Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on tabular data? In *NeurIPS Datasets and Benchmarks*.
- Hornik, K., Stinchcombe, M., & White, H. (1989). Multilayer feedforward networks are universal approximators. *Neural Networks, 2*(5), 359–366.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- Shwartz-Ziv, R., & Armon, A. (2022). Tabular data: Deep learning is not all you need. *Information Fusion, 81*, 84–90.

Align bibliography style with course requirements (APA, IEEE, or Harvard).

---

## Appendix (optional)

- Commands to reproduce: [`README.md`](../README.md).
- Full methodology: [`methodology.md`](../methodology.md).
- Feature lists and code paths: `src/london_pipeline.py`.
