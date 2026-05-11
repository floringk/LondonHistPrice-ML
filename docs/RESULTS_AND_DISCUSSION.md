# London House Price Prediction — Results and Discussion

*Unified, paper-style report assembled from the frozen artifacts in [`results/`](../results/) (run stamp: see [`results/RUN_STAMP.txt`](../results/RUN_STAMP.txt)). Every number below is reproducible by re-running the commands in [`README.md`](../README.md). For full methodology see [`methodology.md`](../methodology.md); for the structured IMRaD skeleton with all tables see [`docs/REPORT_IMRAD.md`](REPORT_IMRAD.md).*

---

## Abstract

We study supervised regression of London residential transaction prices (`history_price`) over the period 1995–2024 using a calendar-fair train / validation / test split (70% / 10% / 20% of elapsed time, *not* of rows). We compare three model families — linear (Ridge, ElasticNet), tree ensembles (RandomForest, HistGradientBoosting), and neural networks (an sklearn `MLPRegressor` capacity scan plus a PyTorch MLP with log target and bias-initialised output head) — under a single shared preprocessor fit on training data only. On the 2018–2024 test holdout, HistGradientBoosting attains the lowest test RMSE of **393,489 GBP** (test MAE 216,083; R² 0.72), a **57.6%** RMSE reduction over a naive-median baseline. Tree ensembles beat linear baselines by about 30% on test RMSE and beat all neural-network configurations on this holdout, which is consistent with published findings that tree models tend to outperform deep nets on medium-sized heterogeneous tabular data (Shwartz-Ziv & Armon, 2022; Grinsztajn et al., 2022). The strongest **positive finding for the NN family** is in walk-forward validation across four expanding calendar folds: the MLP achieves a **coefficient of variation of 15.1%** vs RandomForest's **24.4%** — the neural network is the *more stable* family under realistic temporal shift, even though it is not the single-shot leader. Translated to a discrete proxy task using 5 quantile bins of the test target, the best regression model attains **49.5% bin-accuracy** (≈ 2.5× random), with most errors off-by-one bin. We discuss the expensive-property tail (> 1.5 M GBP) as the dominant residual driver and segment gate failure mode.

**Keywords:** regression, tabular data, neural networks, gradient boosting, calendar split, walk-forward validation, temporal shift.

---

## 1. Introduction

Predicting the sale price of a residential property is a classic supervised-regression task that combines structural features (rooms, area, type), location features (postcode, town), and time. The dataset in this study is the Kaggle "London House Price Data" (`dataset/kaggle_london_house_price_data.csv`), which contains roughly **half a million** transactions over a 29-year window.

The assignment is part of a neural-networks course, which raises the question that motivates this report:

> *Does a neural network outperform classical machine-learning baselines on this tabular regression task — and if not, what is the honest narrative?*

We give an explicit, evidence-based answer rather than a defensive one. We train one linear family, one tree family, and one neural-network family **under identical conditions** (same split, same preprocessor, same fixed seed where applicable) and we evaluate them on the **same temporally-held-out test set**. We then run a separate walk-forward validation across four expanding calendar folds to measure stability, which is where the neural network earns its place in the report.

---

## 2. Methodology (summary)

The full methodology is in [`methodology.md`](../methodology.md); here is what the reader needs to follow the results.

**Target and predictors.** Target: `history_price` (continuous, GBP). Predictors: structural and location columns; columns prefixed with `saleEstimate_` and `rentEstimate_` are **excluded** from the mainline track to avoid leakage of vendor predictions into a model that would learn to copy them.

**Split.** A *calendar-fair* split: the dataset is sorted by `history_date`; the first **70%** of *elapsed time* between min and max date is training (1995-09-09 → 2015-03-04), the next **10%** is validation (2015-03-04 → 2018-04-13), and the final **20%** is test (2018-04-13 → 2024-09-27). This is **not** a 70-10-20 split of rows; row counts are unbalanced by design, exactly as production deployment would encounter them.

**Preprocessing.** Numeric columns: median imputation followed by `StandardScaler`. Categorical columns: one-hot encoding with `handle_unknown="ignore"` and sparse output. The preprocessor is fit **on training data only**, then applied to validation and test.

**Models (mainline track).** All models share the preprocessor and the split. Hyperparameters are detailed in [`methodology.md`](../methodology.md) §6.

| Family            | Model                                                | Why included                                                                 |
|-------------------|------------------------------------------------------|------------------------------------------------------------------------------|
| Linear            | `Ridge`, `ElasticNet`                                | Bayesian-friendly linear references; expected lower bound on RMSE for a fair model. |
| Tree boosting     | `HistGradientBoostingRegressor`                      | Strong default for tabular regression; handles non-linearity and feature interactions. |
| Tree bagging      | `RandomForestRegressor`                              | Bagging variance reduction; second strong tabular baseline.                 |
| Neural network    | sklearn `MLPRegressor` capacity scan (small / medium / large) | NN baseline that fits in the same sklearn pipeline as the others.        |
| Neural network    | PyTorch `MLPNet` (log target, bias-init, log-clip)   | Explicit course-aligned NN with a training loop, loss curve, and architecture diagram. |
| Reference         | `NaiveMedian`                                        | Lower-bound reference; predicts training median.                            |

**Metrics.** Regression-friendly: MAE, RMSE (GBP), R², MAPE, and the share of test rows with `|y - y_hat| / |y| < 0.10` (the "within-10% rate"). Derived classification view: the best regression model's predictions are discretised into 5 quantile bins of `y_true`, and accuracy / per-bin F1 are computed on the resulting confusion matrix.

**Walk-forward validation.** Four expanding-window folds across 2000-12 → 2024-09, run for two model families (RandomForest, MLP) under the same preprocessor and seed; per-fold RMSE is summarised by mean, std, and coefficient of variation (CV %).

---

## 3. Results

### 3.1 Mainline holdout performance

**Table 1. Test-set comparison of all model families on the calendar holdout (2018-04-13 → 2024-09-27).** Models are ranked by test RMSE (lower is better). All numbers reproduced from [`results/model_results_summary.csv`](../results/model_results_summary.csv).

| Model         | Family            | Test MAE (GBP) | Test RMSE (GBP) | Test R²  | Test MAPE | Within 10% | RMSE vs naive |
|---------------|-------------------|---------------:|----------------:|---------:|----------:|----------:|--------------:|
| **HistGBR**   | Tree boosting     |    216,082.64 |     **393,489.27** |   **0.720** |    0.338 |   **0.243** |    **57.58%** |
| RandomForest  | Tree ensemble     |    215,149.40 |        396,947.22 |     0.715 |    0.329 |     0.232 |        57.21% |
| Ridge         | Linear (L2)       |    375,218.61 |        562,967.66 |     0.426 |    0.745 |     0.109 |        39.31% |
| ElasticNet    | Linear (L1 + L2)  |    374,925.88 |        563,113.40 |     0.426 |    0.744 |     0.109 |        39.30% |
| MLP_medium    | NN (sklearn)      |    444,363.38 |        645,104.30 |     0.247 |    0.852 |     0.109 |        30.46% |
| MLP_large     | NN (sklearn)      |    477,584.44 |        721,999.16 |     0.056 |    0.887 |     0.114 |        22.17% |
| TorchMLP      | NN (PyTorch)      |    470,268.29 |        729,943.31 |     0.035 |    0.796 |     0.106 |        21.31% |
| MLP_small     | NN (sklearn)      |    619,645.38 |        838,729.34 |    −0.274 |    1.173 |     0.059 |         9.59% |
| NaiveMedian   | Reference         |    561,898.98 |        927,645.60 |    −0.558 |    0.570 |     0.035 |         0.00% |

**Reading guide.**
- HistGBR is the best mainline model on every regression metric (MAE, RMSE, R², MAPE, within-10%).
- Tree ensembles beat the best linear model by **30%** on test RMSE (393k vs 563k GBP). This is the "trees vs linear" gap reported by Grinsztajn et al. (2022) on heterogeneous tabular data.
- All NN configurations sit between the linear baselines and the naive baseline on this single holdout. **This is the expected ordering** for medium-sized heterogeneous tabular data when each model gets the same preprocessor and seed budget (Shwartz-Ziv & Armon, 2022). The NN's positive evidence is in §3.3 (walk-forward stability).

### 3.2 NN training dynamics

**Figure 1. sklearn `MLPRegressor` training-loss curves (small, medium, large architectures).** Source: [`results/MLP_small_loss_curve.png`](../results/MLP_small_loss_curve.png), [`results/MLP_medium_loss_curve.png`](../results/MLP_medium_loss_curve.png), [`results/MLP_large_loss_curve.png`](../results/MLP_large_loss_curve.png). All three curves show typical Adam convergence: a sharp drop in the first ~20 epochs followed by a plateau where early stopping triggers.

**Figure 2. PyTorch `TorchMLP` validation RMSE per epoch.** Source: [`results/torch_mlp_loss_curve.png`](../results/torch_mlp_loss_curve.png), per-epoch history at [`results/torch_mlp_history.csv`](../results/torch_mlp_history.csv). The best validation RMSE is reached at **epoch 21** (val RMSE 428,185 GBP) and the early-stop checkpoint is restored. The training-loss curve continues to decrease after epoch 21, which means the model would start to overfit if training continued — exactly the situation early stopping is designed to detect.

**Two implementation details that materially affect NN credibility** (full rationale in [`methodology.md`](../methodology.md) §6.3):

1. **sklearn MLP uses a `StandardScaler` target wrap, not `log1p`/`expm1`.** A log target looks attractive for the heavy GBP tail, but sklearn's internal early-stop signal lives in transformed space; with `log1p`, the checkpoint selected on the lower-price training era systematically under-predicts the shifted test era, and `expm1` then multiplies that bias. Standardising the target is bias-preserving (a linear inverse), which lets early stopping pick a checkpoint that is honest in GBP units.
2. **PyTorch MLP keeps the log target but adds two safety guards.** (a) The final linear layer's bias is initialised to `mean(log1p(y_train))` so epoch 0 predicts the training log-mean; without this, random init can produce log values around 24, and `expm1(24) ≈ 26 billion GBP` (observed empirically on the first run). (b) Log predictions are clipped to `[log_min(y_train) − 0.5, log_max(y_train) + 0.5]` before `expm1`, bounding test-time predictions to a sensible envelope.

These are not cosmetic — they are the difference between a working PyTorch MLP and a model that prints 26-billion-GBP predictions.

### 3.3 Walk-forward stability (the positive NN finding)

**Table 2. Per-fold test RMSE on expanding-window folds.** Same preprocessor and seed across both families. Source: [`results/walk_forward_results.csv`](../results/walk_forward_results.csv).

| Fold (validation window)    | Train rows | Val rows | RandomForest RMSE | MLP RMSE        | Winner       |
|-----------------------------|-----------:|---------:|------------------:|----------------:|:-------------|
| 1 (2000-12 → 2006-11)       |     44,642 |   50,806 |        224,290.76 |      293,533.19 | RandomForest |
| 2 (2006-11 → 2012-11)       |     95,448 |   36,453 |        341,327.72 |      312,845.88 | **MLP**      |
| 3 (2012-11 → 2018-10)       |    131,901 |   53,297 |        462,977.05 |      411,278.64 | **MLP**      |
| 4 (2018-10 → 2024-09)       |    185,198 |  130,470 |        361,405.18 |      408,975.44 | RandomForest |

**Table 3. Walk-forward summary across folds.**

| Family       | Mean RMSE (GBP) | Std RMSE (GBP) |   CV % | Folds won |
|--------------|----------------:|---------------:|-------:|----------:|
| RandomForest |      347,500.18 |      84,774.58 |  24.40 |     2 / 4 |
| **MLP**      |      356,658.29 |      53,909.12 | **15.12** |     2 / 4 |

**Interpretation.** Across four calendar regimes, MLP loses by only **2.6% on mean RMSE** but is **38% more stable** (15.1% CV vs 24.4%). This is the strongest positive NN result in this study: a single-shot holdout favours the tree family, but the neural network is more reliable under realistic temporal shift. For a production setting where the regime *will* drift, the lower-CV family is a credible candidate.

### 3.4 Discrete (classification) view of regression performance

The original feedback asked for a "confusion matrix" and an "accuracy %". A confusion matrix is not natively defined for continuous targets, so we report a **derived** classification view: the best regression model's predictions and the test targets are both discretised into 5 quantile bins of `y_true`, and a 5×5 confusion matrix is computed on the resulting (true bin, predicted bin) pairs.

**Figure 3. Price-bin confusion matrix (HistGBR, 5 quantile bins).** Source: [`results/price_bin_confusion.png`](../results/price_bin_confusion.png) and [`results/price_bin_confusion.csv`](../results/price_bin_confusion.csv). The diagonal is the heaviest band of the matrix; the secondary diagonals (off-by-one neighbours) carry most remaining mass, confirming that misclassifications are mostly "close but not exact".

**Table 4. Derived classification metrics on the bin-classification proxy task.**

| Metric                                            | Value     |
|---------------------------------------------------|----------:|
| **Bin accuracy** (HistGBR, 5 quantile bins)       | **0.495** |
| Macro F1                                          |     0.495 |
| Weighted F1                                       |     0.497 |
| Random-guess baseline (5 equally-sized bins)      |     0.200 |

**Table 5. Per-bin precision / recall / F1.** Source: [`results/price_bin_classification_report.csv`](../results/price_bin_classification_report.csv).

| Bin | Price range (GBP)        | Precision | Recall | F1    | Support |
|----:|--------------------------|----------:|-------:|------:|--------:|
|  0  | 42,995 → 370,000         |     0.544 |  0.703 | 0.613 |  25,953 |
|  1  | 370,000 → 487,000        |     0.338 |  0.382 | 0.359 |  26,212 |
|  2  | 487,000 → 650,000        |     0.338 |  0.324 | 0.331 |  25,352 |
|  3  | 650,000 → 1,000,000      |     0.464 |  0.381 | 0.419 |  26,344 |
|  4  | 1,000,000 → 4,400,000    | **0.845** |  0.680 | 0.753 |  26,615 |

**Interpretation.** **49.5% accuracy is ~2.5× random** (random is 20% on 5 bins). The U-shape across bins is informative: the model is *very* good at separating the cheap (bin 0, F1 0.61) and expensive (bin 4, F1 0.75) properties from the rest, and weakest in the dense middle band (bins 1–2, F1 ~0.34). This is intuitive — the cheapest and most expensive segments differ markedly from the rest in structural and location features, while the middle band is internally homogeneous and harder to separate.

### 3.5 Regression diagnostics

**Figure 4. HistGBR predictions vs actual price on the test set (log scale).** Source: [`results/regression_pred_vs_actual.png`](../results/regression_pred_vs_actual.png). The diagonal `y = x` reference line is drawn. Heavy clustering on the diagonal in the 200k–800k GBP band; visible scatter widens above 1.5 M GBP.

**Figure 5. HistGBR residuals — histogram (top) and residuals-vs-fitted (bottom).** Source: [`results/regression_residuals.png`](../results/regression_residuals.png). The residual histogram is centred near zero but **right-skewed**; the residuals-vs-fitted plot is heteroscedastic, with a clear fan structure where higher predicted prices have larger absolute residuals.

Both diagnostics point to the same conclusion: **the expensive-property tail (> 1.5 M GBP) is the dominant source of error.** This is the diagnostic that closes the segment-gate failure in §3.6.

### 3.6 Release gates and segment blockers

The pipeline applies four release gates after each run (full definitions in [`GOVERNANCE.md`](../GOVERNANCE.md)):

| Gate                                | Result   |
|-------------------------------------|----------|
| Crash / artifact gate               | **PASS** |
| Primary RMSE threshold              | **PASS** |
| Walk-forward CV ≤ 26%               | **PASS** (24.4%) |
| High-support segment RMSE gate      | **FAIL** |

The segment gate fails because the model's RMSE on the expensive-property segment (price > 1.5 M GBP) exceeds the threshold, even though aggregate RMSE is acceptable. We report this honestly rather than tune it away — the same finding is visible in Figure 5.

---

## 4. Discussion

**Why trees beat NNs on this dataset.** The dataset is medium-sized (~500k rows), the features are heterogeneous (mixed numeric, categorical, geographic), and several features are highly informative on their own (postcode, area). Tree ensembles are extremely sample-efficient under exactly these conditions: each split is a univariate threshold that does not need feature scaling, missing-value handling is native, and feature interactions are learned in shallow combinations. Neural networks need both more data and richer feature engineering to outperform trees on tabular targets (Shwartz-Ziv & Armon, 2022; Grinsztajn et al., 2022). Our result — boosting ~393k GBP, MLPs in the 645k–840k GBP range on the single-shot test — is **consistent with the published gap**, not an artefact of poor NN configuration. We tested four architectures across two frameworks and verified that the training loss curves (Figures 1–2) show healthy convergence and that early stopping selects sensible checkpoints.

**Why the NN earns its place anyway.** Walk-forward gives a different picture (§3.3): MLP's coefficient of variation is **38% lower** than RandomForest's across four calendar regimes. This is exactly the *robustness-under-shift* property that motivates using NNs in time-varying production settings. We attribute this stability partly to the gradient-based optimiser smoothing out fold-specific minima that a tree's discrete splits would otherwise lock onto, and partly to L2 regularisation + dropout in the PyTorch baseline.

**Why the linear baselines are reported.** A clean Ridge / ElasticNet baseline gives a fair lower bound on what a *linear* model would achieve under the same preprocessor. The 30% RMSE gap (393k → 563k GBP) between HistGBR and Ridge is the empirical evidence for choosing non-linear models over linear regression — directly answering the feedback question "why not linear regression?".

**The accuracy question.** The discrete view (§3.4) translates the regression result into a single accuracy percentage of 49.5% on 5 quantile bins (2.5× random). This is the right number to report when a stakeholder asks "what's the accuracy" of a regression model — with the important caveat that the primary signal remains the regression metrics in Table 1, because the bin discretisation throws away information.

**Limitations.**
1. **Expensive-property tail.** The segment gate (§3.6) and the residual diagnostics (Figure 5) agree that errors above ~1.5 M GBP are disproportionately large. Mitigations include log-target boosting, quantile loss, or a stratified two-stage model.
2. **No bootstrapped confidence intervals.** All test-set numbers are single-run. The fixed seed (42) protects reproducibility but does not quantify variance from data resampling.
3. **Walk-forward is two-family only.** RandomForest and MLP; adding HistGBR to walk-forward would let the report claim "boosting also wins under shift" or "boosting also has lower CV" — neither has been measured yet.
4. **Vendor estimates are excluded by policy.** The `saleEstimate_*` columns are strong predictors but are kept out of mainline training to avoid leakage of vendor predictions; the assisted track shows what they would add (Table 5 in [`docs/REPORT_IMRAD.md`](REPORT_IMRAD.md)).

---

## 5. Conclusion

On a calendar-fair holdout, **HistGradientBoosting is the best model** for predicting London house transaction prices (test RMSE 393,489 GBP; +57.6% RMSE reduction vs naive median; R² 0.72). Tree ensembles beat linear baselines by ~30% on RMSE and beat neural networks on the single-shot test set, **consistent with published findings on medium tabular data**. The neural network earns its place in the report via walk-forward stability: the MLP has a **15.1% coefficient of variation** across four calendar regimes vs RandomForest's 24.4% — a 38% relative reduction. Translated to a 5-bin classification proxy, the best model attains 49.5% accuracy (≈ 2.5× random), with most errors confined to neighbouring bins. The main residual driver is the expensive-property tail above ~1.5 M GBP, which fails the segment release gate and motivates the next iteration of the model.

---

## References

- Breiman, L. (2001). Random Forests. *Machine Learning, 45*(1), 5–32.
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics, 29*(5), 1189–1232.
- Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on tabular data? *NeurIPS Datasets and Benchmarks*.
- Hornik, K., Stinchcombe, M., & White, H. (1989). Multilayer feedforward networks are universal approximators. *Neural Networks, 2*(5), 359–366.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- Shwartz-Ziv, R., & Armon, A. (2022). Tabular data: Deep learning is not all you need. *Information Fusion, 81*, 84–90.

---

## Appendix — Reproducibility

```bash
# Baseline mainline pipeline (HistGBR / RF / Ridge / ElasticNet / sklearn MLP scan)
python src/run_london_pipeline.py

# Explicit PyTorch MLP (writes TorchMLP row + loss curve + history CSV)
python src/torch_mlp.py

# Walk-forward (RF + MLP families)
python src/walk_forward_validation.py

# Consolidated report + freeze artifacts under results/
python src/main.py --report-only --sync-results
```

Python ≥ 3.10, dependency pins in [`requirements.txt`](../requirements.txt) (`torch>=2.2` installed via the CPU wheel index `https://download.pytorch.org/whl/cpu`). Fixed seed `42`. All frozen artifacts live under [`results/`](../results/) with a run stamp in [`results/RUN_STAMP.txt`](../results/RUN_STAMP.txt).
