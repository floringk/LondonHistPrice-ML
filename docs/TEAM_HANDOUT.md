# Team handout — literature, slides, report

**Frozen snapshot:** cite numbers from [`results/`](../results/) only (especially [`results/run_report.md`](../results/run_report.md)). If someone reruns the pipeline, numbers in `data/` may drift; recommit `results/` after `python "src/main.py" --report-only --sync-results` before locking the paper/slides.

---

## Who reads what

| You work on… | Start here | Also use |
|--------------|------------|----------|
| **Literature / related work** | [`task.md`](../task.md) (literature table + motivation) | Course papers; [`methodology.md`](../methodology.md) §2 principles |
| **Methods (how we built it)** | [`methodology.md`](../methodology.md) | [`README.md`](../README.md) commands |
| **Results / numbers** | [`results/run_report.md`](../results/run_report.md) | CSVs below |
| **Paper / unified report (EN)** | [`docs/RESULTS_AND_DISCUSSION.md`](RESULTS_AND_DISCUSSION.md) — abstract + all tables + captioned figures | [`docs/REPORT_IMRAD.md`](REPORT_IMRAD.md) for IMRaD layout |
| **Feedback replies (RO)** | [`docs/Dedicatie_Radu.md`](Dedicatie_Radu.md) (long) | [`docs/FEEDBACK_REPLIES_SCURT_RO.md`](FEEDBACK_REPLIES_SCURT_RO.md) (short) |
| **Two-model story (main vs assisted)** | [`GOVERNANCE.md`](../GOVERNANCE.md) | [`results/model_decision_summary.csv`](../results/model_decision_summary.csv) |
| **Honest limitations** | [`results/run_report.md`](../results/run_report.md) gates + blockers | [`rollout/SEGMENT_BLOCKER_RESOLUTION.md`](../rollout/SEGMENT_BLOCKER_RESOLUTION.md) |

---

## Where the numbers live (single source of truth)

| What | File |
|------|------|
| Headline metrics, gates, blockers | [`results/run_report.md`](../results/run_report.md) |
| Mainline vs naive vs RF | [`results/model_results_summary.csv`](../results/model_results_summary.csv) |
| Mainline vs assisted vs benchmark | [`results/track_comparison_summary.csv`](../results/track_comparison_summary.csv) |
| Governance keywords | [`results/model_decision_summary.csv`](../results/model_decision_summary.csv) |
| Raw benchmark columns | [`results/external_estimate_benchmark.csv`](../results/external_estimate_benchmark.csv) |
| Walk-forward folds (per `model_family`) | [`results/walk_forward_results.csv`](../results/walk_forward_results.csv) |
| Price-bin classification (accuracy/F1) | [`results/price_bin_classification_summary.csv`](../results/price_bin_classification_summary.csv), [`results/price_bin_classification_report.csv`](../results/price_bin_classification_report.csv) |
| NN training history | [`results/torch_mlp_history.csv`](../results/torch_mlp_history.csv), `results/*_loss_curve.png` |

---

## Vocabulary (use consistently)

- **Mainline:** HistGBR / RF / linear / NN trained **without** `saleEstimate_*` / `rentEstimate_*` as inputs — **leakage-safe** deployment story.
- **Assisted:** separate models **with** vendor estimate columns — better RMSE here, but **depends on vendor data**; not the same as mainline for “fair” comparison.
- **Benchmark:** comparing test prices to a single `saleEstimate_*` column — **not** a trained model; comparator only.
- **NN baselines:** `MLP_small/medium/large` (sklearn `MLPRegressor` capacity scan, log-target) and `TorchMLP` ([`src/torch_mlp.py`](../src/torch_mlp.py)) — same split, same preprocessor, course-alignment baselines.
- **Linear baselines:** `Ridge`, `ElasticNet` — same sparse preprocessor as the trees.
- **Price-bin classification:** derived 5-quantile task on `y_true`; gives an *accuracy %* and *macro F1* on top of the regression metrics — not a replacement for RMSE/MAE.

---

## Slide bullets (copy/paste, current numbers from this snapshot)

1. **Problem.** Predict London transaction price (`history_price`, GBP) from 18 property features; evaluate with a **time-based** train/val/test split — train 1995-01 → 2015-10, val 2015-10 → 2018-10, test **2018-10 → 2024-09** (≈130k test rows).
2. **Main result.** Best mainline model is **HistGBR** — test **RMSE 393,489**, test **MAE 216,083**, **R² 0.720**, **57.6%** RMSE reduction vs predicting the training median everywhere. Source: [`results/model_results_summary.csv`](../results/model_results_summary.csv).
3. **Three families on the same split.** **Linear** (Ridge **562,968**, ElasticNet **563,113**), **trees** (RandomForest **396,947**, HistGBR **393,489**), and **NN** (best of sklearn scan: MLP_medium **645,104**; PyTorch TorchMLP **729,943**). Same preprocessor for all. Trees beat linear by **~30%** RMSE; NN underperforms both on this single holdout (calendar shift).
4. **Accuracy on a 5-bin classification view.** Discretise `y_true` into 5 quantile bins (best model: HistGBR): **accuracy 49.51%**, **macro F1 0.495**, weighted F1 0.497. Source: [`results/price_bin_classification_summary.csv`](../results/price_bin_classification_summary.csv).
5. **Walk-forward stability (4 folds).** RandomForest mean RMSE **347,500** (CV **24.4%**), MLP mean RMSE **356,658** (CV **15.1%**). **MLP is more stable** in CV than RF and beats RF in 2 of 4 folds (2006-2012 and 2012-2018). Source: [`results/walk_forward_results.csv`](../results/walk_forward_results.csv).
6. **NN methodology evidence.** sklearn `MLPRegressor` capacity scan (3 architectures + standardized target wrap) + PyTorch `MLPNet` with explicit training loop, dropout(0.2), AdamW, SmoothL1Loss, log-target with safe clip + final-layer bias initialised to `log_mean(y_train)`, early stopping on val RMSE in GBP space. Loss curves at `results/*_loss_curve.png`, PyTorch per-epoch history at [`results/torch_mlp_history.csv`](../results/torch_mlp_history.csv).
7. **Context.** Vendor benchmark (`saleEstimate_lowerPrice`) hits RMSE **350,168** as a single-column comparator. The **assisted** track (HistGBR + vendor features) gets to **304,436**, but depends on vendor data and is **not** the mainline narrative.
8. **Honest NN takeaway.** On this calendar holdout NNs lose to boosting; same finding reported in Shwartz-Ziv & Armon (2022) and Grinsztajn et al. (2022) on tabular benchmarks. NNs are included for **course alignment** and a **fair comparison**, not as production candidates. On walk-forward (shorter shift windows), the MLP becomes competitive with — and more stable than — RandomForest.
9. **Limitations.** Segment gate **FAIL** on high-support tail slices (`Detached House`, `Semi-Detached Property`, `price_band > 1M GBP`); regression metrics in raw GBP; calendar shift from 1995 to 2024 is large (Mean train price << mean test price).

---

## One sentence each author can reuse

- **Literature:** We follow time-aware evaluation and train-only preprocessing to limit leakage, consistent with standard practice for temporal tabular forecasting; on the model-family question, we cite Shwartz-Ziv & Armon (2022) and Grinsztajn et al. (2022) for the finding that tree ensembles still outperform deep nets on heterogeneous tabular data.
- **Methods:** We use a calendar-fair 70/10/20 split on `history_date`, median/scaling + one-hot preprocessing fit on train only, and compare four families on the same matrix: a naive median baseline, linear regression (Ridge + ElasticNet), tree ensembles (RandomForest + HistGBR), and neural networks (sklearn `MLPRegressor` capacity scan + a PyTorch `MLPNet` with explicit training loop).
- **Results:** On the held-out period, tuned HistGBR achieves the best mainline test RMSE (393,489 GBP, R² 0.720, within-10% rate 24.3%, price-bin accuracy 49.51% on 5 quantile bins); walk-forward analysis (4 calendar folds, two model families) gives RandomForest CV 24.4% vs MLP CV 15.1%, showing the NN is more stable across shorter windows even though boosting wins the single holdout.
- **Discussion:** Neural baselines underperform boosting on the calendar holdout because of distribution shift (test era 2018-2024 has substantially higher prices than the train era); vendor-assisted models achieve lower error but introduce dependency on external estimates; mainline models remain the basis for leakage-sensitive claims; segment errors remain large in expensive market segments (`Detached`, `Semi-Detached`, price band above 1M GBP).

---

## Full report outline (IMRaD + tables)

See [`REPORT_IMRAD.md`](REPORT_IMRAD.md) in this folder.

---

## Feedback Q&A (Romanian, short)

For copy/paste replies to reviewers (four numbered answers), see [`FEEDBACK_REPLIES_SCURT_RO.md`](FEEDBACK_REPLIES_SCURT_RO.md). Extended narrative: [`Dedicatie_Radu.md`](Dedicatie_Radu.md).

