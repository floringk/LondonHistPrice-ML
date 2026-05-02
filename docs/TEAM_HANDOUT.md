# Team handout — literature, slides, report

**Frozen snapshot:** cite numbers from [`results/`](../results/) only (especially [`results/run_report.md`](../results/run_report.md)). If someone reruns the pipeline, numbers in `data/` may drift; recommit `results/` after `python "src/main.py" --report-only --sync-results` before locking the paper/slides.

---

## Who reads what

| You work on… | Start here | Also use |
|--------------|------------|----------|
| **Literature / related work** | [`task.md`](../task.md) (literature table + motivation) | Course papers; [`methodology.md`](../methodology.md) §2 principles |
| **Methods (how we built it)** | [`methodology.md`](../methodology.md) | [`README.md`](../README.md) commands |
| **Results / numbers** | [`results/run_report.md`](../results/run_report.md) | CSVs below |
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
| Walk-forward folds | [`results/walk_forward_results.csv`](../results/walk_forward_results.csv) |

---

## Vocabulary (use consistently)

- **Mainline:** HistGBR / RF trained **without** `saleEstimate_*` / `rentEstimate_*` as inputs — **leakage-safe** deployment story.
- **Assisted:** separate models **with** vendor estimate columns — better RMSE here, but **depends on vendor data**; not the same as mainline for “fair” comparison.
- **Benchmark:** comparing test prices to a single `saleEstimate_*` column — **not** a trained model; comparator only.

---

## Slide bullets (copy/adapt)

1. **Problem:** Predict London transaction price (`history_price`) from property features; evaluate with a **time-based** train/val/test split (not random rows).
2. **Main result:** Best mainline model **HistGBR** — test RMSE **~393.5k**, test MAE **~216.1k** GBP; **~57.6%** RMSE reduction vs predicting the training median everywhere.
3. **Baselines:** Naive median vs **RandomForest** vs **HistGBR** on the same holdout (see table in report template).
4. **Robustness:** Walk-forward **4** folds; mean RMSE **~347.5k**, CV **~24.4%** (stability check, not identical to HistGBR config each fold).
5. **Context:** Single-column vendor benchmark (`saleEstimate_lowerPrice`) RMSE **~350.2k** — strong comparator; **not** mixed into mainline features by design.
6. **Dual track:** Assisted HistGBR test RMSE **~304.4k** — discuss **only** if your brief allows “vendor-assisted” pricing; otherwise emphasize **mainline** for governance.
7. **Limitations:** Segment gate **FAIL** on some high-support slices (expensive tail, some property types); metrics in **currency units**; assisted track needs vendor availability.

---

## One sentence each author can reuse

- **Literature:** We follow time-aware evaluation and train-only preprocessing to limit leakage, consistent with standard practice for temporal tabular forecasting.
- **Methods:** We use a calendar-fair split on `history_date`, median/scaling + one-hot preprocessing fit on train only, and compare boosting vs forest vs a naive median.
- **Results:** On the held-out period, tuned HistGBR achieves the best mainline test RMSE among compared models; walk-forward analysis shows moderate fold-to-fold variability.
- **Discussion:** Vendor-assisted models achieve lower error but introduce dependency on external estimates; mainline models remain the basis for leakage-sensitive claims; segment errors remain large in expensive market segments.

---

## Full report outline (IMRaD + tables)

See [`REPORT_IMRAD.md`](REPORT_IMRAD.md) in this folder.
