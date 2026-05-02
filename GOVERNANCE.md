# Governance and rollout policy

This repository runs two **separate** prediction tracks. This document is the operational counterpart to [`methodology.md`](methodology.md) (methods) and [`README.md`](README.md) (commands).

## 1. Leakage-safe mainline (locked)

- **Primary reporting model:** tuned **`HistGradientBoostingRegressor`** on the calendar holdout (see [`results/model_results_summary.csv`](results/model_results_summary.csv)).
- **Feature policy:** mainline training **must not** use columns matching `saleEstimate_*` or `rentEstimate_*`. That exclusion is enforced in [`src/london_pipeline.py`](src/london_pipeline.py) (`_feature_lists`).
- **Claims:** any statement that the model is “free of vendor-estimate leakage” applies **only** to the mainline path.

Do **not** merge vendor estimate columns into mainline features without an explicit architecture review and a new methodology version.

## 2. Dual-track deployment

| Track | Script entrypoint | Role |
|-------|-------------------|------|
| Mainline | [`src/run_london_pipeline.py`](src/run_london_pipeline.py) | Default for regulated or leakage-sensitive use; metrics in `model_results_summary.csv`. |
| Assisted | [`src/assisted_track.py`](src/assisted_track.py) | Uses numeric `saleEstimate_*` (and related) as predictors—**only** where vendor data is acceptable at inference time. |

After each full evaluation, [`src/main.py`](src/main.py) writes [`data/model_decision_summary.csv`](data/model_decision_summary.csv) (gitignored working copy) with:

- `governance_recommendation`: e.g. `deploy_assisted`, `deploy_mainline`, or `pilot_assisted_only`.
- Use this as a **governance hint**, not an automatic production deploy. Deploy assisted models only on a **controlled** path (separate endpoint, feature flag, or batch job) with explicit product approval.

## 3. Segment blockers before broad rollout

The release report includes a **high-support segment gate** (see [`src/main.py`](src/main.py)). When it fails, failing segments are listed in `segment_blocker_actions.csv`.

**Before org-wide rollout:**

1. Review [`rollout/SEGMENT_BLOCKER_RESOLUTION.md`](rollout/SEGMENT_BLOCKER_RESOLUTION.md) and complete actions per segment (model mitigation vs business sign-off).
2. Re-run the pipeline and regenerate the consolidated report until segment policy is satisfied **or** documented risk acceptance is recorded for remaining gaps.

## 4. Reproducibility and frozen bundles

1. Run the full evaluation (see [README.md](README.md)) so `data/` contains fresh artifacts.
2. Regenerate the consolidated report:  
   `python "src/main.py" --report-only --sync-results`  
   The `--sync-results` flag copies small summary files from `data/` into [`results/`](results/) and updates [`results/RUN_STAMP.txt`](results/RUN_STAMP.txt) so teammates can commit a **frozen** snapshot.

Commit **`results/`** (not `data/`) for milestone numbers.

## 5. Publishing a teammate bundle

Share (without committing raw CSV data):

- [`results/`](results/) — frozen summaries and `run_report.md`
- [`methodology.md`](methodology.md), [`GOVERNANCE.md`](GOVERNANCE.md), [`README.md`](README.md)
- [`src/`](src/) as needed for audit

Omit `dataset/*.csv` from shares unless teammates have a data-handling agreement.

## 6. Making the GitHub repository private

GitHub does not allow deleting a repository from this environment without your credentials. Use one of:

**A. Change visibility (keeps history and URLs)**

1. On GitHub: **Settings** → **General** → **Danger zone** → **Change repository visibility** → **Private**.

**B. Delete and recreate (new empty repo)**

1. **Settings** → **Danger zone** → **Delete this repository** (confirm with repo name).
2. Create a **new private** repository under your account or org.
3. Locally:  
   `git remote set-url origin https://github.com/<user>/<new-private-repo>.git`  
   `git push -u origin main`

Use SSH or a personal access token as required by your org.
