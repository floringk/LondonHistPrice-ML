# Segment blocker resolution (pre–broad rollout)

Source of truth for **automated** blocker rows: `data/segment_blocker_actions.csv` after [`src/main.py`](../src/main.py). The committed mirror for teammates is [`results/segment_blocker_actions.csv`](../results/segment_blocker_actions.csv).

Complete the checklist below before treating the release as cleared for broad rollout. Closed items should be reflected in meeting notes or ticket IDs.

## Current blockers (snapshot)

| Segment type | Segment value | Rows (test support) | RMSE | Required action |
|--------------|---------------|---------------------|------|-----------------|
| propertyType | Semi-Detached Property | 223 | high | **Business sign-off** (`accept_with_business_signoff`) — documented residual risk |
| price_band | (1_000_000, 4_400_000] | 26_061 | high | **Mitigate in model** — features, weighting, calibration, or segment-specific model |
| propertyType | Detached House | 3_733 | high | **Mitigate in model** — same as above |

## Closure checklist

- [ ] **Semi-Detached (low row count):** Product owner acknowledges elevated RMSE for this slice or accepts constrained use (document reference: `________________`).
- [ ] **High price band:** ML backlog item for tail calibration / interaction features / objective tuning (ticket: `________________`).
- [ ] **Detached houses:** Same mitigation track as price tail where applicable (ticket: `________________`).
- [ ] Re-run full pipeline + `python "src/main.py" --report-only --sync-results`; verify [`results/run_report.md`](../results/run_report.md) segment gate and update frozen CSVs.

## Notes

- Segments with **fewer than 200** test rows cannot fail the automated high-support gate but may still need business review.
- Clearing the gate via **model improvements** is preferred; pure policy waiver requires explicit governance approval.
