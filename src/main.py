from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import assisted_track
import external_benchmark_estimates
import run_london_pipeline
import walk_forward_validation


def _read_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _pass_fail(flag: bool) -> str:
    return "PASS" if flag else "FAIL"


def write_consolidated_report(root: Path) -> Path:
    data_dir = root / "data"
    results_path = data_dir / "model_results_summary.csv"
    wf_path = data_dir / "walk_forward_results.csv"
    bench_path = data_dir / "external_estimate_benchmark.csv"
    assisted_path = data_dir / "assisted_track_results.csv"
    segment_path = data_dir / "segment_metrics.csv"
    decision_path = data_dir / "model_decision_summary.csv"
    track_cmp_path = data_dir / "track_comparison_summary.csv"
    blockers_actions_path = data_dir / "segment_blocker_actions.csv"
    report_path = data_dir / "run_report.md"

    results_df = _read_if_exists(results_path)
    wf_df = _read_if_exists(wf_path)
    bench_df = _read_if_exists(bench_path)
    assisted_df = _read_if_exists(assisted_path)
    segment_df = _read_if_exists(segment_path)

    lines: list[str] = []
    lines.append("# London Model Run Report")
    lines.append("")
    blockers: list[str] = []

    best_primary_rmse = None
    best_benchmark_rmse = None
    walkforward_cv = None

    best_model_name = None
    if results_df is not None and not results_df.empty:
        best = results_df.sort_values("test_rmse").iloc[0]
        best_model_name = str(best["model"])
        best_primary_rmse = float(best["test_rmse"])
        lines.append("## Baseline Summary")
        lines.append("")
        lines.append(f"- Best model: `{best['model']}`")
        lines.append(f"- Test RMSE: `{best['test_rmse']:.2f}`")
        lines.append(f"- Test MAE: `{best['test_mae']:.2f}`")
        lines.append(
            f"- RMSE improvement vs naive: `{best['test_rmse_improvement_vs_naive_pct']:.2f}%`"
        )
        lines.append("")
    else:
        lines.append("## Baseline Summary")
        lines.append("")
        lines.append("- No `model_results_summary.csv` found.")
        lines.append("")

    if wf_df is not None and not wf_df.empty:
        mean_rmse = float(wf_df["rmse"].mean())
        std_rmse = float(wf_df["rmse"].std(ddof=0))
        cv_pct = 100.0 * std_rmse / mean_rmse if mean_rmse else float("nan")
        walkforward_cv = cv_pct
        lines.append("## Walk-Forward Stability")
        lines.append("")
        lines.append(f"- Folds: `{len(wf_df)}`")
        lines.append(f"- Mean RMSE: `{mean_rmse:.2f}`")
        lines.append(f"- RMSE std: `{std_rmse:.2f}`")
        lines.append(f"- Coefficient of variation: `{cv_pct:.2f}%`")
        lines.append("")
    else:
        lines.append("## Walk-Forward Stability")
        lines.append("")
        lines.append("- No `walk_forward_results.csv` found.")
        lines.append("")

    if bench_df is not None and not bench_df.empty:
        best_b = bench_df.sort_values("test_rmse").iloc[0]
        best_benchmark_rmse = float(best_b["test_rmse"])
        lines.append("## External Estimate Benchmark")
        lines.append("")
        lines.append(f"- Best benchmark channel: `{best_b['benchmark']}`")
        lines.append(f"- Benchmark RMSE: `{best_b['test_rmse']:.2f}`")
        lines.append(f"- Rows used: `{int(best_b['test_rows_used'])}`")
        lines.append("")
    else:
        lines.append("## External Estimate Benchmark")
        lines.append("")
        lines.append("- No `external_estimate_benchmark.csv` found.")
        lines.append("")

    best_assisted_rmse = None
    best_assisted_mae = None
    if assisted_df is not None and not assisted_df.empty:
        best_a = assisted_df.sort_values("test_rmse").iloc[0]
        best_assisted_rmse = float(best_a["test_rmse"])
        best_assisted_mae = float(best_a["test_mae"])
        lines.append("## Assisted Track")
        lines.append("")
        lines.append(f"- Best assisted model: `{best_a['model']}`")
        lines.append(f"- Test RMSE: `{best_assisted_rmse:.2f}`")
        lines.append(f"- Test MAE: `{best_assisted_mae:.2f}`")
        lines.append("")
    else:
        lines.append("## Assisted Track")
        lines.append("")
        lines.append("- No `assisted_track_results.csv` found.")
        lines.append("")

    if best_primary_rmse is not None and best_benchmark_rmse is not None:
        cmp_row = {
            "mainline_best_rmse": best_primary_rmse,
            "mainline_best_mae": float(best["test_mae"]) if results_df is not None and not results_df.empty else None,
            "assisted_best_rmse": best_assisted_rmse,
            "assisted_best_mae": best_assisted_mae,
            "external_best_rmse": best_benchmark_rmse,
            "external_best_mae": float(best_b["test_mae"]) if bench_df is not None and not bench_df.empty else None,
        }
        cmp_row["delta_assisted_vs_mainline_rmse"] = (
            (best_assisted_rmse - best_primary_rmse) if best_assisted_rmse is not None else None
        )
        cmp_row["delta_benchmark_vs_mainline_rmse"] = best_benchmark_rmse - best_primary_rmse
        if best_assisted_rmse is not None and best_assisted_rmse < best_primary_rmse:
            cmp_row["preferred_track_rmse"] = "assisted"
        else:
            cmp_row["preferred_track_rmse"] = "mainline"
        pd.DataFrame([cmp_row]).to_csv(track_cmp_path, index=False)

        lines.append("## Cross-Track Comparison")
        lines.append("")
        lines.append(f"- Mainline RMSE: `{best_primary_rmse:.2f}`")
        lines.append(f"- Assisted RMSE: `{best_assisted_rmse:.2f}`" if best_assisted_rmse is not None else "- Assisted RMSE: `n/a`")
        lines.append(f"- External benchmark RMSE: `{best_benchmark_rmse:.2f}`")
        lines.append(
            f"- Preferred track by RMSE: `{cmp_row['preferred_track_rmse']}`"
        )
        lines.append("")

    if best_primary_rmse is not None and best_benchmark_rmse is not None:
        delta_rmse = best_primary_rmse - best_benchmark_rmse
        recommendation = "open_assisted_track" if delta_rmse > 0 else "keep_primary_only"
        governance_recommendation = "pilot_assisted_only"
        if best_assisted_rmse is not None:
            if best_assisted_rmse < best_primary_rmse and best_assisted_rmse <= best_benchmark_rmse:
                governance_recommendation = "deploy_assisted"
            elif best_primary_rmse <= best_assisted_rmse and best_primary_rmse <= best_benchmark_rmse:
                governance_recommendation = "deploy_mainline"
        decision_df = pd.DataFrame(
            [
                {
                    "best_primary_model_rmse": best_primary_rmse,
                    "best_external_benchmark_rmse": best_benchmark_rmse,
                    "best_assisted_model_rmse": best_assisted_rmse,
                    "delta_rmse": delta_rmse,
                    "recommendation": recommendation,
                    "governance_recommendation": governance_recommendation,
                }
            ]
        )
        decision_df.to_csv(decision_path, index=False)
        lines.append("## Decision Summary")
        lines.append("")
        lines.append(f"- Best primary RMSE: `{best_primary_rmse:.2f}`")
        lines.append(f"- Best benchmark RMSE: `{best_benchmark_rmse:.2f}`")
        lines.append(f"- Delta (primary - benchmark): `{delta_rmse:.2f}`")
        lines.append(f"- Recommendation: `{recommendation}`")
        lines.append(f"- Governance recommendation: `{governance_recommendation}`")
        lines.append("")
    else:
        lines.append("## Decision Summary")
        lines.append("")
        lines.append("- Insufficient data to build `model_decision_summary.csv`.")
        lines.append("")

    lines.append("## Release Gates")
    lines.append("")
    gate_oom = results_df is not None and wf_df is not None and bench_df is not None
    lines.append(f"- No crash across baseline + walk-forward + benchmark: `{_pass_fail(gate_oom)}`")
    if not gate_oom:
        blockers.append("Missing one or more required artifacts from baseline/walk-forward/benchmark runs.")

    gate_primary = best_primary_rmse is not None and best_primary_rmse <= 396614.68
    lines.append(f"- Best primary RMSE <= 396614.68: `{_pass_fail(gate_primary)}`")
    if not gate_primary:
        blockers.append("Primary model RMSE regressed above 396614.68.")

    gate_cv = walkforward_cv is not None and walkforward_cv <= 26.0
    lines.append(f"- Walk-forward CV <= 26%: `{_pass_fail(gate_cv)}`")
    if not gate_cv:
        blockers.append("Walk-forward stability above CV threshold (26%).")

    segment_threshold = best_primary_rmse * 1.8 if best_primary_rmse is not None else None
    if segment_df is not None and not segment_df.empty and segment_threshold is not None and best_model_name is not None:
        # Support-aware rule: only high-support segments (>=200 rows) can hard-fail this gate.
        high_support = segment_df[(segment_df["rows"] >= 200) & (segment_df["model"] == best_model_name)].copy()
        critical = high_support[high_support["rmse"] > segment_threshold].copy()
        gate_segment = critical.empty
        lines.append(
            f"- High-support segment RMSE <= 1.8x overall ({segment_threshold:.2f}): `{_pass_fail(gate_segment)}`"
        )
        if not gate_segment:
            worst = critical.sort_values("rmse", ascending=False).head(5)
            actions = []
            for r in worst.itertuples():
                status = "mitigate_in_model"
                owner = "ml_team"
                if int(r.rows) < 500:
                    status = "accept_with_business_signoff"
                    owner = "product_owner"
                actions.append(
                    {
                        "segment_type": r.segment_type,
                        "segment_value": r.segment_value,
                        "rows": int(r.rows),
                        "rmse": float(r.rmse),
                        "status": status,
                        "action_owner": owner,
                        "mitigation_plan": "segment_features_or_calibration" if status == "mitigate_in_model" else "business_risk_acceptance",
                    }
                )
            pd.DataFrame(actions).to_csv(blockers_actions_path, index=False)
            blockers.append(
                "High-support segments above threshold: "
                + "; ".join(
                    f"{r.segment_type}:{r.segment_value} ({r.model}, rmse={r.rmse:.2f}, rows={int(r.rows)})"
                    for r in worst.itertuples()
                )
            )
        else:
            pd.DataFrame(
                columns=["segment_type", "segment_value", "rows", "rmse", "status", "action_owner", "mitigation_plan"]
            ).to_csv(blockers_actions_path, index=False)
    else:
        gate_segment = False
        lines.append("- High-support segment gate unavailable: `FAIL`")
        blockers.append("Missing `segment_metrics.csv` or primary RMSE needed for segment gate.")
    lines.append("")

    lines.append("## Blockers")
    lines.append("")
    if blockers:
        for b in blockers:
            lines.append(f"- {b}")
    else:
        lines.append("- None")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run London housing pipeline tasks.")
    parser.add_argument("--all", action="store_true", help="Run all tasks then write report.")
    parser.add_argument("--baseline", action="store_true", help="Run baseline pipeline.")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward validation.")
    parser.add_argument("--benchmark", action="store_true", help="Run external benchmark.")
    parser.add_argument("--assisted", action="store_true", help="Run assisted track benchmark model.")
    parser.add_argument("--report-only", action="store_true", help="Only write consolidated report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    run_baseline = args.all or args.baseline
    run_wf = args.all or args.walk_forward
    run_benchmark = args.all or args.benchmark
    run_assisted = args.all or args.assisted
    report_only = args.report_only

    if not any([run_baseline, run_wf, run_benchmark, run_assisted, report_only]):
        run_baseline = True
        run_wf = True
        run_benchmark = True
        run_assisted = True

    if run_baseline:
        print("[main] Running baseline pipeline...", flush=True)
        run_london_pipeline.main()
    if run_wf:
        print("[main] Running walk-forward validation...", flush=True)
        walk_forward_validation.main()
    if run_benchmark:
        print("[main] Running external benchmark...", flush=True)
        external_benchmark_estimates.main()
    if run_assisted:
        print("[main] Running assisted track...", flush=True)
        assisted_track.main()

    report_path = write_consolidated_report(root)
    print(f"[main] Saved consolidated report: {report_path}", flush=True)


if __name__ == "__main__":
    main()

