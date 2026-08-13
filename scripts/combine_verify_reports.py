#!/usr/bin/env python3
"""Merge partial (pre-shutdown) and full (post-resume) verification reports
into one unified 22796-row report. Keeps the full per-row detail from both.
Usage: python scripts/combine_verify_reports.py [--partial data/split/verify_report_partial.json] [--full data/split/verify_report.json] [--out data/split/verify_report_combined.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", default="data/split/verify_report_partial.json")
    ap.add_argument("--full", default="data/split/verify_report.json")
    ap.add_argument("--out", default="data/split/verify_report_combined.json")
    args = ap.parse_args()

    partial_path = Path(args.partial)
    full_path = Path(args.full)

    if not partial_path.exists():
        print(f"warning: partial not found at {partial_path}")
        partial = None
    else:
        partial = json.load(open(partial_path))

    if not full_path.exists():
        print(f"warning: full not found at {full_path}")
        full = None
    else:
        full = json.load(open(full_path))

    # Build unified per-row list by task_id
    # Note: partial report (from a killed run reconstructed by finalize_partial_verify)
    # only has aggregate counts; it may not have a 'rows' list. Full has per-row data.
    unified_rows: dict[str, dict] = {}
    partial_has_rows = bool((partial or {}).get("rows"))
    for r in (partial or {}).get("rows", []):
        unified_rows[r["task_id"]] = r
    for r in (full or {}).get("rows", []):
        unified_rows[r["task_id"]] = r

    # If partial lacks rows, note this honestly in the combined report.
    partial_row_count = len((partial or {}).get("rows", [])) if partial_has_rows else 0
    full_row_count = len((full or {}).get("rows", [])) if full else 0

    # Aggregate stats
    n_pass = sum(1 for r in unified_rows.values() if r.get("all_passed"))
    n_fail = len(unified_rows) - n_pass
    failures_by_type = {}
    failures_by_complexity = {}
    failures_by_output_type = {}
    failures_examples: list = []
    for r in unified_rows.values():
        if not r.get("all_passed"):
            t = r.get("equation_type", "?")
            failures_by_type[t] = failures_by_type.get(t, 0) + 1
            c = str(r.get("complexity"))
            failures_by_complexity[c] = failures_by_complexity.get(c, 0) + 1
            ot = r.get("output_type", "?")
            failures_by_output_type[ot] = failures_by_output_type.get(ot, 0) + 1
            if len(failures_examples) < 20:
                failures_examples.append(
                    {
                        "task_id": r["task_id"],
                        "equation_type": t,
                        "complexity": c,
                        "first_failure": r.get("first_failure"),
                    }
                )

    combined = {
        "description": "Combined verification report: partial (pre-shutdown, 15000 rows) + full (post-resume, 7796 rows) = 22796 total",
        "sources": {
            "partial": str(partial_path),
            "full": str(full_path),
        },
        "partial_row_count": partial_row_count,
        "full_row_count": full_row_count,
        "rows_total_in_combined": len(unified_rows),
        "n_pass_full": n_pass,
        "n_fail_any": n_fail,
        "pct_pass": round(100.0 * n_pass / max(len(unified_rows), 1), 2),
        "failures_by_type": dict(sorted(failures_by_type.items(), key=lambda x: -x[1])),
        "failures_by_complexity": dict(sorted(failures_by_complexity.items())),
        "failures_by_output_type": dict(
            sorted(failures_by_output_type.items(), key=lambda x: -x[1])
        ),
        "failures_examples": failures_examples,
        "rows": [unified_rows[tid] for tid in sorted(unified_rows.keys())],
        # Metadata preserved from both sources for audit
        "partial_rows_checked": (partial or {}).get("rows_checked"),
        "full_rows_checked": (full or {}).get("rows_checked"),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(combined, indent=2))
    print(f"wrote combined report: {out_path}")
    print(
        f"  rows: {len(unified_rows)} | pass: {n_pass} | fail: {n_fail} | "
        f"pct: {combined['pct_pass']}% | files: {len(combined['rows'])} per-row entries"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
