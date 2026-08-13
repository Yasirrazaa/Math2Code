#!/usr/bin/env python3
"""Write a final report from a partial verify_competition.log run.

When the full run was killed before completion, this script reconstructs
the report by parsing the progress log lines ('[N/M P%] pass=X fail=Y').

Usage:
    python scripts/finalize_partial_verify.py \\
        --log m2c_verify_partial.log \\
        --out data/split/verify_report_partial.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PROGRESS_RE = re.compile(r"\[(\d+)/(\d+) ([\d.]+)%\] pass=(\d+) fail=(\d+)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default=str(ROOT / "m2c_verify_partial.log"))
    ap.add_argument(
        "--out", default=str(ROOT / "data/split/verify_report_partial.json")
    )
    args = ap.parse_args()

    rows_done = 0
    rows_total = 0
    n_pass = 0
    n_fail = 0
    last_line = ""
    for line in Path(args.log).read_text().splitlines():
        m = PROGRESS_RE.search(line)
        if not m:
            continue
        rows_done = int(m.group(1))
        rows_total = int(m.group(2))
        n_pass = int(m.group(4))
        n_fail = int(m.group(5))
        last_line = line

    if not rows_done:
        print("no progress lines found in log; nothing to write")
        return 1

    pct = 100.0 * n_pass / max(rows_done, 1)
    print(
        f"reconstructed: {n_pass}/{rows_done} pass ({pct:.2f}%), "
        f"{n_fail} fail, {rows_total - rows_done} not yet run"
    )
    print(f"last progress line: {last_line}")

    report = {
        "source_log": args.log,
        "rows_checked": rows_done,
        "rows_total_target": rows_total,
        "rows_remaining": rows_total - rows_done,
        "n_pass_full": n_pass,
        "n_fail_any": n_fail,
        "pct_pass": pct,
        "status": "PARTIAL — full run was killed before completion",
        "how_to_resume": (
            "Re-run `python scripts/verify_competition.py --split all "
            "--workers 6 --timeout 10 --out data/split/verify_report.json`. "
            "The script is idempotent: it re-runs all rows. If you want "
            "incremental resume, modify the script to skip task_ids "
            "already in this partial report (none — the partial report "
            "has only aggregates, no per-row data). Suggestion: just "
            "re-run from scratch (~2h on this machine)."
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
