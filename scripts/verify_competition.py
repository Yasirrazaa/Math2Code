# mypy: disable-error-code="attr-defined,operator,index,arg-type,no-any-return,call-overload"
#!/usr/bin/env python3
# mypy: disable-error-code="attr-defined,operator,index,arg-type,no-any-return"
"""Verify the frozen competition split end-to-end.

For every row in data/split/{train,val,test}.jsonl[.json], run the row's
committed `solution` (Python code) against its committed `test_cases`
(input/output pairs) in the sandbox pool, and compare outputs with the
same tolerance the eval pipeline uses (`outputs_match`).

Outputs a per-row report:
  data/split/verify_report.json — {n_pass, n_fail, n_error, per-row details}
The frozen split files are NEVER modified; this is observational.

Honest caveats:
- Sandbox timeouts (10s/solve) flag slow solutions as failures.
- Some competition solutions use numpy/pandas — the sandbox allows numpy.
- This script does NOT re-derive truth from `sympy_exp`; it checks the
  committed (solution, test_cases) PAIR for internal consistency.
- Sympy_exp is a separate truth cross-check (verify_sympy_exp.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from math2code.evaluation.metrics import outputs_match
from math2code.sandbox import SandboxPool

ROOT = Path(__file__).resolve().parent.parent
SPLIT = ROOT / "data" / "split"


def _load(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        with path.open() as fh:
            return [json.loads(line) for line in fh if line.strip()]
    with path.open() as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    ap.add_argument("--workers", type=int, default=4, help="sandbox pool size")
    ap.add_argument("--timeout", type=int, default=10, help="per-spawn timeout (s)")
    ap.add_argument("--limit", type=int, default=None, help="cap rows for smoke test")
    ap.add_argument(
        "--skip-rows", type=int, default=0,
        help="skip the first N rows of the concatenated dataset (for resume after a "
             "previous run was killed mid-way). Use with the same --out and --split "
             "as the previous run. Combine with --checkpoint-in to safely merge partials.",
    )
    ap.add_argument(
        "--checkpoint-in", type=str, default=None,
        help="path to a previous partial report JSON (with 'rows' list). Rows whose "
             "task_id appears in the prior report are skipped. Combined with --skip-rows.",
    )
    ap.add_argument(
        "--checkpoint-every", type=int, default=500,
        help="write the partial report every N rows (default 500) so a kill never "
             "loses more than N rows of work. Set 0 to disable.",
    )
    ap.add_argument(
        "--out", default=str(SPLIT / "verify_report.json"), help="report path"
    )
    args = ap.parse_args()

    paths: list[Path] = []
    if args.split in ("all", "train"):
        paths.append(SPLIT / "train.jsonl")
    if args.split in ("all", "val"):
        paths.append(SPLIT / "val.json")
    if args.split in ("all", "test"):
        paths.append(SPLIT / "test.json")

    rows: list[dict] = []
    for p in paths:
        if not p.exists():
            print(f"missing split file: {p}", file=sys.stderr)
            return 1
        rows.extend(_load(p))

    # Load prior checkpoint for dedup
    prior: dict[str, dict] = {}
    if args.checkpoint_in and Path(args.checkpoint_in).exists():
        for entry in json.load(open(args.checkpoint_in)).get("rows", []):
            prior[entry["task_id"]] = entry
        print(f"loaded {len(prior)} prior verified rows from {args.checkpoint_in}")

    if args.skip_rows:
        skipped = rows[: args.skip_rows]
        rows = rows[args.skip_rows :]
        print(f"skipped first {len(skipped)} rows (--skip-rows)")
    if args.limit:
        rows = rows[: args.limit]
    print(f"verifying {len(rows)} rows across {[p.name for p in paths]}")

    n_cases_total = sum(len(r.get("test_cases", [])) for r in rows)
    print(
        f"rows: {len(rows)}, total test cases: {n_cases_total}, "
        f"workers={args.workers}, timeout={args.timeout}s"
    )

    # Initialize from prior if present so aggregates are consistent.
    if prior:
        agg = json.load(open(args.checkpoint_in))
        report: dict[str, object] = {
            "split_files": [p.name for p in paths],
            "workers": args.workers,
            "timeout_s": args.timeout,
            "rows_checked": len(prior),
            "n_pass_full": agg.get("n_pass_full", 0),
            "n_fail_any": agg.get("n_fail_any", 0),
            "n_solution_error": agg.get("n_solution_error", 0),
            "failures_by_type": dict(agg.get("failures_by_type", {})),
            "failures_by_complexity": dict(agg.get("failures_by_complexity", {})),
            "failures_by_output_type": dict(agg.get("failures_by_output_type", {})),
            "failures_examples": list(agg.get("failures_examples", [])),
            "rows": list(prior.values()),
        }
    else:
        report: dict[str, object] = {
            "split_files": [p.name for p in paths],
            "workers": args.workers,
            "timeout_s": args.timeout,
            "rows_checked": 0,
            "n_pass_full": 0,
            "n_fail_any": 0,
            "n_solution_error": 0,  # solution failed to run at all
            "failures_by_type": {},
            "failures_by_complexity": {},
            "failures_by_output_type": {},
            "failures_examples": [],  # up to 20 with first_failure
            "rows": [],
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint() -> None:
        # Write atomically: tmp file + rename. Survives a kill mid-write.
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(report, indent=2))
        tmp.rename(out_path)

    n_skipped_dups = 0
    with SandboxPool(
        n_workers=args.workers, timeout_s=args.timeout, memory_mb=2048
    ) as pool:
        for idx, r in enumerate(rows):
            tid = r.get("task_id", f"row_{idx}")
            if tid in prior:
                n_skipped_dups += 1
                continue
            solution = r.get("solution", "")
            test_cases = r.get("test_cases", [])
            # IMPORTANT: each test case is {input: {...}, output: ...};
            # run_solution_on_cases expects the input dict, not the full case.
            inputs = [tc.get("input", {}) for tc in test_cases]
            outputs, errors = pool.run_solution_on_cases(solution, inputs)

            row_pass = True
            first_err = None
            n_passed_cases = 0
            failed_cases: list[dict] = []

            for ci, (tc, out, err) in enumerate(zip(test_cases, outputs, errors)):
                expected = tc.get("output")
                if out is None:
                    row_pass = False
                    if first_err is None:
                        first_err = f"case {ci}: solution error: {err[:160]}"
                    failed_cases.append(
                        {"case": ci, "reason": err[:160] if err else "no output"}
                    )
                    continue
                try:
                    ok = outputs_match(out, expected)
                except Exception as exc:
                    ok = False
                    if first_err is None:
                        first_err = f"case {ci}: compare error: {exc}"
                if ok:
                    n_passed_cases += 1
                else:
                    row_pass = False
                    if first_err is None:
                        first_err = (
                            f"case {ci}: got={out!r:.80} expected={expected!r:.80}"
                        )
                    failed_cases.append(
                        {
                            "case": ci,
                            "got": str(out)[:120],
                            "expected": str(expected)[:120],
                        }
                    )
                if len(failed_cases) >= 3:
                    break  # cap trace

            entry = {
                "task_id": tid,
                "equation_type": r.get("equation_type"),
                "complexity": r.get("complexity"),
                "domain": r.get("domain"),
                "output_type": r.get("output_type"),
                "n_cases": len(test_cases),
                "n_passed": n_passed_cases,
                "all_passed": row_pass,
                "first_failure": None if row_pass else first_err,
            }
            if not row_pass:
                entry["failed_cases_sample"] = failed_cases
            report["rows"].append(entry)

            if row_pass:
                report["n_pass_full"] += 1
            else:
                report["n_fail_any"] += 1
                t = r.get("equation_type", "?")
                report["failures_by_type"][t] = (
                    report["failures_by_type"].get(t, 0) + 1
                )
                c = str(r.get("complexity"))
                report["failures_by_complexity"][c] = (
                    report["failures_by_complexity"].get(c, 0) + 1
                )
                ot = r.get("output_type", "?")
                report["failures_by_output_type"][ot] = (
                    report["failures_by_output_type"].get(ot, 0) + 1
                )
                # Keep up to 20 example failures with their first reason
                if len(report["failures_examples"]) < 20:
                    report["failures_examples"].append(
                        {
                            "task_id": tid,
                            "equation_type": t,
                            "complexity": c,
                            "first_failure": first_err,
                        }
                    )

            if (idx + 1) % 500 == 0 or idx == len(rows) - 1:
                pct = 100.0 * (idx + 1) / len(rows)
                print(
                    f"  [{idx+1}/{len(rows)} {pct:.1f}%] pass={report['n_pass_full']} "
                    f"fail={report['n_fail_any']}"
                )

            # Periodic checkpoint so a kill loses at most --checkpoint-every rows.
            if args.checkpoint_every and (
                (idx + 1) % args.checkpoint_every == 0
            ):
                report["rows_checked"] = len(report["rows"])
                checkpoint()
                print(f"  [checkpoint] wrote {len(report['rows'])} rows to {out_path}")

    report["rows_checked"] = len(report["rows"])
    checkpoint()

    pct = 100.0 * report["n_pass_full"] / max(report["rows_checked"], 1)
    print(
        f"\nVERIFICATION COMPLETE: {report['n_pass_full']}/{report['rows_checked']} "
        f"({pct:.1f}%) fully pass; {report['n_fail_any']} have at least one failing case; "
        f"{n_skipped_dups} duplicates skipped via --checkpoint-in"
    )
    print("\nfailures by equation_type:")
    for t, c in sorted(report["failures_by_type"].items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")
    print("\nfailures by complexity:")
    for c, n in sorted(report["failures_by_complexity"].items()):
        print(f"  {c}: {n}")
    print("\nfailures by output_type:")
    for ot, n in sorted(report["failures_by_output_type"].items(), key=lambda x: -x[1]):
        print(f"  {ot}: {n}")
    print("\nfirst 20 example failures:")
    for ex in report["failures_examples"]:
        print(
            f"  {ex['task_id']} ({ex['equation_type']}, c={ex['complexity']}): "
            f"{ex['first_failure'][:100] if ex['first_failure'] else '?'}"
        )
    print(f"\nreport -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
