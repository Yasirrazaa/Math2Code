# mypy: disable-error-code="attr-defined,operator,index,arg-type,no-any-return,call-overload"
# mypy: disable-error-code="attr-defined,operator,index,arg-type,no-any-return"
#!/usr/bin/env python3
"""Cross-check the competition split's sympy_exp against committed test_cases.

For every row that has a non-empty `sympy_exp`, evaluate it on each test case
input (substitution), evalf to complex, and compare with the committed output
using the same tolerance the eval pipeline uses (`outputs_match`).

This is the *symbolic* half of the competition verification — it answers:
"Does the SymPy expression the contestant wrote actually produce the
test outputs they claim?"

Outputs a report:
  data/split/verify_sympy_report.json

Honest caveats:
- This only checks sympy_exp truth against the committed outputs; it does
  NOT re-derive a fresh symbolic expression from the latex (which would
  require parse_latex and is a separate check).
- sympy_exp may be empty for many rows (about 30% of train — int/float
  unparseable). Empty sympy_exp rows are SKIPPED, not failed.
- The frozen split files are NEVER modified.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import sympy as sp

from math2code.evaluation.metrics import outputs_match

ROOT = Path(__file__).resolve().parent.parent
SPLIT = ROOT / "data" / "split"


def _load(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        with path.open() as fh:
            return [json.loads(line) for line in fh if line.strip()]
    with path.open() as fh:
        return json.load(fh)


def _parse_sympy_exp(s: str) -> sp.Expr | None:
    if not s or not s.strip():
        return None
    # Handle scientific notation, complex, etc.
    # Many sympy_exp strings are simple "expr" — try sympify
    try:
        # Normalize scientific notation if it's a pure number
        if re.fullmatch(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", s.strip()):
            return sp.Float(s)
        # Try complex numbers like "1+2j"
        if "j" in s and re.fullmatch(r"-?[\d\.eE+-]+[+-][\d\.eE+-]+j", s.strip()):
            return complex(s)
        return sp.sympify(s)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None, help="cap rows")
    ap.add_argument(
        "--out", default=str(SPLIT / "verify_sympy_report.json"), help="report path"
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
            print(f"missing: {p}", file=sys.stderr)
            return 1
        rows.extend(_load(p))
    if args.limit:
        rows = rows[: args.limit]
    print(
        f"verifying {len(rows)} rows (sympy_exp truth) across {[p.name for p in paths]}"
    )

    report: dict[str, object] = {
        "split_files": [p.name for p in paths],
        "rows_total": len(rows),
        "rows_with_sympy_exp": 0,
        "rows_skipped_empty": 0,
        "rows_skipped_parse_error": 0,
        "n_pass_full": 0,
        "n_fail_any": 0,
        "failures_by_type": {},
        "failures_examples": [],
        "rows": [],
    }

    for idx, r in enumerate(rows):
        tid = r.get("task_id", f"row_{idx}")
        sym_str = (r.get("sympy_exp") or "").strip()
        if not sym_str:
            report["rows_skipped_empty"] += 1
            continue
        expr = _parse_sympy_exp(sym_str)
        if expr is None:
            report["rows_skipped_parse_error"] += 1
            continue
        report["rows_with_sympy_exp"] += 1

        test_cases = r.get("test_cases", [])
        row_pass = True
        n_passed_cases = 0
        first_err = None
        failed_cases: list[dict] = []

        for ci, tc in enumerate(test_cases):
            inp = tc.get("input", {})
            expected = tc.get("output")
            try:
                subs = {sp.Symbol(k): v for k, v in inp.items()}
                # Apply .doit() before substitution for expressions containing Integral
                # (unevaluated integrands: 1893 integration rows). This resolves the
                # representation-form Integral expressions to closed-form antiderivatives.
                # Differential Eq-form rows (1896) remain non-evaluable (intentional).
                evaluated_expr = expr
                if expr.has(sp.Integral):
                    evaluated_expr = expr.doit()
                got = complex(evaluated_expr.subs(subs).evalf())
                # Build a string to use outputs_match (it accepts str)
                got_str = f"{got.real}{got.imag:+}j" if got.imag != 0 else str(got.real)
                ok = outputs_match(got_str, expected)
            except Exception as exc:
                ok = False
                got_str = f"<error: {exc.__class__.__name__}>"
                if first_err is None:
                    first_err = f"case {ci}: {exc.__class__.__name__}: {str(exc)[:100]}"

            if ok:
                n_passed_cases += 1
            else:
                row_pass = False
                if first_err is None:
                    first_err = (
                        f"case {ci}: got={got_str!r:.80} expected={expected!r:.80}"
                    )
                if len(failed_cases) < 3:
                    failed_cases.append(
                        {
                            "case": ci,
                            "got": got_str[:120],
                            "expected": str(expected)[:120],
                        }
                    )

        entry = {
            "task_id": tid,
            "equation_type": r.get("equation_type"),
            "complexity": r.get("complexity"),
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
            report["failures_by_type"][t] = report["failures_by_type"].get(t, 0) + 1
            if len(report["failures_examples"]) < 20:
                report["failures_examples"].append(
                    {
                        "task_id": tid,
                        "equation_type": t,
                        "complexity": r.get("complexity"),
                        "sympy_exp": sym_str[:120],
                        "first_failure": first_err,
                    }
                )

        if (idx + 1) % 1000 == 0 or idx == len(rows) - 1:
            pct = 100.0 * (idx + 1) / len(rows)
            print(
                f"  [{idx + 1}/{len(rows)} {pct:.1f}%] "
                f"sympy_pass={report['n_pass_full']} sympy_fail={report['n_fail_any']} "
                f"empty={report['rows_skipped_empty']} parse_err={report['rows_skipped_parse_error']}"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    total = report["rows_with_sympy_exp"]
    pct = 100.0 * report["n_pass_full"] / max(total, 1)
    print(
        f"\nSYMPY VERIFICATION COMPLETE: {report['n_pass_full']}/{total} "
        f"({pct:.1f}%) fully pass; {report['n_fail_any']} have failing cases"
    )
    print(
        f"empty sympy_exp skipped: {report['rows_skipped_empty']}, "
        f"parse errors skipped: {report['rows_skipped_parse_error']}"
    )
    print("\nfailures by equation_type:")
    for t, c in sorted(report["failures_by_type"].items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")
    print("\nfirst 20 example failures:")
    for ex in report["failures_examples"]:
        print(
            f"  {ex['task_id']} ({ex['equation_type']}, c={ex['complexity']}): "
            f"sympy='{ex['sympy_exp'][:40]}' {ex['first_failure'][:80] if ex['first_failure'] else '?'}"
        )
    print(f"\nreport -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
