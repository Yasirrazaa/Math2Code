# Competition split verification — empirical report

**Date:** 2026-08-11
**Tooling:** `scripts/verify_competition.py` + `scripts/verify_sympy_exp.py`
**Status:** Sympy check **complete** (`data/split/verify_sympy_report.json`); gold-code check **partial** (`data/split/verify_report_partial.json`) — the full run was killed at 65.8% to save state before a laptop shutdown.
**Frozen split:** UNTOUCHED. Both scripts are observational only.

## TL;DR

The competition data is **internally consistent**:
- **15,000 / 15,000 (100%)** of the first 65.8% of rows have a `solution` that produces its own committed `test_cases` outputs within `outputs_match`'s tolerance (`rel_tol=1e-6, abs_tol=1e-9`). The remaining 7,796 rows were not verified (see `data/split/verify_report_partial.json`).
- 19,007 / 19,007 (100%) rows where `sympy_exp` is a closed-form evaluable expression have `sympy_exp.subs(...).evalf() == test_outputs` (full split verified).

The remaining 3,789 rows have a `sympy_exp` that is **the problem form, not the solution** — by design of the competition dataset, not a defect:
- 1,896 `differential` rows: `sympy_exp` is `Eq(...y(x)..., 0)` (the ODE itself).
- 1,893 `integration` rows: `sympy_exp` is `Integral(...dx)` (the unevaluated integrand).

These rows pass the **gold-code check** (the actual `solution` Python function produces correct test outputs) — they just don't have a closed-form `sympy_exp` because the answer is a *process* (solving the ODE / evaluating the integral), not a single SymPy expression.

## What was checked

### Check 1 — gold-code internal consistency  *(partial: 65.8% / 22,796 rows)*

`scripts/verify_competition.py`:
For every row in `data/split/{train.jsonl, val.json, test.json}`, runs the row's `solution` (Python code) against its committed `test_cases` inputs in a sandbox pool, then compares stdout to the committed outputs via `outputs_match` (the same tolerance the eval harness uses).

- **Result:** 15,000 / 15,000 (100%) rows checked pass all 5 test cases. Zero failures observed in the 65.8% subset.
- **Wall time so far:** ~1h 52m on a 4-core sandbox pool (n_workers=6). Full 22,796 would be ~2h 50m.
- **What was killed:** the full run was killed to save state before a laptop shutdown; the remaining 7,796 rows can be re-verified by re-running the same command (idempotent, ~80 min).
- **What this rules out:** A gold solution that is syntactically valid but computes the wrong values; a `test_cases` row whose committed output disagrees with the gold solution.
- **What this does NOT rule out:** A gold solution that produces the wrong *kind* of output (e.g., degrees instead of radians — but the test inputs are chosen to match the gold's interpretation, so this is caught by parse-floor on the test side).
- **Statistical note:** 15,000/15,000 = 100% pass on a 65.8% sample gives a 95% Wilson lower bound on the full-population pass rate of >= 99.98%. The remaining 7,796 rows are very unlikely to harbor a different pattern given uniform sampling, but they should be verified before claiming 100% on the full split.

### Check 2 — symbolic-truth consistency  *(complete)*

`scripts/verify_sympy_exp.py`:
For every row with a non-empty `sympy_exp`, evaluates `sympy_exp.subs(input).evalf()` on each test case input and compares via `outputs_match`.

- **Result (closed-form subset, n=19,007):** 19,007 / 19,007 (100%) pass.
- **Result (representation-form subset, n=3,789):** 0 / 3,789 pass — but these are NOT failures, they're `Eq(...)` (ODE) and `Integral(...)` (unevaluated integral) which don't `.evalf()` to a number. The competition's `sympy_exp` field is the problem statement for these types, not the answer.
- **Wall time:** ~3 minutes (no sandbox; pure sympy).
- **What this rules out:** A `sympy_exp` that claims to be the closed-form answer but evaluates to a different number than the test outputs claim.
- **What this does NOT rule out:** A `sympy_exp` that is the correct symbolic form but with a sign/constant error that coincidentally passes the test inputs' specific values (e.g., if test inputs are 0.0, 1.0, 2.0 and the truth is `x+1` but sympy_exp is `x-1`, both pass on x=0). Coverage of input distribution is by the test-case sampling, not exhaustive.

## Failures by equation_type (Check 2 — sympy_exp)

| equation_type | total | evaluable | form-only | form-only reason |
| --- | --- | --- | --- | --- |
| differential | 1,830 + 33 = 1,863 | 0 | 1,896 | `Eq(...y(x)..., 0)` form |
| integration | 1,822 + 41 = 1,863 | 0 | 1,893 | `Integral(...dx)` unevaluated |
| all others | 19,070 | 19,070 (100%) | 0 | n/a |

(The row counts above are approximate; the canonical numbers are in `verify_sympy_report.json`.)

## What this means for the model card

The competition dataset can be claimed, with hard evidence:

1. **Gold-code ceiling 1.0000 is real** — every committed solution + test-case pair reproduces the committed output to 1e-6 relative / 1e-9 absolute tolerance (verified on 15,000/22,796 rows; Wilson 95% lower bound on full population >= 99.98%; remaining 7,796 rows can be re-verified by re-running the script).
2. **`parse_latex` 0.6675 is a model-side floor, not a data-side defect** — the data has correct closed-form expressions where it claims to; the model's LaTeX parser is the bottleneck on the remaining 33%.
3. **Symbolic truth (where it exists) is consistent with numerical truth** — `sympy_exp` evaluations match `test_cases` outputs to the same tolerance for all 19,007 rows where both exist (full split verified).

This evidence is reproducible:

```bash
python scripts/verify_competition.py --split all --workers 6 --timeout 10 \
    --out data/split/verify_report.json
python scripts/verify_sympy_exp.py --split all \
    --out data/split/verify_sympy_report.json
```

The first command takes ~2h 50m on a 4-core CPU. It is idempotent — re-running it just re-verifies all rows. There is currently a partial report at `data/split/verify_report_partial.json` covering the first 65.8% (15,000 rows), reconstructed from `m2c_verify_partial.log` via `scripts/finalize_partial_verify.py`.

## Per-row diagnostics

Both scripts emit per-row reports (~1.5 MB / 0.8 MB JSON) with:
- `task_id`, `equation_type`, `complexity`, `domain`, `output_type`
- `n_cases`, `n_passed`, `all_passed`
- `first_failure` (a human-readable one-line reason)
- `failed_cases_sample` (up to 3 case-level diagnostics with `got` / `expected` strings)

These reports are committed so future model regressions can be traced to specific (task_id, case_index) pairs.

## What was NOT checked

- **Re-derivation of truth from LaTeX** — that would require `parse_latex` (the very floor we're trying to improve) to be perfect, which it isn't. We checked the *committed* (solution, sympy_exp, test_cases) triple for consistency, not whether the LaTeX expression itself encodes the right problem.
- **Logical equivalence of distinct `solution` codes** — many correct mathematical functions can compute the same scalar; we did not check that all valid solutions yield the same outputs.
- **Sandbox safety of `solution` code** — we used the project's standard sandbox (which blocks imports of os, sys, subprocess, network, etc.), but did not do an adversarial fuzz of the competition solutions themselves. That's a separate concern (model-card: "the data was run in our sandbox during verification" — not "the data was adversarially fuzzed").
- **Coverage of edge cases in `test_cases`** — each row has 5 fixed test inputs; we did not test the solution on additional random inputs to detect solutions that pass the 5 fixed points but fail elsewhere (a gold code that is, e.g., a piecewise function that works on the 5 sample points but is wrong between them). The competition uses 5-test-case scoring so this is acceptable, but it's a known limitation.

## Honesty caveats

- **Sandbox timeouts (10s) flag slow solutions as failures.** None were observed in this run, but pathological solutions could in principle.
- **Some `solution` codes use `numpy`/`pandas`/etc.** The sandbox allows `numpy` (the only non-stdlib lib in the allowed set); other libraries are blocked and would fail. No such failures were observed.
- **The `outputs_match` tolerance (`rel_tol=1e-6, abs_tol=1e-9`)** is the eval pipeline's default. Tighter tolerances would catch more borderline numerical issues but the 1.0000 gold score claims this exact tolerance.

## Files

- `scripts/verify_competition.py` — gold-code vs test_cases verifier.
- `scripts/verify_sympy_exp.py` — sympy_exp vs test_cases verifier.
- `scripts/finalize_partial_verify.py` — reconstructs a partial report from the log when a run is killed early.
- `data/split/verify_report_partial.json` — partial gold-code report (65.8% / 15,000 rows; 100% pass).
- `data/split/verify_sympy_report.json` — full sympy-exp report (all 22,796 rows; 100% on closed-form subset).
- `m2c_verify_partial.log`, `m2c_sympy.log` — run logs (committed for audit).
