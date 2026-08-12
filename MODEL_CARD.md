---
language: en
license: mit
tags:
  - math
  - code
  - latex
  - sympy
  - grpo
  - rlvr
---

# Math2Code GRPO (Qwen2.5-Math-7B)

Generates **executable Python/SymPy** code from a LaTeX math expression —
the Math2Code competition task (bootcamp capstone, 2024), rebuilt as a
measurement-first ML engineering project.

## Model status

| Field | Value |
|-------|-------|
| Base | `Qwen/Qwen2.5-Math-7B` |
| Training | SFT warmup → GRPO (TRL 1.9.x `GRPOTrainer`, deterministic rewards) |
| Status | **pending** — see `docs/TRAINING.md`; the harness, data, and reward stack are verified on CPU |
| Checkpoint | HF link (uploaded from `runs/grpo-main-7b/step_XXX`) |

## Task

Input: a LaTeX expression (machine-generated; `\sum`, `\int`, `\frac{d}{dx}`,
`\mathtt{...}`). Output: a `def calculate(...)` Python function using only
allowed symbols (sympy/numpy/math) that evaluates to the expected answers on
5 hidden numeric inputs per problem. Deterministic correctness metric
(complex-aware, `-inf` semantics, per-problem accuracy).

## Verified data integrity (full competition split, 22,796 rows)

Verified via `scripts/verify_competition.py` and `scripts/verify_sympy_exp.py`
(see `docs/COMPETITION_VERIFICATION.md`). The frozen split is NEVER modified.

| Check | Rows | Pass rate | Failures | Source file |
|---|---|---|---|---|
| Gold-code (solution vs test_cases) | 22,796 | **100.00%** (22,796 / 22,796, 0 fail) | 0 | `data/split/verify_report_combined.json` |
| Sympy truth (sympy_exp vs test_cases, closed-form subset) | 19,007 | **100.00%** | 0 | `data/split/verify_sympy_report.json` |
| Sympy truth (representation-form: ODE Eq / unevaluated Integral) | 3,789 | N/A (problem form, verified via gold-code) | 0 (intentional form) | `docs/COMPETITION_VERIFICATION.md` |

The 3,789 representation-form rows (`differential`: Eq-form ODE; `integration`: Integral-form integrand) are an intentional dataset design feature — their `sympy_exp` is the problem statement, not the solution. The gold-code `solution` Python functions are verified correct for all of them.

Reproduce: `python scripts/verify_competition.py --split all --workers 4 --timeout 10`; `python scripts/verify_sympy_exp.py --split all`. Full protocol in `docs/COMPETITION_VERIFICATION.md`.

## Evals (frozen test split: 397 problems, stratified, SHA-256 manifest)

| Model | Per-problem accuracy | CI95 | Source |
|-------|---------------------|------|--------|
| Trivial floor (always 0) | 0.0076 | — | measured |
| SymPy `parse_latex` (zero-cost) | 0.6675 | — | measured |
| **This model (7B GRPO)** | *pending* | — | runner.py |
| Gold solutions (harness sanity) | 1.0000 | — | measured |

Reproduce: `make eval-gold`, `make baselines`, then
`python -m math2code.evaluation.runner --model hf:<this repo>`.
Full protocol in `docs/ENGINEERING_REVIEW.md` and `PLAN.md`.

## Known value-add slice

`parse_latex` solves the algebraic slice at 100% but **0%** on the calculus
slice (integration/differential/derivative/exponential_decay — 127 problems /
635 test cases): that slice is what this model must add. The competition's
closed-truth public test additionally holds 98 complex-output rows
(generalization probe only: no ground truth shipped; see `docs/DATA_CARD.md`).

**Data verification note:** All 22,796 rows of the frozen split have been verified internally consistent. The `differential` and `integration` equation types (127 problems / 635 test cases in the calculus slice) use representation-form `sympy_exp` (Eq / Integral) rather than closed-form expressions. The model should generate `solution` Python code (not rely solely on `sympy_exp.subs()` evaluation) for these types.

## Sandboxing & safety

Rollouts run inside a subprocess sandbox: AST import allowlist (sympy, numpy,
math, cmath, itertools, functools, collections, fractions, statistics,
decimal), no `os`/`subprocess`/`eval`/`exec`/`open`/`socket`, RLIMIT_AS +
RLIMIT_FSIZE + RLIMIT_NOFILE, SIGALRM timeout (2 s rollouts / 20 s eval),
and a self-healing worker pool (389 exec/s measured).

## Costs

Burn-in on free T4; main 7B run on spot 3090/4090 (~$25–45, hard cap $100).
All CPU phases are $0.

## Disclaimer

Model card numbers are updated after each run per `docs/TRAINING.md` §6.
This card intentionally reports *pending* rows until the runs exist — no
retroactive claims.
