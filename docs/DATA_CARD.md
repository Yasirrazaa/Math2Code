# Data card: Math2Code

Provenance, construction, and measured composition of every dataset artifact.
All counts below are computed from the committed files (`scripts/` + this doc
are reproducible).

## Artifacts

| File | Rows | Ground truth | Role |
|------|------|--------------|------|
| `data/train.json` (gitignored, 51 MB; `train.json.zip` committed) | 26,846 | solutions + 5 expected outputs each | source of the splits |
| `data/split/train.jsonl` (committed) | 22,002 | yes | SFT / GRPO training |
| `data/split/val.json` (committed) | 397 | yes | checkpoint selection |
| `data/split/test.json` (committed) | 397 | yes | **frozen eval** (never trained on) |
| `data/public_test_new_no_sol_no_out.json` (gitignored) | 1,004 | **no** (closed truth) | generalization probe |
| `data/synthetic/*_v1.jsonl` (committed, 15 families + regen of calculus/derivative) | 14,701 (9,871 unique latex) | yes (oracle-verified) | verified code-first synthetic pool |
| `data/synthetic/train_mixture_v2.jsonl` (committed, default RL) | 30,000 (65% comp + 35% synth) | yes | SFT/GRPO mixture (gated slices excluded) |
| `data/synthetic/train_mixture_gated.jsonl` (committed, --include-gated-slices) | 30,000 (with 834 gated rows: special/stats/sets/algebraic) | yes | opt-in gated portfolio mixture |

## Construction (frozen split)

1. `load_competition_train` parses the 26,846 raw rows into the canonical
   `MathCodePair` schema (nan-safe `domain`, 5 test cases each).
2. `dedup_by_latex` keeps first occurrence per identical LaTeX → **22,796**.
3. `make_splits.py` (seed 42) stratifies by `(domain × complexity)` and
   shuffles deterministically → 22,002 / 397 / 397. **Regeneration is
   byte-identical** (verified: `git status` clean after re-run); `manifest.json`
   holds SHA-256 digests enforced by CI's data-smoke job; `test_ids.txt` makes
   contamination checks trivial (grep a model's training ids against it).

## Measured composition (from the committed files)

**Equation types (test split, 397):** integration 41, fractional 35,
differential 33, diophantine 33, summation 33, exponential 32, multivariable
32, exponential_decay 28, trigonometric 29, rational 31, logrithmic 22,
algebraic 22, derivative 25, Geometry 1.

**Complexity (test):** c2 202 · c3 95 · c4 66 · c5 33 · c1 1
(train mirrors this: c2 11,197 / c3 5,275 / c4 3,664 / c5 1,830 / c1 36).

**Output types:** all train/val/test outputs are **real** (floats/ints) —
verified over all 26,846 source rows (0 string/complex outputs).

## The value-add slice (measured)

Counts are **per case** in the analyze output (5 cases/problem); per-problem
figures are stated separately below.

| Slice | `parse_latex` baseline | Count (test) |
|-------|------------------------|--------------|
| Algebraic (rational, diophantine, summation, exponential, multivariable, fractional, logrithmic, algebraic, Geometry) | **100%** | 241 problems / 1,205 cases |
| Trigonometric | 82.8% | 29 problems / 145 cases |
| Calculus (integration, differential, derivative, exponential_decay) | **0%** | 127 problems / 635 cases (7,356 in train) |

Overall the parse baseline is 0.6675 per-problem (265/397); the calculus
slice is the model's measurable value-add target.

## Closed-truth probe (public_test — no outputs shipped)

1,004 synthetic rows; **906 real / 98 complex** outputs; 500 of 1,004 are the
OOD `augmented_equation` type; no `sympy_exp`/`solution`/expected outputs.
Usable for prediction-only generalization probes (e.g., complex-output
robustness), never for accuracy claims. The metric's complex path
(`re±imj` strings) is unit-tested (`test_metrics`), not validated on this
split.

## Known limitations

- Frozen split is real-only; complex generalization is documented, untested.
- `public_test` truth was never released (private bootcamp competition);
  any public accuracy figure must cite OUR frozen split, not the competition.
- `train.json` itself is not re-published (bootcamp licensing) — the split
  JSONs and the generator scripts are the redistributable artifacts.

## Synthetic pool — family inventory (as of 2026-08-11)

24 families registered in `FAMILIES` (21 from SYNTHETIC_EXPANSION + 3 legacy):
- **Existing (9):** `derivative`, `integration`, `functions`, `geometry`, `multivariate`, `numtheory`, `ode`, `sequences`, `edge`
- **Plan v1 (12):** `differential_c1`, `limits`, `series_coeff`, `summation`, `polynomial_invariants`, `matrix_scalars`, `ntheory_ext`, `combinatorics`, `elementary_ext`, `complex_eval`, `geometry_ext`, `solving_scalarized`
- **Gated portfolio (4, opt-in):** `special_functions`, `stats_moments`, `sets_cardinality`, `solving_scalarized` — excluded from default RL mixture.

Pool size: **14,701 rows / 9,871 unique latex** (after regen of calculus_*_v1 and derivative_v1 with `repr_surface=True` to cover the competition's `\mathtt{\text{Integral/Derivative}}` surface).

**Notation alignment vs frozen test (5/5 patterns covered, 100%):**

| Test pattern | Test rows | Pool rows |
|---|---|---|
| `\frac{d}{dx}` | 27 | 1,100 |
| `\log` | 22 | 498 |
| `\sum` | 33 | 426 |
| `\mathtt{\text{Integral(...}}` | 41 | 795 |
| `\mathtt{\text{Derivative(...}}` | 25 | 400 |

**Coefficient kind distribution:** `n/a 10761, rational 541, integer 2378, decimal 920, float 101` (synthetic pool). Train's decimal share: 16.9%.
