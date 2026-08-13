---
license: mit
pretty_name: Math2Code — LaTeX to Executable SymPy
task_categories:
  - text-generation
language:
  - en
tags:
  - math
  - latex
  - code-generation
  - sympy
  - grpo
  - rlvr
  - competition
  - verification
size_categories:
  - 10K<n<100K
configs:
  - config_name: split
    data_files:
      - split: train
        path: data/train.jsonl
      - split: validation
        path: data/val.json
      - split: test
        path: data/test.json
  - config_name: synthetic
    data_files:
      - split: train
        path: data/synthetic/*.jsonl
---

# Math2Code — LaTeX → executable, verifiable SymPy

A dataset for the task: **given a LaTeX math expression, generate a Python
`calculate(...)` function (SymPy/numpy/math) that evaluates it correctly** —
scored by *executing* the code on numeric test inputs and comparing outputs
with `isclose` (complex-aware), never by string matching.

Two artifacts are published here:

1. **`split`** — the frozen competition split: 22,002 train / 397 val / 397
   test rows with `solution`, ground-truth `sympy_exp` and 5 test cases each,
   SHA-256-pinned in `data/manifest.json` (enforced by CI).
2. **`synthetic`** — an oracle-verified, code-first synthetic pool: **14,701
   rows / 9,871 unique LaTeX** across 26 family pools (24 registered
   programmatic families; calculus spans definite/indefinite/variable pools),
   plus per-family accept/reject reports. Every row passed the sandbox oracle
   on 20 fresh inputs.

## Why this dataset exists (measured)

The 22k-row competition train is balanced-by-count but structurally narrow:
~7 functions, 0 cross-terms, 0 `\int`/`\lim`/`\partial`, 100% real outputs.
The `synthetic` config is the repair: notation-mutated LaTeX (same AST, many
renderings), a full elementary-function vocabulary, coupled multivariates,
and complex-output rows — all deterministically verifiable.

The model's measurable value-add is the **calculus slice**: the zero-cost
SymPy `parse_latex` baseline scores 0.6675 per-problem overall but **0%** on
the calculus slice (integration/differential/derivative/exponential_decay —
127 problems / 635 cases in the frozen test).

## Row schema (shared by both configs)

```json
{
  "task_id": "derivative_17:v0",
  "latex_expression": "\\mathtt{\\text{Derivative(7*x**3 + 2, x)}}",
  "solution": "import sympy as sp\n_expr = sp.sympify('...')\ndef calculate(x):\n    ...",
  "sympy_exp": "21*x**2",
  "truth_code": null,
  "test_cases": [{"input": {"x": 1.0}, "output": 21.0}, {"input": {"x": 2.5}, "output": 131.25}],
  "domain": "Mathematics_Calculus",
  "equation_type": "derivative",
  "complexity": 2,
  "output_type": "real",
  "synthetic": true,
  "metadata": {"source": "code_first_synth", "family": "derivative", "slice": "calculus"}
}
```

- `solution` — canonical runnable code (defines `calculate`); the sandbox
  executes it on `test_cases[].input` and compares against `test_cases[].output`.
- `sympy_exp` — closed-form ground truth when the truth is sympify-expressible.
- `truth_code` — ground-truth *code* for families whose truth is not an AST
  (gcd/lcm/mod-inverse/digit ops); the oracle runs both candidate and truth in
  the same sandbox contract.
- `output` is always a **canonical string** in the published files
  (`'2.5'`, `'10.46+9.42j'`) — the same form the reference metric emits and
  `parse_number` round-trips, so float/complex rows share one loadable schema.
- `truth_code` / `sympy_exp` are `""` when not applicable (schema-uniform);
  check truthiness before using either.
- **`synthetic` config only:** `test_cases[].input` and `metadata` are lossless
  JSON strings. The 26 family pools use different variable keys (`x` vs
  `a,b,m` …) and different metadata keys, which cannot share one Arrow struct;
  solutions also declare exact parameter lists, so keys are never padded.
  `json.loads(tc["input"])` restores the dict exactly. The `split` config
  keeps `input` as a native JSON object (its keys are uniform).

## Verification spine (evidence included)

Every published row satisfies the same accept/reject spine: build-time
family-specific symbolic gates (differentiate-back for integrals,
`checkodesol` + IC for ODEs, exact-integer recomputation for matrix/ntheory)
**and** an independent fresh-input oracle (syntax → Monte Carlo numeric →
identity). Reports are in `data/`:

| Check | Rows | Result |
|---|---|---|
| Gold `solution` vs committed test cases (full competition split) | 22,796 | **100%** pass (0 fail) — `data/verify_report_combined.json` |
| `sympy_exp` truth vs test cases (closed-form subset) | 19,007 | **100%** pass — `data/verify_sympy_report_fixed.json` |
| Representation-form rows (ODE Eq / unevaluated Integral) | 3,789 | verified via gold code (intentional form) |
| Synthetic pool (all 24 families) | 14,701 | oracle-accepted; per-family `data/synthetic/*.report.json` |

## Usage

```python
from datasets import load_dataset

# frozen split: SFT/GRPO + eval (test has expected outputs)
split = load_dataset("{{repo}}", "split")
train, val, test = split["train"], split["validation"], split["test"]

# verified synthetic pool (SFT mixture / curriculum slices)
synth = load_dataset("{{repo}}", "synthetic")

row = train[0]
for tc in row["test_cases"]:
    inputs = json.loads(tc["input"]) if isinstance(tc["input"], str) else tc["input"]
    outputs = run_in_sandbox(row["solution"], inputs)  # your sandbox
    assert outputs_match(outputs, tc["output"])  # canonical-string outputs
```

`outputs_match` / `parse_number` come from the reference evaluation
(`math2code/evaluation/metrics.py` in the repo); `output` strings like
`'10.46+9.42j'` parse back to complex with no loss.

The reference evaluation metric (bootstrap 95% CI, per-problem accuracy) and
the sandbox are open-source at
[github.com/Yasirrazaa/Math2Code](https://github.com/Yasirrazaa/Math2Code)
(`math2code/evaluation/metrics.py`, `math2code/sandbox/`).

## Integrity & contamination

- `data/manifest.json` pins the split's SHA-256 digests; regeneration with
  seed 42 is byte-identical (CI verifies).
- `data/test_ids.txt` lists frozen test ids — grep training data against it
  to prove no contamination.
- The frozen split is NEVER modified; synthetic rows are rejected if their
  LaTeX collides with test/val.
- `data/public_test_new_no_sol_no_out.json` (1,004 rows) is a closed-truth
  generalization probe (98 complex + 500 `augmented_equation` rows) — shipped
  for prediction-only probes, never for accuracy claims.

## Known limitations

- The frozen split's outputs are 100% real; complex-output generalization is
  documented but not eval-validated.
- `train.json` (the 26,846-row competition source) is **not** re-published
  here due to bootcamp licensing — the split JSONs, generator scripts and the
  synthetic pool are the redistributable artifacts (all in this repo).
- Synthetic rows only cover what SymPy can express and verify — this is a
  LaTeX→executable-SymPy mapping dataset, not general mathematical reasoning.

## License

MIT. Derived from a private bootcamp Math2Code competition dataset (see
limitations above); competition rules prohibit re-publishing the raw
`train.json`, which this repo does not do.
