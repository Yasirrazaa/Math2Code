# Family Development Contract (for parallel synthesis agents)

You are writing ONE new family module + ONE test file for the code-first
synthesizer. Read this contract and the referenced template files completely
before writing anything.

## Ground rules

- **Create exactly two files, nothing else:**
  - `src/math2code/data/synthesizer/families/<family>.py`
  - `tests/test_<family>.py`
- **NEVER edit** (read-only): `core.py`, `printer.py`, `sampler.py`,
  `verify.py`, `__init__.py`, `oracle.py`, `schemas.py`, any existing family
  file, any `scripts/*`, `Makefile`, `configs/*`, `tests/conftest.py`,
  docs. If you need a shared change, note it in your final report instead.
- **Never** run `git` commands. **Never** run the full test suite or lint
  tools that fail on unrelated files. Only run your own test file.
- Use the interpreter `/tmp/m2c_venv/bin/python` (dev env with math2code
  installed as editable; ruff/mypy/pytest live there too):
  - tests: `/tmp/m2c_venv/bin/python -m pytest tests/test_<family>.py -q`
  - lint: `/tmp/m2c_venv/bin/ruff check <file>`
  - types: `/tmp/m2c_venv/bin/mypy <file>`

## Read first (in this order)

1. `src/math2code/data/synthesizer/core.py` — `SynthFamily`, `_build_pair`,
   `solution_code`, `roundtrips`, `complexity_of` (the contract you hook into).
2. `src/math2code/data/synthesizer/families/derivative.py` — template for a
   standard evaluation family (problem AST → result AST → `_build_pair`).
3. `src/math2code/data/synthesizer/families/numtheory.py` — template for a
   `truth_code` family (parameterized truths that are NOT sympify-expressible).
4. `src/math2code/data/synthesizer/families/ode.py` — ODE generation helpers
   (`checkodesol` gate, solution structure) — import from it read-only if
   useful.
5. `src/math2code/schemas.py` — the `MathCodePair` / `TestCase` schema.
6. `src/math2code/data/synthesizer/sampler.py` — `sample_inputs` options.

## The `_build_pair` contract (exact signature)

```python
def _build_pair(self, task_id, problem_expr, result_expr, variables,
                seed, n_variants=2, n_cases=5, meta=None, sample_kwargs=None,
                equation_type=None, custom_code=None,
                latex_override=None) -> list[MathCodePair]
```

- `problem_expr` — rendered to LaTeX (the *question* surface).
- `result_expr` — the ground truth AST. Must `roundtrips()` (gate 1, automatic)
  unless you use `custom_code` or `latex_override` (see patterns below).
- `variables` — input variable symbols (may be `[]` for constant-output rows).
- `seed` — pass `int_seed(f"{family}:{base_seed}:{i}")` per object.
- `sample_kwargs` — overrides: `ints_only`, `low`, `high`, `log_scale`,
  `pole_margin`, `allow_complex`.
- `meta` — **must include** `"slice": "<your-slice-tag>"` (see below), plus any
  useful extra metadata (`vocab`, `coefficient_kind`, `n_vars`...).
- `latex_override` — NEW: a list of exact LaTeX strings that replace
  `render_variants` (for queries whose notation isn't derivable from a single
  AST, e.g. `\det(...)`, `\operatorname{coeff}_{x^{k}}`, `\mathbb{E}[X]`).
  When used, `problem_expr` is still required for `solution_code`? No —
  `result_expr` is what matters; pass a dummy fine AST for `problem_expr` (it
  is unused for rendering when `latex_override` is given).

**Rows returned:** `_build_pair` returns one `MathCodePair` per variant. Your
`generate()` loops creating objects and collects them; cap attempts
(`while len(out) < count * n_variants and i < count * 60:`) so gates can
reject. `generate` returns exactly `count * n_variants` rows.

## The five working patterns (choose the right one)

### A. Standard evaluation (derivative.py style)
`problem` = the question AST (e.g. `sp.Derivative(f, x)`), `result` = the
transformed AST (`sp.diff(f, x)`). Variables sampled; oracle numeric.

### B. Constant-output rows (limits, concrete sums, dets, stats)
`result_expr` is a NUMBER (`sp.Integer`, `sp.Rational`, `sp.Float`),
`variables = []`. `sample_inputs` yields `n_cases` empty-input cases;
`solution_code` emits `def calculate(): return float(_expr)`. The oracle
handles no-arg functions (`identity_check`). Always gate on the value being
finite: `result.is_finite` — reject `sp.oo`, `sp.nan`, unevaluated objects
(`result.is_number` may be False for unevaluated — check `sp.N(result)` works).

### C. Parameterized truth_code (numtheory.py style)
Truth depends on input values and is not a sympify-expressible AST
(`math.gcd(a,b)`). Use `custom_code=(solution, truth)` — both full
`calculate(...)` functions. Must set `sample_kwargs={"ints_only": True, ...}`
and `n_variants=1`. `_gate` documents the identity.

### D. `.evalf()` solution/truth (unevaluated special functions)
If `sp.<func>(val)` stays UNEVALUATED (e.g. `sp.erf(1)`, `sp.besselj(1,2)`)
then `float(sp.sympify("erf(1)"))` RAISES — the standard `solution_code` would
crash the sandbox. Use `custom_code` where BOTH solution and truth call
`.evalf()`, e.g.:
```python
_code = "import sympy as sp\n_expr = sp.sympify('erf(1)')\ndef calculate():\n    return float(_expr.evalf())\n"
```
Prefer exact forms when they exist (`sp.gamma(5)` → 24, `sp.gamma(Rational(5,2))`
→ `3*sqrt(pi)/4`, `sp.legendre(3,2)` → 17) and use pattern A for those.

### E. latex_override (named-operator / invented notation queries)
If the query needs notation SymPy won't render (coefficient-of-x^k, Vieta,
`\det(A)` with concrete pmatrix, `\mathbb{E}[X]`, `|A \cap B|`), pass exact
LaTeX strings. Rows are constant-output (pattern B) or evaluation.

## Gate discipline (the epistemology — do not skip)

Every family MUST:
1. Override `_gate(self, result_expr) -> str` returning a description of the
   asserted identity ("checkodesol(eq, sol) is True", "limit is finite",
   "diff(F, x) == f", "subst residual == 0").
2. **Skip rows where the gate fails** (continue the loop). Never emit a row
   the family cannot prove.
3. Only emit rows whose ground truth evaluates to a FINITE number (the
   sampler does this automatically via `eval_gate` unless you use
   `custom_code`, in which case the family guarantees definedness).
4. Avoid huge outputs: keep inputs bounded (`ints_only` ranges, sum lengths
   ≤ 20, matrix size ≤ 4, powers with exp ≤ ~8) so no AST/text blowup.

## Slice tags (metadata["slice"]) — exact mapping

Use ONLY these existing/new tags (mixture caps key off them):

- `differential_c1` family → `"ode"`
- `summation` → `"summation"`   (new cap)
- `limits` → `"limits"`         (new cap)
- `series_coeff` → `"series"`   (new cap)
- `elementary_ext`/`complex_eval` → `"vocab"`
- `polynomial_invariants` → `"polynomial"` (new cap)
- `matrix_scalars` → `"matrix"` (new cap)
- `ntheory_ext` → `"numtheory"`
- `combinatorics` → `"combinatorics"` (new cap)
- `geometry_ext` → `"geometry"`
- **gated** families (`special_functions`, `stats_moments`,
  `sets_cardinality`, `solving_scalarized`) → set `meta["gated"] = True` AND a
  slice tag matching their domain (e.g. `"special"`, `"stats"`, `"sets"`,
  `"algebraic"`). Gated rows are EXCLUDED from the default RL mixture by the
  builder.

## Your family's tests (tests/test_<family>.py)

Write ≥4 tests, each fast (<10 s), using `from math2code.data.synthesizer.families.<family> import <Family>`:

1. **Determinism**: `generate(seed=42, prefix="t", count=5)` twice →
   identical `task_id` + `latex_expression` sequences.
2. **Contract shape**: every row is a `MathCodePair` with non-empty
   `latex_expression`, runnable `solution`, 5 test cases, correct
   `equation_type`/`domain`, `synthetic is True`, metadata `slice` correct,
   outputs parseable by `math2code.evaluation.metrics.parse_number`.
3. **Gate holds**: for every generated row, re-assert the family's symbolic
   identity in the test (e.g. `sp.diff(row result)` — where feasible; for
   truth_code families, run `execute_code(row.truth_code, inputs)` and check
   the output equals the committed `test_cases` output).
4. **Oracle verification**: generate ~3 objects and run
   `oracle_verify(row, row.solution, pool=pool)` (create a small
   `SandboxPool(n_workers=2)`) — all pass. Keep to 1-2 rows to stay fast.
5. One robustness test (finitude of outputs / no empty rows).

Do not add slow tests (no `@pytest.mark.slow`) — keep the default suite fast.

## Style gates

- Match the existing family style exactly: module docstring, `from __future__
  import annotations`, typed signatures, `_rand_ints`-style local helpers,
  `int_seed` usage.
- ruff + mypy clean on YOUR files only: `/tmp/m2c_venv/bin/ruff check
  src/math2code/data/synthesizer/families/<family>.py tests/test_<family>.py`
  and `/tmp/m2c_venv/bin/mypy
  src/math2code/data/synthesizer/families/<family>.py`.
- Run your tests: `/tmp/m2c_venv/bin/python -m pytest tests/test_<family>.py -q`.

## Deliverable report (final message)

Summarize: file paths created, oracle accept rate you measured, any shared
changes you needed (request only, don't make), and test/lint results.