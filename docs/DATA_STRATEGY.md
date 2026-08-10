# Data strategy: distribution analysis + verified synthetic data roadmap

Status: analysis (measured against committed files). Implementation follows.

## 1. Measured reality of the current distribution

All numbers below are computed from the committed `data/split/*` files.

### 1.1 Label diversity is real, structural diversity is not

The 14 `equation_type` labels are balanced by construction (~1,800 rows each
of the 13 real types, 31 Geometry) — but the *underlying structure* they
encode is narrow:

| Structural family | Types it covers | Reality check (measured) |
|---|---|---|
| Symbolic arithmetic (poly, exponent, diophantine, sum-prefix) | algebraic, multivariable, exponential, diophantine, summation | Parseable by SymPy `parse_latex` at ~100% — the model adds nothing here |
| Rational functions | fractional, rational | Also ~100% parseable |
| One function + linear transform | logrithmic, trigonometric, exponential_decay | `log, cos, sin, sec, tan, csc, cot` + float `e^{-kx}` |
| First-order linear ODEs | differential | `y' + p(x)y = 0`, p ∈ {linear in x, cos, tan, sin}; 163/1,830 are 2nd-order |
| Python-repr calculus | integration, derivative (3,664 rows, 16.6%) | `\mathtt{\text{Integral(...)}}` / `\mathtt{\text{Derivative(...)}}` — **SymPy repr, not math notation** |
| Everything else | — | **0.00% of train** |

### 1.2 The full function vocabulary of 22,002 training rows

```text
log, tan, cos, sin, sec, csc, cot        (7 functions + pi as a constant)
```

**That is the entire mathematical universe.** Zero rows contain: `\exp`/`e^{}`
as a function call, inverse trig, hyperbolic, `abs`/`|·|`, floor/ceil, factorial,
binomial, `log_b`, nth roots, gcd/lcm/mod, min/max, piecewise, complex ops.

### 1.3 Notation presence (train-wide, measured)

| Notation | Rows | Share |
|---|---|---|
| `\sum` | 1,832 | 8.33% |
| `\frac{d}{dx}` ODEs | 1,667 | 7.58% |
| repr-wrapped `Integral/Derivative` | 3,664 | 16.65% |
| `\int` | **0** | 0% |
| `\lim`, `\prod`, `\partial`, `\sqrt[n]`, `\binom`, `\log_b`, `\ln`, `\cdot`, `\operatorname` | **0** | 0% |

### 1.4 Other measured skews

- **Multivariable coupling: 0 cross-terms in 1,806 rows** — every "multivariable"
  problem is a separable sum of univariate polynomials (`x^3 + y^2 + z`). The model
  never learns `x*y`, `sin(x)*cos(y)`, `e^{x+y}`, or a rational in 2+ vars.
- **Complexity:** c2 = 11,197 (50.9%), c5 = 1,830 (8.3%) — skewed easy.
- **Coefficients:** 16.9% of rows carry 15-significant-digit float coefficients
  (the `exponential_decay` / rational style); the rest are small integers.
- **Outputs:** 100% real in train/val/test; the public probe holds 98 complex rows
  with **zero** complex examples in train.
- **Geometry:** 31 rows (0.14%), dominated by trivial volume-like formulas (`a^3`).
- **Public-test OOD family `augmented_equation` (500 rows):** rational × exp-decay
  products with float coefficients — **absent from train entirely**.
- **Summation ranges are tiny fixed** (`\sum_{x=1}^{3}`, `\sum_{z=1}^{5}`).
- **Sums/prefixes use `-8281 + \sum ...` anti-shape** — the constant is *added to
  the sum*, i.e. `constant + Σ` with the sum at the end, an awkward render.

### 1.5 Verdict

The 14 labels are **diversity theater**: balanced counts over ~6 narrow template
families. A model trained on this will interpolate well inside each family and
generalize poorly to anything outside the 7-function vocabulary, real calculus
notation, non-separable multivariables, richer ODEs, or complex outputs.

Note: 100% of the competition data is itself flagged `synthetic: True`. Synthetic
data is not a departure from this dataset — it *is* the dataset. The goal is
**better** synthetic data: broader, notation-diverse, and oracle-verified.

## 2. Generalization dimensions for LaTeX → executable code

1. **Notation forms** — the same math rendered many ways (`\int`, `\int_a^b`,
   `\frac{d}{dx}`, `y'`, `\partial`, implicit vs `\cdot`, `\left(\right)`,
   `\ln` vs `\log`, `e^{x}` vs `\exp(x)`).
2. **Function vocabulary** — breadth beyond {log, tan, cos, sin, sec, csc, cot}.
3. **Operator / structural depth** — nesting, composition, multi-operator.
4. **Problem classes** — equations, systems, sequences, geometry, number theory.
5. **Data types** — real, integer-exact, complex, boolean/predicate outputs.
6. **Coefficient realism** — int, rational, float, symbolic parameter.
7. **Complexity / depth** — operator counts, nested args, many variables.
8. **Multivariate coupling** — cross-terms, shared symbols, joint domains.
9. **Edge cases** — singularities, near-zero denominators, boundary inputs,
   tiny/huge magnitudes.
10. **Output contracts** — scalar, vector, tuple, piecewise branches.

## 3. Prioritized addition catalog (value × verifiability)

### P0 — Notation diversity (cheapest, biggest generalization lever)

Render each math object in multiple LaTeX forms so the model learns notation is
surface and math is deep:

- `\int x^2 \, dx` and `\int_{0}^{1} x^2 \, dx` (currently 0% of train)
- `\frac{d}{dx}\left(...\right)`, `y'`, `\frac{dy}{dx}`, `\partial`
- `e^{x}`, `\exp(x)`; `\log(x)`, `\ln(x)`; `\sqrt[n]{x}`, `x^{1/n}`
- `\cdot` explicit multiplication; `\left( \right)`; `\binom{n}{k}`; `n!`
- `\operatorname{Re}, \operatorname{Im}, \arg, \bar{z}` for complex
- `\begin{cases}` for piecewise

### P0 — Function vocabulary (verifiable: all direct SymPy numerics)

| Family | Examples | Verify |
|---|---|---|
| exp as function | `e^{2x+1}`, `\exp(x/3)` | numeric vs SymPy |
| inverse trig | `\arcsin, \arccos, \arctan` (+ `atan2`) | numeric (domain-bounded inputs) |
| hyperbolic | `\sinh, \cosh, \tanh` (+ sech/csch/coth) | numeric |
| abs / sign | `|x|`, `\lvert x\rvert`, `\operatorname{sgn}` | numeric incl. boundaries |
| floor / ceil | `\lfloor x\rfloor`, `\lceil x\rceil` | numeric incl. integer args |
| factorial family | `n!`, `\binom{n}{k}` | exact integer |
| nth roots | `\sqrt[n]{x}` | numeric (positive/odd-domain aware) |
| log base | `\log_{b}(x)` | change-of-base verify |
| min / max | `\min(a,b,c)`, `\max(...)` | numeric incl. ties |
| integer ops | `\gcd, \operatorname{lcm}, a \bmod b` | exact integer |
| complex ops | `\operatorname{Re}, \operatorname{Im}, |z|, \arg, \bar{z}` | complex isclose (exercises `parse_number`) |

### P0 — Multivariate coupling

- Cross-terms: `x*y`, `x*y*z`, `x^y`, `e^{x+y}`, `sin(x)*cos(y)`
- Rationals in 2–3 vars with joint denominators
- Coupled symbolic parameters (`a*x + b*y` where a,b are parameters, x,y inputs)

### P1 — Problem kinds (all verifiable)

- **Definite integrals** (numeric) and **indefinite** (verify by
  differentiate-back: `simplify(diff(result, x) - integrand) == 0`)
- **Limits** (`\lim_{x\to a}`) — numeric from both sides
- **Partial derivatives** (`\partial`)
- **ODE breadth**: 2nd-order, non-homogeneous, with initial conditions
  (verify: residual substitution + IC check)
- **Sequences**: arithmetic/geometric nth-term, partial sums
- **Geometry**: real formula set — area, perimeter, volume, surface area,
  coordinate geometry (distance, midpoint, slope)
- **Number theory**: divisibility predicates, modular arithmetic, prime/composite
- **Systems of linear equations** (2×2/3×3) — output contract: tuple
- **Matrix basics**: determinant, trace, transpose, matvec (SymPy-verifiable)
- **`augmented_equation` family**: rational × exp-decay products with float
  coefficients — legitimate train addition; the public probe remains the
  generalization check for *unseen instances of a now-seen family*

### P1 — Distribution shape

- Rebalance complexity toward c3–c5 (model needs deep compositions; c2 = 51%)
- Coefficient realism: int/rational majority + a float-coefficient robustness slice
- More hard 1-variable problems (real-world common case)
- A complex-output slice (trains the `parse_number` complex path)

### P2 — Edge-case / robustness slice (small, high value)

- Singularity-adjacent: `1/(x-2)` with input sampling avoiding the pole
- Domain-constrained: `log(x), x>0`; `sqrt(x), x≥0` — teaches input guards
- Magnitudes: 1e-8 .. 1e12 — tests numeric stability under `isclose`
- Degenerate: `x=0`, even powers of negative bases, `0^0` avoidance
- Nested composition: `log(sin(x^2)+1)`, `sqrt(1+tan(x)^2)`, `e^{cos x}`

## 4. What NOT to add (now)

- **Output contracts that break the schema**: sets of solutions, inequality
  ranges, word problems — requires a schema change; defer.
- **Non-verifiable / LLM-only labels** without a SymPy oracle.
- **More easy algebra** — saturated (`parse_latex` ≈ 100%).
- **Anything derived from the frozen test/val expressions** — the generator must
  never be tuned against the benchmark.

## 5. Verification spine (accept/reject)

Every generated row passes the same spine; a row is **accepted only when the
oracle and the sandbox agree on fresh inputs**, never on the generation inputs.

| Family | Truth source | Acceptance check |
|---|---|---|
| Indefinite integral | template antiderivative | `simplify(diff(result) − integrand) ≡ 0` + fresh numeric |
| Definite integral | numeric `integrate` | isclose on fresh inputs |
| Derivative / partial | SymPy `diff` of template | numeric + fresh inputs |
| Limit | SymPy `limit` | numeric from both sides |
| ODE | template solution | residual substitution ≈ 0 + IC check |
| Inverse trig / hyperbolic / abs / floor | SymPy direct | numeric incl. boundaries |
| factorial / binomial / gcd / lcm / mod | exact integer | exact integer equality |
| log base | change-of-base | numeric |
| min / max | direct | numeric incl. ties |
| complex ops | SymPy | complex isclose (`parse_number` path) |
| piecewise | case evaluation | every branch sampled |
| sequences | closed form | index sweep |
| systems | solve + residual | residual on fresh RHS |
| geometry | formulas | numeric vs template |
| matrices | SymPy | numeric |

Structural filters (reject): duplicate LaTeX against train/val/test, unsafe AST,
timeout/OOM, > complexity cap, degenerate (all-zero / non-finite everywhere).

Every accepted row: **5 committed cases + ≥20 fresh oracle cases**, plus metadata
(`functions`, `operators`, `n_vars`, `cross_terms`, `coefficient_kind`,
`output_type`, `notation_form`) so curriculum targeting and slice analysis stay
measurable — mirroring the existing `analyze_results.py` breakdown but richer.

## 6. Composition target (synthetic supplement, ~10–15k rows)

| Slice | Share | Purpose |
|---|---|---|
| Calculus family expansion (def/indef integrals, partials, limits) | 30% | attack the 0% `parse_latex` slice |
| Function-vocabulary expansion (P0 list) | 20% | generalization breadth |
| Multivariate coupling | 15% | fix the separable-only hole |
| ODE breadth (2nd-order, ICs) | 10% | richer dynamics |
| Sequences + geometry + number theory | 10% | new problem classes |
| `augmented_equation` family | 10% | match the public-test family |
| Edge-case / robustness | 5% | stability |

Mixture with competition train (~22k): **~65/35 competition:synthetic**,
both retained, frozen test/val untouched.

## 7. Implementation order

1. `data/synth/` programmatic generator: P0 notation forms × P0 function
   vocabulary, derivative/definite-integral families first (highest value,
   simplest verification)
2. Accept/reject runner with per-family verifiers + structural filters →
   `accepted.jsonl` / `rejected.jsonl` / manifest / report
3. Distribution-shape pass: complexity rebalance, coefficient kinds, cross-terms
4. Richer metadata → extend `analyze_results.py` to slice by functions/notation
5. Merge accepted rows into train; re-run gold + quick baseline sanity; SFT/GRPO
   mixtures documented in `configs/`

## Appendix: evidence commands

```bash
# function vocabulary, notation presence, cross-term counts
python - <<'EOF'
import json, re, collections
from pathlib import Path
train = [json.loads(l) for l in Path('data/split/train.jsonl').open()]
LX = [r['latex_expression'] for r in train]
print(collections.Counter(m for l in LX for m in re.findall(r'\\([a-zA-Z]+)', l)))
EOF
```
