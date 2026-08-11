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

## 4a. Epistemological boundary — what this dataset can and cannot prove

This is a **synthetic, single-scalar-output** training set over SymPy's mathematical surface. It is honest about three things:

1. **Universality is "evaluate any scalar mathematical query over SymPy's surface, deterministically verifiable"** — NOT "answer every math question". The frozen harness accepts exactly one scalar per row. Anything that would require a set, matrix, interval, or proof is out of scope by contract.
2. **Ground truth is sandbox-executed, not judged by an LLM.** Every row's truth is either (a) a closed-form SymPy expression that roundtrips through `sympify`, (b) a `truth_code` snippet executed in a sandboxed `python` process, or (c) `.evalf()` on a known unevaluated expression. There is no fourth class. Integer truths never accept float drift; complex truths parse via `parse_number` which produces clean `'a-bj'` strings (the `format_output` path uses `{c.imag:+}j` to avoid the legacy `'a+-bj'` round-trip edge case).
3. **The synthetic pool can only teach what SymPy can express.** Everything in this dataset is, by construction, computable by a small Python program with `sympy` (and possibly `mpmath` for special functions via `.evalf()`). Problems requiring human reasoning, multi-step proof construction, or mathematical creativity are not representable — and we will not pretend otherwise. The model learns **mapping LaTeX → executable SymPy/Python**, not mathematics from first principles. That is a real, useful, narrower thing.

The portfolio value of this dataset is therefore NOT "math AI" — it is **measured, reproducible LaTeX→code over a verifiable subset of SymPy**, with deterministic acceptance gates and frozen contamination checks. Every metric claimed in the repo can be regenerated from a single seed and the SHA-256 manifests in `data/split/`.

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

## 8. Blueprint audit (code-first synthesizer) — validated with corrections

**Status: implementation complete for all eight strategy slices.**
`src/math2code/data/synthesizer/` (printer + sampler + core + derivative/
integral/function-vocab/ODE/multivariate/sequences/geometry/edge/numtheory
families + verify) is built; `make synth*` generates oracle-verified rows into
`data/synthetic/` (7,450-row pool), `make mixture` builds the 65/35 mixture
(21.5k rows, all slices). The `truth_code` contract extends the oracle to
families whose truth is code (gcd/lcm). 105 tests pass. Remaining: nothing
blocking — user-run W3 API baselines + GPU training consume this dataset.

Verdict: the code-first synthesis + notation-mutation + family-specific
verification architecture is correct and is the implementation path. Every
claim was checked against the actual code and the installed sympy 1.14.

### Confirmed good

- **Code-first synthesis is right.** The current `data/generate.py` is LLM-only
  (Groq llama3-70b via instructor) and emits rows with **no `sympy_exp` and no
  `test_cases`** — so `curate()` cannot oracle-verify them (falls back to
  “runs without error”). Code-first construction fixes this by construction.
- **Printer subclassing works** — `_print_exp`, `_print_Derivative`,
  `_print_Integral` dispatch correctly on `LatexPrinter` subclasses.
- **SymPy 1.14 already renders the full P0/P1 vocabulary** out of the box:
  `\left|{x}\right|`, `\sqrt[3]{x}`, `x!`, `\begin{cases}`, `\operatorname{asin}`,
  `\frac{\log{x}}{\log{2}}`. Notation mutation is therefore *variant selection*
  on top of a working base renderer, not new rendering work.
- **Family-specific verification is required** — numeric-only checking genuinely
  fails for indefinite integrals (constant of integration) and ODEs
  (equivalent solution forms). Differentiate-back and residual-substitution
  are the right verifiers.
- **Complex outputs are already supported end-to-end** by the stack
  (`TestCase.output: Number | str`, sandbox `_format` prints `re+imj`,
  `parse_number` parses it) — the complex slice needs no metric change.

### Corrections (each verified empirically)

1. **`sp.latex(expr, printer=P)` is broken in sympy 1.14** — the `printer`
   setting no longer exists (`TypeError: Unknown setting 'printer'`). Use
   `P(settings).doprint(expr)` directly.
2. **`_print_Rational` footguns integers** — `sp.Integer` subclasses
   `sp.Rational`, so overriding `_print_Rational` without a `q == 1` guard
   renders `2` as `\frac{2}{1}` (observed). Guard integers first.
3. **Never merge synthetic rows into `data/split/train.jsonl`.** The split is
   frozen: manifest SHA-256s, byte-identical regeneration, CI integrity check.
   Merging breaks all three, and `make splits` regenerates train.jsonl from
   `data/train.json` anyway — any merge is wiped on the next regen, and
   re-running the splitter would silently shift the benchmark. Correct design:
   synthetic rows live in `data/synthetic/`; `train.py`/`grpo.py` consume
   `competition_train + synthetic_train` as a documented mixture at load time;
   the frozen files never move. Contamination guard: the synthesizer rejects
   any row whose LaTeX collides with `test_ids.txt`/val LaTeX.
4. **The metric compares ONE scalar per case** (`parse_number` → complex,
   isclose). Full-matrix/tuple/set outputs do not fit the frozen metric.
   “Frobenius norm of difference” is a *verifier* concept, not a metric
   concept. Keep matrix rows scalar-output (determinant, trace, norm, sum)
   or extend the metric later. Vector/tuple output rows are deferred.
5. **Determinism**: no `random.random()` inside the printer. Style variant
   selection must be a pure function of (row seed, expression) so regeneration
   is byte-identical — the project already proves this bar with the splits.
6. **Keep the LLM generator as a secondary source**, but it must go through
   the code-first oracle to fill `sympy_exp` + `test_cases` before curation;
   never curate LLM rows on “runs without error” alone.
7. **Indefinite integrals use the C=0 convention** (competition contract:
   outputs are numeric, evaluated at inputs). Differentiate-back verifies the
   *solution code*; numeric cases come from the canonical C=0 antiderivative.
   Watch `sp.integrate` returning `Piecewise` (e.g. `∫1/x dx = log|x|`) —
   restrict families or take the continuous antiderivative.
8. **Sampler extension is needed, seeded from the existing `jitter_inputs`**
   (which already preserves int-ness for diophantine rows and couples
   `x`/`x_val`). Add: pole avoidance (denominator-root exclusion radius),
   branch-cut-aware complex sampling, log-scale magnitudes, correlated
   multivariable inputs. Keep the 5-case contract for rows; use 20+ fresh
   points for oracle verification.

### Implementation order (revised)

1. `data/synthesizer/` core: seeded `DiverseLatexPrinter` + variant selector,
   expression→ground-truth→code→cases pipeline, MathCodePair emission.
2. Family modules: P0 function vocabulary first, then derivative/integral
   (differentiate-back verifier), then ODEs (residual + IC).
3. Domain-aware sampler with pole/branch/magnitude handling.
4. Accept/reject runner → `data/synthetic/` (accepted/rejected/manifest/report).
5. Training-time mixture in `train.py`/`grpo.py` (competition + synthetic).
6. Metadata extension → richer `analyze_results.py` slices.

The remainder of this doc (sections 3–7) is the family catalog, verification
spine, composition target, and roadmap — unchanged by the audit.

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



### Deep Analysis & Engineering Blueprint: The Data Generalization Flywheel

First, I must commend the `DATA_STRATEGY.md` document. It is an exceptionally sharp, measured, and ruthless audit of the current dataset. Your identification of **"diversity theater"** (14 balanced labels masking only 7 underlying functions and separable multivariables) is exactly the kind of epistemological rigor that separates senior ML engineers from script-kitties. You have correctly diagnosed that the model is currently learning to interpolate within a narrow template family, not learning mathematical reasoning.

To make the model generalize to a wide set of mathematical expressions, we must transition the data pipeline from **LLM-Prompted Generation** (which drifts and hallucinates) to **Programmatic Code-First Synthesis** with **Semantic Notation Mutation**.

Here is the Staff-Level Engineering Blueprint to extend your data generation and verification pipeline, directly addressing the gaps identified in your strategy document.

---

### Phase 1: The Code-First Synthesizer Architecture
Your current `generate.py` relies on Groq/Llama to generate LaTeX and Code. This is the root cause of the narrow vocabulary. LLMs default to the most common patterns in their training data (basic trig and polynomials).

**The Fix: SymPy AST Construction $\rightarrow$ Custom Rendering $\rightarrow$ Test Case Sampling.**

Instead of asking an LLM for math, we write Python scripts that construct SymPy Abstract Syntax Trees (ASTs) using the expanded P0/P1 vocabulary. The AST *is* the ground truth code.

#### 1. The `MathSynthesizer` Engine
Create a modular generator in `src/math2code/data/synthesizer/` that builds expressions programmatically.
```python
# src/math2code/data/synthesizer/calculus.py
import sympy as sp
import random

def generate_indefinite_integral(max_depth: int = 3) -> dict:
    # 1. Build the integrand (AST)
    x = sp.Symbol('x')
    # Mix of P0 functions: inverse trig, hyperbolic, cross-terms
    integrand = sp.sinh(x) * sp.exp(x) + sp.atan(x) 
    
    # 2. Compute Ground Truth (The Code)
    integral_expr = sp.integrate(integrand, x)
    
    # 3. Generate Test Cases (Domain-Aware Sampling)
    # Avoid poles/branch cuts by sampling in safe complex/real regions
    test_cases = sample_safe_inputs(integral_expr, n=10)
    
    return {
        "sympy_exp": str(integral_expr),
        "solution": format_function(integral_expr),
        "test_cases": test_cases,
        "equation_type": "integration_indefinite"
    }
```

#### 2. Semantic Notation Mutation (The "Diversity" Engine)
SymPy’s default `sp.latex()` is rigid. It will always render `exp(x)` as `e^{x}` or `\exp(x)` depending on settings, but it won't randomly vary notation. To achieve the **P0 Notation Diversity** you outlined, you must subclass SymPy's `LatexPrinter`.

```python
from sympy.printing.latex import LatexPrinter
import random

class DiverseLatexPrinter(LatexPrinter):
    def _print_exp(self, expr):
        # 50% chance to render as e^{...}, 50% as \exp(...)
        if random.random() > 0.5:
            return f"e^{{{self._print(expr.args[0])}}}"
        return f"\\exp\\left({self._print(expr.args[0])}\\right)"
        
    def _print_Derivative(self, expr):
        # Randomly choose between Leibniz (d/dx), Lagrange (y'), or partial
        # ... implementation logic ...
```
This guarantees that the underlying mathematical truth (the AST) remains identical, but the LaTeX prompt varies wildly, forcing the model to learn that `\frac{dy}{dx}` and `y'` map to the same SymPy `Derivative` object.

---

### Phase 2: Extending the Verification Spine (`oracle.py`)
Your current `oracle.py` relies on `numeric_check` (Monte Carlo `isclose`) and `identity_check`. This fails for the P1 families you want to add. You must implement a **Family-Specific Verifier Registry** (Strategy Pattern).

#### 1. Indefinite Integrals (The Constant of Integration Problem)
Numeric checking fails for indefinite integrals because the model might output `x**2 + 5` while the ground truth is `x**2 + C`.
*   **The Verifier:** Differentiate the candidate's output and compare it to the integrand.
    ```python
    def verify_indefinite_integral(candidate_code, integrand_expr):
        # Execute candidate to get the expression
        cand_expr = execute_and_extract_expr(candidate_code)
        # d/dx (Candidate) must equal Integrand
        diff = sp.simplify(sp.diff(cand_expr, x) - integrand_expr)
        return diff == 0
    ```

#### 2. Ordinary Differential Equations (ODEs)
Numeric checking is insufficient for ODEs because you must verify the *dynamics* and the *Initial Conditions (ICs)*.
*   **The Verifier (Residual Substitution):**
    1. Parse the candidate solution $y(x)$.
    2. Substitute $y(x)$ and its derivatives back into the original ODE expression.
    3. Verify `sp.simplify(ODE_residual) == 0`.
    4. **IC Check:** Numerically evaluate the candidate at $x_0$ and verify it matches the required initial condition $y(x_0) = y_0$.

#### 3. Matrices and Linear Algebra
*   **The Verifier:** Do not use `isclose` on string representations. Parse the candidate output into a SymPy `Matrix` object. Compute the Frobenius norm of the difference: `||A_candidate - A_truth||_F < tolerance`.

#### 4. Exact Integer Arithmetic (Factorials, GCD, Modulo)
*   **The Verifier:** Floating-point `isclose` is dangerous here due to precision limits on large factorials. Use **exact integer equality** (`==`). Ensure the sandbox returns Python `int` types, not floats.

---

### Phase 3: The Domain-Aware Test Case Sampler
Your `DATA_STRATEGY.md` correctly identifies that the current summation ranges are tiny and fixed, and multivariable problems lack cross-terms. The model needs to see **edge cases** to generalize.

You must upgrade the `test_cases` generator in `schemas.py` to be **Domain-Aware**:
1.  **Pole Avoidance:** For rational functions (e.g., $\frac{1}{x-2}$), the sampler must analytically find the roots of the denominator and exclude a safety radius (e.g., $\pm 0.1$) around them when generating random inputs.
2.  **Branch Cut Awareness:** For complex functions like `log(z)` or `sqrt(z)`, the sampler must generate inputs from both the principal branch and the negative real axis to ensure the model handles complex phases correctly.
3.  **Magnitude Scaling:** Generate inputs across logarithmic scales ($10^{-8}$ to $10^{8}$) to test numeric stability under `isclose`.
4.  **Cross-Term Coupling:** For multivariable functions, ensure the sampler generates correlated inputs (e.g., $x$ and $y$ where $x \approx y$) to test denominators like $x^2 - y^2$ for catastrophic cancellation.

---

### Phase 4: The Adversarial "Hard Negative" Flywheel
To make the GRPO training truly robust, the pipeline should generate "near-miss" data. This is the ultimate generalization test.

1.  **Notation Traps:** Generate LaTeX with ambiguous or non-standard notation (e.g., implicit multiplication `xy` vs `x(y)`, or `\log x` without a base). The model must learn to infer standard SymPy conventions or ask for clarification (if using TIR).
2.  **Domain Violations:** Generate prompts that ask for `sqrt(-1)` in a context where the model might default to real numbers. The model must learn to output `1.0j` (complex) or handle the exception gracefully.
3.  **The "Augmented" Family:** Your strategy notes the public test holds 500 `augmented_equation` rows (rational $\times$ exp-decay) that are absent from train. **Synthesize exactly this family** using the Code-First engine, but hold back 10% as a "validation probe" to ensure you haven't accidentally contaminated the benchmark.

---

### Implementation Roadmap (Next 2 Weeks)

1.  **Days 1-3: The Synthesizer Core.** Build the `MathSynthesizer` base class and implement the `DiverseLatexPrinter`. Generate 1,000 samples of P0 Function Vocabulary (inverse trig, hyperbolic, factorials) and verify them with the existing `oracle.py`.
2.  **Days 4-6: The Verifier Registry.** Refactor `oracle.py` to use a Strategy Pattern. Implement `IntegralVerifier`, `ODEVerifier`, and `MatrixVerifier`.
3.  **Days 7-10: Calculus & ODE Generation.** Implement the programmatic generators for Definite/Indefinite Integrals and 1st/2nd Order ODEs. Run the curation pipeline (`curate.py`) to filter out generation bugs.
4.  **Days 11-14: Multivariate Coupling & Edge Cases.** Generate the cross-term dataset ($x \cdot y$, $e^{x+y}$). Implement the Domain-Aware Sampler. Merge the accepted rows into `data/split/train.jsonl` and re-run the `make eval-gold` sanity check to ensure the new data doesn't break the harness.

By executing this blueprint, you transition Math2Code from a model that memorizes 7 trigonometric templates to a **general-purpose mathematical compiler** that understands the deep structural isomorphism between LaTeX notation and SymPy execution. This is the exact methodology used to build frontier reasoning models.
