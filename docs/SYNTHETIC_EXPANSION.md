# Synthetic Expansion Plan — SymPy-Surface Catalog (Verifiable-Scalar Only)

**Status:** ✅ All 21 families **implemented & oracle-verified** (2026-08-11). See §Implemented Status at bottom.
**Scope:** Maximize dataset diversity across SymPy's module surface while keeping every row deterministically verifiable in the sandbox without an LLM judge. No compute/cost constraint on generation; RL-mixture composition policy remains governed by eval alignment and contract safety.

---

## 0. The Contract (immutable)

Every generated row must satisfy the existing `MathCodePair` contract:

- **Query:** LaTeX expression (or expression-with-operator) → the model emits `calculate(inputs)` sandbox code.
- **Output:** exactly **one scalar per test case** — `int`, `float`, exact `Rational`, or `'re+imj'` string. The frozen `parse_number` harness accepts nothing else.
- **Verification:** accept only when the oracle and the sandbox candidate agree on **fresh inputs** (5 committed cases + ≥20 fresh oracle cases per row).
- **No LLM judges, no float-drift acceptance for integer truths, no unevaluated SymPy objects as truths.**

Consequence: *every family is a scalarized query over a mathematical object* — we ask the invariant, the value, the count, or the coefficient, never the object itself.

---

## 1. Diversity Axes (the design space)

| Axis | Range |
|---|---|
| 1. Domain | 21 families across 14 SymPy modules (catalog below) |
| 2. Operator vocabulary | full elementary set (trig, inverse trig, hyperbolic, inverse hyperbolic, exp/log/sqrt/cbrt/abs/factorial), special functions, named ops (`\det`, `\gcd`, `\sigma`, `\sum`, `\lim`, `\partial`, `\binom`) |
| 3. Structure depth | poly degree 1–6; composition depth 1–4; rational deg ≤3/≤3; matrix 2×2–4×4; sum length 2–20; matrix entries int/rational |
| 4. Coefficient domain | int, exact Rational, decimal (log-scale), symbolic params (a,b,c), transcendental (π, e), mixed |
| 5. Input domain | real (domain-gated), integer, complex point, rational point |
| 6. Arity | 0 (constant queries: limits, dets, counts) … 6 vars |
| 7. Output type | int / Rational / float / `'re+imj'` / π-containing float |
| 8. Notation surface | repr-wrapped `\mathtt{\text{Integral(f, x)}}`, `\frac{d}{dx}`, `\int`, `\sum`, `\lim`, `\partial`, `\det`, `\binom`, `\operatorname{}`, pmatrix |
| 9. Question flavor | evaluate / derived-value / invariant / closed-form / count / moment |
| 10. Parameter coupling | independent / shared (test's `sqrt(a*x + b + x**2)` pattern) / correlated |

---

## 2. The Family Catalog (by SymPy module — verified against 1.14.0)

**Tier 1 — Test-adjacent (default mixture, highest caps).** Each row shape is measured in the frozen test.

| # | Family | SymPy module | Scalar query | LaTeX form | SymPy truth (verified) | Oracle |
|---|---|---|---|---|---|---|
| 1 | `evaluation` (extend `functions.py`) | `core`, `functions.elementary` | value of f(x) at input | full elementary vocabulary, nested | `sp.sympify(expr)` | numeric fresh-inputs |
| 2 | `derivative_repr` / `integral_repr` (NEW variants) | `calculus`, `integrals` | f′(x) value / F(x) value | **`\mathtt{\text{Derivative(f, x)}}` / `\mathtt{\text{Integral(f, x)}}`** — matches 25+41 frozen rows | `sp.Derivative(f,x).doit()`, `sp.Integral(f,x)` | differentiate-back + numeric |
| 3 | `differential_c1` (NEW ODE variant) | `solvers.ode` | y(x) given C1 | `\frac{d}{dx} y(x) = f(x,y)` with **C1 as input var** — matches 33 frozen rows | `sp.dsolve(ode, y(x))` | `checkodesol` + C1-subs numeric (1e-6) |
| 4 | `summation` (NEW) | `concrete` | Σ value / Π value | `\sum_{k=1}^{n} f(k)`, `\prod` | `sp.Sum(f,(k,1,n)).doit()` → 91 ✓, `sp.Product` → 720 ✓, telescoping `1/6` ✓ | exact int/Rational + closed-form identity |
| 5 | `multivariate` (extend) | `polys` | eval at point | int coeff + **int inputs** (33 diophantine rows), float inputs (32 rows), cross-terms, augmented shape | `sp.sympify` | numeric / exact int |
| 6 | `rational` (extend) | `polys` | P(x)/Q(x) at point, pole-avoided | `\frac{P}{Q}` (35 rows) | `sp.sympify` | numeric |
| 7 | `exponential`/`logarithmic` (extend) | `functions.elementary` | e^{kx}, log compositions | decimal coeffs (28 decay rows), `\log(6x)+2` (22 rows) | `sp.sympify` | numeric |
| 8 | `ode` (keep) | `solvers.ode` | IC-pinned solution value | `y' + p y = 0`, non-homogeneous, separable, 2nd-order | `dsolve` + IC | `checkodesol` + \|y(0)−y0\|<1e-6 |

**Tier 2 — Breadth (mixture with caps; real math diversity, partial/zero eval overlap).**

| # | Family | SymPy module | Scalar query | LaTeX form | SymPy truth (verified) | Oracle |
|---|---|---|---|---|---|---|
| 9 | `limits` (NEW) | `calculus`, `series` | lim value at c ∈ {0,±1,±2,∞, Rational} | `\lim_{x \to c} f(x)` | `sp.limit(f,x,c)` → sinx/x=1 ✓, (eˣ−1)/x=1 ✓, rational=2 ✓, ∞=2 ✓ | symbolic limit + **finiteness gate** (reject oo/unevaluated); ∞ via x→1/t substitution (naive ±ε fallback breaks at ∞) |
| 10 | `series_coeff` (NEW) | `series` | Nth Taylor coefficient of f at a | `\text{coeff}_{x^{k}}` | `sp.series(f,x,a,n).coeff(x,k)` → −1/6 ✓ | exact Rational |
| 11 | `polynomial_invariants` (NEW) | `polys` | Vieta sum/product of roots, coefficient of x^k, discriminant, resultant | `\operatorname{rootsum}`, `\operatorname{coeff}`, `\operatorname{disc}`, `\operatorname{res}` | `-b/a`, `c/a`; `Poly.coeff_monomial` → 32 ✓; `sp.discriminant` → b²−4a ✓; `sp.resultant` → 9 ✓ | exact (sympify-constant; no truth_code needed) |
| 12 | `matrix_scalars` (NEW) | `matrices` | det, trace, (A⁻¹)ᵢⱼ, charpoly(λ), Frobenius norm | `\det(A)`, `\operatorname{tr}(A)`, pmatrix A | `A.det()` −2 ✓, `A.trace()` 5 ✓, `A.inv()[0,1]` 3/2 ✓, `A.adjugate()[0,1]` −2 ✓, charpoly eval −8 ✓ | exact Rational (entries int/Rational) |
| 13 | `calculus_multivariate` (NEW) | `calculus`, `vector` | ∂²f/∂x∂y value, Laplacian ∇²f at point, div | `\partial`, `\nabla^{2}` | `laplacian(field, C)` (1.14 signature — takes expr only), `divergence(V)` | symbolic diff → numeric |
| 14 | `ntheory_ext` (extend) | `ntheory` | σ(n), τ(n), φ(n), μ(n), π(n), a⁻¹ mod m, a^b mod m, p(n) | `\operatorname{}` forms | `divisor_sigma` 28 ✓, `totient` 4 ✓, `mobius` 1 ✓, `primepi` 25 ✓, `mod_inverse` 5 ✓, **`pow(a,b,m)`** (sp.powermod absent — measured), `sp.partition` (npartitions deprecated) | exact int (`truth_code` for parameterized) |
| 15 | `combinatorics` (NEW) | `functions.combinatorial.numbers` | Bell(n), Catalan(n), Stirling, derangements !n, binomial identities | `B_{n}`, `C_{n}`, `\binom{n}{k}` | `catalan` 42 ✓, `bell` 52 ✓, `subfactorial` 44 ✓, `binomial` 120 ✓, Stirling via explicit import path | exact int |
| 16 | `geometry_ext` (extend) | `geometry` | triangle area (Heron), point-line distance, circumference, angle | distance/area formulas | exact Rational or mpmath π-float | coordinate substitution |
| 17 | `complex_eval` (extend) | `functions.elementary` | f(z) at complex point | standard forms | sympy evalf → 're+imj' | cmath.isclose on complex (public-probe adjacent, 98 rows) |

**Tier 3 — Gated (portfolio breadth; **excluded from default RL mixture**, available via `--include-gated-slices`).**

| # | Family | SymPy module | Scalar query | LaTeX form | SymPy truth (verified) | Oracle |
|---|---|---|---|---|---|---|
| 18 | `special_functions` (NEW) | `functions.special` | orthogonal poly values (**exact**), Γ/B (integer & half-integer args, exact), ζ(n≥2), erf, J_ν at rational points | `\Gamma`, `J_\nu`, `\operatorname{erf}` | `legendre(3,2)` 17 ✓, `chebyshevt(4,½)` −1/2 ✓, `gamma(5)` 24 ✓; erf/besselj stay unevaluated symbolically → mpmath | mpmath 50-digit almosteq; exact for orthogonal/Γ-int |
| 19 | `stats_moments` (NEW) | `stats` | E[X], Var[X], E[X²] | `\mathbb{E}`, `\mathrm{Var}`, `X \sim \mathcal{N}` | `E(Uniform(2,6))` 4 ✓, `variance(Binomial(10,½))` 5/2 ✓, `E(Normal(3,2))` 3 ✓ | closed-form identity + mpmath quad cross-check |
| 20 | `sets_cardinality` (NEW) | `sets` | |A∩B|, |A∪B| (intervals: measure; finite sets: **len()** — `.cardinality` absent, measured), membership 0/1 | `\left|A \cap B\right|` | `Interval.intersect(...).measure` 2 ✓, `len(FiniteSet.union(...))` | exact int + boundary checks |
| 21 | `solving_scalarized` (NEW) | `solvers` | unique root of ax+b=0, x-coordinate of system solution, positive quadratic root | equation forms | `sp.solve` → [2] ✓ / [2,3] | substitution residual == 0; **no Mod-equations via solve** (NotImplementedError — measured; use mod_inverse queries instead) |

---

## 3. Verification Oracle Matrix (extended)

| Oracle | Families | Mechanism | Precision |
|---|---|---|---|
| symbolic identity | all evaluation families | `sp.simplify(cand−truth)==0` / `sp.equals` | exact |
| differentiate-back | integral + repr variants | `sp.diff(cand, x) == integrand` | exact |
| `checkodesol` + IC/C1 | ode, differential_c1 | dsolve agreement + numeric subs | symbolic + 1e-6 |
| symbolic limit + finiteness gate | limits | `sp.limit` finite, reject unevaluated/oo | exact |
| series coefficient match | series_coeff | `sp.series(...).coeff(x,k)` equality | exact |
| exact scalar | polys/matrix/ntheory/combinatorics/sets | sympify-constant equality | exact |
| `truth_code` | parameterized truths (gcd, C1-ODE, modinv) | sandbox-executed ground truth, symmetric to candidate | exact/int |
| numeric fresh-inputs | evaluation, multivariate, rational, … | sandbox candidate vs oracle, `cmath.isclose` | 1e-9 rel |
| mpmath 50-digit | special_functions | `mpmath.almosteq` on candidate float vs high-precision truth | 50 digits |
| mpmath quad | stats, improper integrals | integrate PDF over support == 1 + moment identity | 1e-12 |
| substitution residual | solving_scalarized | plug root back, residual == 0 | exact |

---

## 4. Notation Alignment Audit (Phase 0 — closes the measured gaps)

Measured frozen-test surface: `\mathtt{\text{Integral(f, x)}}` (41), `\mathtt{\text{Derivative(f, x)}}` (25), `\frac{d}{dx} y(x)` + C1 input (33), `\sum` (33), `\log`, `e^{-kx}` decimal coeffs (28), `\frac{P}{Q}` (35), multi-var int coeff (33). **Zero** `\lim`, `\int` literal, `\partial`, `\det`, `\gcd`, complex outputs in frozen test.

Deliverables:
1. **repr-wrapper printer mode** — render `sp.Integral`/`sp.Derivative` as `\mathtt{\text{Integral(f, x)}}` (the AST-string surface), added as a variant alongside existing `\int`/`\frac{d}{dx}`/prime variants.
2. **`differential_c1`** — ODE family variant exposing C1 as an input.
3. **`summation`** — first `\sum` synthetic family (pool currently has **zero** `\sum` rows despite 1,832 in train and 33 in test).
4. **Standing alignment check** in `summary_synthetic.py`: overlap score between synthetic latex notation coverage and frozen-test notation coverage (the 15 measured patterns). Gate: ≥90% of covered test patterns after each phase.

---

## 5. Gating & Mixture Policy

- `gated: true` metadata on Tier 3 rows; `build_mixture.py` excludes them by default (`--include-gated-slices` opt-in for ablations).
- Mixture caps (of synthetic share): Tier 1 ≈ 65–70%, Tier 2 ≈ 25–30%, Tier 3 = 0% default.
- Contamination guard (latex vs frozen test/val) stays a hard exit for **every** family.
- Dedupe key stays `(latex, outputs)`; mixture regenerates byte-identically (seed 42).
- Scale target (no cost constraint): pool 7,450 → **~50k verified rows** across all 21 families; per-family caps prevent any single family from dominating the mixture.

---

## 6. Phased Execution Plan

| Phase | Content | Gate to proceed |
|---|---|---|
| **0** | notation audit + repr-wrapper variants + `differential_c1` + `summation` | alignment score ≥90% of covered test patterns; mixture rebuild; `make eval-gold` sanity |
| **1** | elementary vocabulary expansion (hyperbolics, inverse trig, atan2, abs, factorial, cbrt), domain engine, `complex_eval` | accept-rate ≥70% on 100-row probe per family; vocab coverage report vs train/test |
| **2** | `limits` + `series_coeff` | measure `sp.limit` latency + accept rate on 100 rows **before** scaling (limit calls are slow) |
| **3** | `polynomial_invariants` + `matrix_scalars` | exact-equality accept rate; pmatrix printer tests |
| **4** | `ntheory_ext` + `combinatorics` + `geometry_ext` | exact-int accept rate; deprecated-API pins (`sp.partition`, stirling import path) |
| **5** | `special_functions` + `stats_moments` + `sets_cardinality` (gated) | mpmath 50-digit gate pass; gated-flag exclusion verified in mixture |
| **6** | `solving_scalarized` + epistemological boundary doc + DATA_CARD/MODEL_CARD updates | full suite green; mixture byte-identical regeneration; 2 baseline rows re-measured |

Every phase: family module + verifier hookup + 2–4 tests (105 → ~140) + mixture wiring + summary report update. All CPU-only generation; GPU/API budget untouched.

---

## 7. Measurement Plan (what we report per phase)

- Accept/reject rate per family; verify latency per oracle (limit/mpmath are the slow ones).
- Complexity histogram (c1–c5) per family vs train's `{2: 11197, 3: 5275, 4: 3664, 5: 1830}`.
- Operator-vocabulary coverage vs train (7 funcs) and test.
- Notation-alignment score vs frozen test.
- Mixture composition + contamination re-check + SHA-256/byte-identical regeneration.
- Baseline rows re-measured after mixture changes (parse_latex 0.6675, floor 0.0076 stay frozen).

---

## Implemented Status (2026-08-11)

All 21 planned families are **implemented, oracle-verified, and integrated** into the code-first synthesizer.

### Family inventory (24 registered in FAMILIES — 21 from this plan + 3 legacy)

| Family | Slice | Gated | File | Verified rows |
| --- | --- | --- | --- | --- |
| differential_c1 | ode | no | `families/differential_c1.py` | 700 |
| limits | limits | no | `families/limits.py` | 400 |
| series_coeff | series | no | `families/series_coeff.py` | 400 |
| summation | summation | no | `families/summation.py` | 600 |
| polynomial_invariants | polynomial | no | `families/polynomial_invariants.py` | 300 |
| matrix_scalars | matrix | no | `families/matrix_scalars.py` | 250 |
| ntheory_ext | numtheory | no | `families/ntheory_ext.py` | 300 |
| combinatorics | combinatorics | no | `families/combinatorics.py` | 500 |
| elementary_ext | vocab | no | `families/elementary_ext.py` | 1200 |
| complex_eval | vocab | no | `families/complex_eval.py` | 600 |
| geometry_ext | geometry | no | `families/geometry_ext.py` | 400 |
| special_functions | special | **yes** | `families/special_functions.py` | 200 |
| stats_moments | stats | **yes** | `families/stats_moments.py` | 400 |
| sets_cardinality | sets | **yes** | `families/sets_cardinality.py` | 200 |
| solving_scalarized | algebraic | **yes** | `families/solving_scalarized.py` | 400 |

**Total new synthetic rows:** 6,850 (oracle-verified, latex-deduped)
**Pool (incl. regen of calculus_* + derivative_v1 with repr_surface=True):** 14,701 rows / 9,871 unique latex
**Notation alignment vs frozen test:** 5/5 patterns covered (100%) — was 2/5 (40%) pre-plan

### Measured notation alignment (scripts/summary_synthetic.py)

```
test patterns : {d/dx: 27, log: 22, repr-derivative: 25, repr-integral: 41, sum: 33}
pool patterns : {
  d/dx: 1100, log: 498, repr-derivative: 400, repr-integral: 795, sum: 426, ...
}
```

### Known gotchas (sympy 1.14) discovered along the way

1. **mpmath Chudnovsky-π hang on huge-arg trig** — `mp.sin(big_arg)` may trigger argument reduction that hangs minutes. Mitigation: per-iteration SIGALRM + sstr/count_ops size caps + tight domain ranges.
2. **Negative-Float rational powers → complex principal root** — `(-2.96)**(1/3) → 0.72+1.24j`. Restrict odd-root bases to positive domains.
3. **`Abs(sech(complex))` ~1e-30 imaginary noise** — breaks `float()`. Filter or sympify-trim.
4. **`sp.degree(expr, x)` returns sympy Integer** — leaks into JSON metadata. Always cast: `int(sp.degree(...))`.
5. **`sp.stirling` doesn't exist** in 1.14 — use `sympy.functions.combinatorial.numbers.stirling`.
6. **`sp.npartitions` deprecated** — use `sp.partition(n)`.
7. **`sp.solve` cannot solve `Mod` equations** (NotImplementedError) — use mod_inverse queries instead.
8. **`laplacian(expr)` takes 1 arg** in 1.14 (not `laplacian(expr, coords)`).
9. **`sp.FiniteSet` has no `.cardinality`** — use `len(...)`.
10. **Truth code under oracle jitter** — every committed (latex, outputs) pair must survive ±15% relative / ±3 int jitter; clamp inputs (max(n,1), gcd-handling for modinv, etc.).
11. **`format_output` uses `{c.imag:+}j`** — produces clean `'a-bj'` (parseable). Legacy `'a+-bj'` round-trip only via `complex()` (no-space form) or fallback `s.replace(' +-', '-')` (with-space form).

### Gated policy

The 4 gated portfolio families (`special_functions`, `stats_moments`, `sets_cardinality`, `solving_scalarized`) explore mathematical domains NOT present in the eval surface. They are excluded from the default RL mixture (`--include-gated-slices` opt-in) because:

- Training on non-eval domains risks output-contract drift (e.g., model emits `\sqrt{2}` form for a numeric truth).
- Negative transfer: RL can reward over-generalization that hurts eval-only coverage.

When gated rows ARE included (e.g., for portfolio demonstration), the slice caps shift: −16% calculus → +4% each in {special, stats, sets, algebraic}.

