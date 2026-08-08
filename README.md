# Math2Code — LaTeX → Executable, Verifiable Python (SymPy)

> **Capstone → portfolio-grade ML engineering.** A math LLM that translates a LaTeX
> expression into executable Python (SymPy) code, trained with **rule-based RLVR
> (GRPO)**, verified by a **deterministic competition-scoring harness**, sandboxed
> end-to-end, and benchmarked against frontier API models — on a **$100 budget**.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI Pipeline](https://github.com/Yasirrazaa/Math2Code/actions/workflows/ci.yml/badge.svg)](https://github.com/Yasirrazaa/Math2Code/actions/workflows/ci.yml)

---

## What it does

Given LaTeX like `\frac{x^{2} + 3y^{2}}{2x + 5y}`, the model emits a `calculate(...)`
function that evaluates the expression correctly — including symbolic domains
(integration, derivatives, diophantine, complex outputs):

```python
import sympy as sp

def algebraic_function(x, y):
    x, y = sp.symbols('x y')
    expression = (x**2 + 3*y**2) / (2*x + 5*y)
    return float(expression.subs({x: x, y: y}))
```

Correctness is **execution-based, not string-based**: generated code runs in a
sandbox on test inputs and the outputs are compared numerically (`isclose`,
complex-aware) exactly like the bootcamp competition judged it.

## Why this repo exists (measurement-first)

The original capstone scored **0.75** on the (private) bootcamp leaderboard with an
evaluation that leaked training data (last 100 rows of the train file) and only
checked "runs without error". This rewrite fixes that:

1. **Frozen, contamination-checked split** — 22,796 deduplicated samples →
   22,002 train / 397 val / **397 test**, SHA-256 manifest + `test_ids.txt`.
   The gold solutions must score **1.0** on the test set before any model run
   counts — they do (397/397, verified twice).
2. **Competition-faithful metric** — per-case + per-problem accuracy, `isclose`
   rel_tol=1e-6 / abs_tol=1e-9, complex strings (`'-10.09+88.33j'`), `-inf`
   overflow equality, bootstrap 95% CIs.
3. **No LLM-judged rewards** — GRPO uses the deterministic rewards in
   `model/rewards.py` (oracle terminal reward, exec success, tool-use, traceback
   meta, complexity penalty) on **freshly resampled inputs** per rollout so the
   policy cannot memorize the 5 fixed test cases.
4. **No custom RL loop** — we stand on TRL's `GRPOTrainer` + `environment_factory`.

## Repository layout

```text
src/math2code/
├── schemas.py            # canonical MathCodePair contract (fixes python_code/solution drift)
├── data/
│   ├── competition.py    # loader, dedup, GRPO input re-sampling (x/x_val coupling)
│   ├── oracle.py         # layered verification: syntax -> Monte Carlo numeric -> identity
│   ├── generate.py       # secondary synthetic data (Groq + instructor)
│   └── curate.py         # oracle-gated curation
├── sandbox/
│   ├── base.py           # AST allowlist, RLIMIT, subprocess isolation
│   └── pool.py           # self-healing worker pool: 389 exec/s, 10k smoke in 26s
├── evaluation/
│   ├── metrics.py        # competition metric + bootstrap CI
│   ├── eval.py           # harness CLI: gold sanity check, score CSV, bench
│   └── runner.py         # model backends: hf:<id> (local) / api:<deepseek|openai>
├── model/
│   ├── prompts.py        # zero-shot + TIR prompts, code extraction
│   ├── rewards.py        # deterministic RLVR rewards (fully unit-tested)
│   ├── train.py          # SFT warmup (LoRA, TIR format) — GPU box
│   └── grpo.py           # TRL GRPOTrainer + environment_factory — GPU box
└── serve/
    ├── api.py            # FastAPI: vLLM completion -> extract -> sandbox execute
    └── app.py            # Gradio UI
```

## Quick start (local, CPU, $0)

```bash
uv pip install -e ".[dev]"
make splits      # rebuild frozen split from data/train.json
make eval-gold   # gold solutions must score 1.0 on the test split
make test        # 69 tests: metric, sandbox, oracle, rewards, runner, API
```

## Training + benchmarking (GPU / API budget)

```bash
uv pip install -e ".[train]"
python -m math2code.model.train           # SFT warmup (NuminaMath-7B-TIR, TIR format)
python -m math2code.model.grpo            # GRPO (1.5B burn-in on free T4, then 7B spot)
python -m math2code.evaluation.runner --model api:deepseek   # zero-shot baselines
python -m math2code.evaluation.runner --model hf:./outputs/grpo_burnin/final
```

Cost plan (see `PLAN.md` §Budget): weeks 1–3 cost **$0** (local CPU + free
Colab/Kaggle tiers); the 1.5B GRPO burn-in runs on a free T4; a single 7B spot
run (~$25–45 on a 3090/4090) plus cheap API baselines (~$6–11) keeps the total
under **$100**.

## Benchmarks

Functional correctness on the frozen test split (pass@1, greedy, bootstrap 95% CI).
The two zero-cost rows below are **measured** — they establish the floor before any
model money is spent; the LLM rows are filled by `evaluation/runner.py`:

| Model | Per-problem accuracy | CI95 |
|-------|---------------------|------|
| Trivial floor (always 0) | **0.0076** (3/397) | — |
| SymPy `parse_latex` (zero-cost) | **0.6675** (265/397) | — |
| GPT-4o-mini (zero-shot) | *pending* | — |
| DeepSeek-V3 (zero-shot) | *pending* | — |
| NuminaMath-7B-TIR (zero-shot) | *pending* | — |
| **Math2Code GRPO (Qwen2.5-Math-7B)** | *pending* | — |
| Gold solutions (harness sanity) | **1.0000** | — |

### What the zero-cost baseline tells us

`parse_latex` solves **100%** of the algebraic slice (rational, diophantine,
summation, exponential, multivariable, fractional, logrithmic, algebraic) but
**0%** of the 635-problem calculus slice — integration, differential,
derivative, exponential_decay (run `scripts/analyze_results.py` to reproduce).
The model's measurable value-add is exactly that slice, which is what the GRPO
curriculum should emphasize (`augmented_equation`, `logrithmic`, `fractional`,
and the 98 complex-output items are the OOD targets).

## Verification story (evidence in-repo)

- `make eval-gold` → `per-problem accuracy: 1.0000 (397/397)` on the frozen split
- `python scripts/smoke_pool.py` → `10,000 snippets in 25.7s → 389 exec/s, 0 failures`
- oracle verifies 200/200 gold solutions on fresh jittered inputs
- CI: ruff, mypy, 69 pytest tests, package build, Docker build, split-integrity check

See `PLAN.md` for the full blueprint and `docs/ENGINEERING_REVIEW.md` for the
original codebase audit.
