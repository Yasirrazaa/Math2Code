"""Competition data access: loaders, dedup, and GRPO re-sampling.

The competition `train.json` is the primary dataset: 26,846 verified samples
with ground-truth `sympy_exp` and 5 test cases (with expected outputs) each.

Re-sampling matters for RL: training GRPO against the *same* 5 fixed inputs per
prompt invites input memorization. `resample_test_cases` jitters each original
test-case input (preserving type/domain semantics) and recomputes the expected
output from the ground-truth sympy expression — a fresh execution target per
rollout.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import sympy as sp

from math2code.schemas import MathCodePair, TestCase, from_competition_row

DATA_DIR = Path(__file__).resolve().parents[3] / "data"


def load_competition_train(path: str | Path | None = None) -> list[MathCodePair]:
    """Load data/train.json into MathCodePair rows."""
    path = Path(path) if path else DATA_DIR / "train.json"
    with open(path) as f:
        rows = json.load(f)
    return [from_competition_row(r) for r in rows]


def load_public_test(path: str | Path | None = None) -> list[MathCodePair]:
    """Load public_test_new_no_sol_no_out.json (no expected outputs)."""
    path = Path(path) if path else DATA_DIR / "public_test_new_no_sol_no_out.json"
    with open(path) as f:
        rows = json.load(f)
    return [from_competition_row(r) for r in rows]


def dedup_by_latex(pairs: list[MathCodePair]) -> list[MathCodePair]:
    """Keep the first occurrence of each latex expression."""
    seen: set[str] = set()
    out: list[MathCodePair] = []
    for p in pairs:
        if p.latex_expression in seen:
            continue
        seen.add(p.latex_expression)
        out.append(p)
    return out


def _jitter(value: int | float | complex, rng: random.Random) -> int | float | complex:
    """Perturb a numeric input, preserving int-ness (diophantine inputs)."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value + rng.randint(-3, 3))
    if isinstance(value, complex):
        return value
    f = float(value)
    return f * (1.0 + rng.uniform(-0.15, 0.15))


def resample_test_cases(
    pair: MathCodePair,
    n: int = 5,
    seed: int = 0,
    use_sympy: bool = True,
) -> list[TestCase]:
    """Fresh test cases with expected outputs recomputed from ground truth.

    Inputs are jittered around the original test-case values (keeps the input
    distribution realistic: ints stay ints, ranges stay in-range). Expected
    outputs come from the ground-truth `sympy_exp` (fast, deterministic) or,
    failing that, from executing the gold solution.
    """
    rng = random.Random(f"{pair.task_id}:{seed}")
    base = pair.test_cases or []
    template_inputs = [tc.input for tc in base] or [{}]
    cases: list[TestCase] = []
    for i in range(n):
        tmpl = template_inputs[i % len(template_inputs)]
        point: dict[str, int | float | complex] = {
            k: _jitter(v, rng) for k, v in tmpl.items()
        }
        expected = None
        if use_sympy and pair.sympy_exp:
            expected = _sympy_eval(pair.sympy_exp, point)
        if expected is None and pair.solution:
            expected = _gold_eval(pair.solution, point)
        cases.append(TestCase(input=point, output=expected))
    return cases


def _sympy_eval(sympy_exp: str, point: dict[str, Any]) -> float | complex | None:
    try:
        expr = sp.sympify(sympy_exp)
        subs = {sp.Symbol(k): v for k, v in point.items()}
        val = complex(expr.subs(subs).evalf())
        if abs(val.imag) < 1e-9:
            return val.real
        return val
    except Exception:
        return None


def _gold_eval(code: str, point: dict[str, Any]) -> float | complex | None:
    from math2code.sandbox import execute_code

    res = execute_code(code, inputs=point)
    if not res.ok:
        return None
    from math2code.evaluation.metrics import parse_number

    try:
        return parse_number(res.stdout)
    except ValueError:
        return None
