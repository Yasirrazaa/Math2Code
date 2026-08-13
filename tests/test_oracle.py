"""Oracle tests: gold passes, broken candidates fail, input coupling."""

from __future__ import annotations

import random

from math2code.data.competition import jitter_inputs
from math2code.data.oracle import (
    identity_check,
    numeric_check,
    oracle_verify,
    syntactic_check,
)
from math2code.schemas import MathCodePair, TestCase


def _pair(task_id: str = "t1") -> MathCodePair:
    return MathCodePair(
        task_id=task_id,
        latex_expression=r"2 \cdot x + 1",
        sympy_exp="2*x + 1",
        solution="def calculate(x):\n    return 2 * x + 1",
        test_cases=[
            TestCase(input={"x": 1.0}, output=3.0),
            TestCase(input={"x": 2.0}, output=5.0),
        ],
    )


def test_syntactic_check() -> None:
    ok, _ = syntactic_check("def calculate(x):\n    return x")
    assert ok
    ok, why = syntactic_check("import os\nos.system('id')")
    assert not ok
    ok, why = syntactic_check("print('no function')")
    assert not ok


def test_numeric_check_gold_passes() -> None:
    ok, why = numeric_check(_pair(), "def calculate(x):\n    return 2 * x + 1")
    assert ok, why


def test_numeric_check_wrong_code_fails() -> None:
    ok, why = numeric_check(_pair(), "def calculate(x):\n    return x + 1")
    assert not ok
    assert "mismatch" in why


def test_numeric_check_falls_back_when_lambdify_unavailable() -> None:
    """factorial is not numpy-expressible -> vectorized truth must fall back
    to the per-point evalf path (semantics unchanged)."""
    p = MathCodePair(
        task_id="t3",
        latex_expression=r"x!",
        sympy_exp="factorial(x)",
        solution=(
            "import sympy as sp\n"
            "def calculate(x):\n"
            "    return int(sp.factorial(int(x)))"
        ),
        test_cases=[TestCase(input={"x": 4}, output=24)],
    )
    ok, why = numeric_check(p, p.solution or "")
    assert ok, why


def test_numeric_check_vectorized_truth_matches_gold() -> None:
    """The lambdify path agrees with the ground truth on fresh jittered inputs."""
    ok, why = numeric_check(_pair(), "def calculate(x):\n    return 2 * x + 1", n=20)
    assert ok, why


def test_numeric_check_complex_truth_is_compared_not_skipped() -> None:
    """Complex-valued ground truths (sqrt of a negative arg -> sympy gives a
    finite complex, numpy gives NaN) must still be compared: a wrong candidate
    must fail, not pass vacuously because the points were skipped."""
    p = MathCodePair(
        task_id="t4",
        latex_expression=r"\sqrt{x}",
        sympy_exp="sqrt(x)",
        solution="import cmath\ndef calculate(x):\n    return cmath.sqrt(x)",
        test_cases=[TestCase(input={"x": -4.0}, output="0.0+2.0j")],
    )
    ok, why = numeric_check(p, p.solution or "")
    assert ok, why
    # a real-constant candidate must FAIL: every jittered point is complex
    bad, why_bad = numeric_check(p, "def calculate(x):\n    return 5.0")
    assert not bad, f"wrong candidate passed: {why_bad}"


def test_oracle_verify_full() -> None:
    ok, reasons = oracle_verify(_pair(), "def calculate(x):\n    return 2 * x + 1")
    assert ok
    assert any(r.startswith("numeric: pass") for r in reasons)
    bad, reasons = oracle_verify(_pair(), "def calculate(x):\n    return 999")
    assert not bad
    assert any(r.startswith("numeric:") for r in reasons)


def test_identity_check_zero_arg() -> None:
    p = MathCodePair(
        task_id="t2",
        latex_expression="42",
        sympy_exp="42",
        solution="def calculate():\n    return 42",
        test_cases=[TestCase(input={}, output=42)],
    )
    ok, why = identity_check(p, "def calculate():\n    return 42")
    assert ok, why


def test_identity_check_skips_parameterized() -> None:
    ok, why = identity_check(_pair(), "def calculate(x):\n    return 2 * x + 1")
    assert not ok
    assert "not applicable" in why


def test_jitter_inputs_couples_val_variants() -> None:
    """x and x_val must receive the SAME jittered value (function contract)."""
    rng = random.Random(7)
    point = jitter_inputs({"x": 2.0, "x_val": 2.0, "y": 3.0}, rng)
    assert point["x"] == point["x_val"]
    assert point["x"] != 2.0  # jittered away from original
    assert point["y"] != 3.0


def test_jitter_inputs_preserves_int() -> None:
    rng = random.Random(1)
    point = jitter_inputs({"n": 5}, rng)
    assert isinstance(point["n"], int)
