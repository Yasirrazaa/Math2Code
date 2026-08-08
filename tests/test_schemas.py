"""Data contract tests: canonical schema + competition row normalization."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from math2code.schemas import MathCodePair, TestCase, from_competition_row


def test_math_code_pair_contract() -> None:
    pair = MathCodePair(
        task_id="t1",
        latex_expression=r"\frac{x}{y}",
        solution="def calculate(x, y): return x / y",
        test_cases=[TestCase(input={"x": 2, "y": 1}, output=2.0)],
    )
    assert pair.solution is not None
    assert pair.test_cases[0].output == 2.0


def test_math_code_pair_requires_task_id() -> None:
    with pytest.raises(ValidationError):
        MathCodePair.model_validate({"latex_expression": r"\frac{x}{y}"})


def test_from_competition_row_full() -> None:
    row = {
        "task_id": "abc123",
        "sympy_exp": "pi*h*r**2",
        "latex_expression": r"\pi h r^{2}",
        "solution": "def geometric_function(r, h):\n    return np.pi*h*r**2\n",
        "synthetic": True,
        "domain": "Mathematics_Geometry",
        "test_cases": [{"input": {"r": 1.0, "h": 2.0}, "output": 6.283185307179586}],
        "complexity": "2",
        "equation_type": "Geometry",
        "output_type": "real",
    }
    pair = from_competition_row(row)
    assert pair.task_id == "abc123"
    assert pair.sympy_exp == "pi*h*r**2"
    assert pair.test_cases[0].output == 6.283185307179586
    assert pair.complexity == "2"


def test_from_competition_row_nan_domain() -> None:
    """Some competition rows have domain: nan -> must normalize to None."""
    row = {
        "task_id": "xyz",
        "latex_expression": "x + 1",
        "domain": float("nan"),
        "equation_type": "derivative",
        "test_cases": [],
    }
    pair = from_competition_row(row)
    assert pair.domain is None
    assert pair.equation_type == "derivative"


def test_roundtrip_real_competition_row() -> None:
    """A real row from data/train.json must parse cleanly."""
    with open("data/train.json") as f:
        rows = json.load(f)
    pair = from_competition_row(rows[0])
    assert pair.task_id == rows[0]["task_id"]
    assert len(pair.test_cases) == 5
    assert all(tc.output is not None for tc in pair.test_cases)
