"""Metric tests: numeric exact-match with tolerance, complex, failure modes."""

from __future__ import annotations

import math

from math2code.evaluation.metrics import (
    ScoreResult,
    format_output,
    outputs_match,
    parse_number,
    score_predictions,
)
from math2code.schemas import TestCase


def test_parse_number_variants() -> None:
    assert parse_number("144328315.93417865") == complex(144328315.93417865)
    assert parse_number("-10.096475+88.331647j") == complex(-10.096475, 88.331647)
    assert parse_number(42) == complex(42)
    assert parse_number("1e-6") == complex(1e-6)
    assert parse_number(complex(1, 2)) == complex(1, 2)
    with __import__("pytest").raises(ValueError):
        parse_number("not a number")


def test_outputs_match() -> None:
    assert outputs_match("0.1", "0.10000000000000002")
    assert outputs_match(complex(1, 2), "1+2j")
    assert outputs_match(1.0000001, 1.0)  # within rel_tol
    assert not outputs_match(1.0, 1.001)
    assert not outputs_match(None, 1.0)  # missing output = mismatch, not crash
    assert not outputs_match("garbage", 1.0)


def test_outputs_match_non_finite() -> None:
    """Competition gold solutions can legitimately overflow to +/-inf."""
    assert outputs_match(float("-inf"), float("-inf"))
    assert outputs_match(float("inf"), float("inf"))
    assert not outputs_match(float("inf"), float("-inf"))
    assert not outputs_match(float("inf"), 1e300)
    assert not outputs_match(float("nan"), float("nan"))


def test_format_output() -> None:
    assert format_output(2.0) == "2.0"
    assert format_output(2 + 3j) == "2.0+3.0j"


def test_score_predictions_per_case_and_per_problem() -> None:
    tcs = [
        [TestCase(input={"x": 1}, output=1.0), TestCase(input={"x": 2}, output=2.0)],
        [TestCase(input={"x": 1}, output=1.0), TestCase(input={"x": 2}, output=2.0)],
    ]
    result: ScoreResult = score_predictions(
        predictions=[[1.0, 2.0], [1.0, 99.0]], test_cases=tcs, task_ids=["a", "b"]
    )
    assert result.per_case_accuracy == 0.75  # 3/4 cases
    assert result.per_problem_accuracy == 0.5  # only problem 'a' fully correct
    assert result.n_correct_cases == 3


def test_score_predictions_complex() -> None:
    tcs = [[TestCase(input={"x": 1}, output=complex(1, 2))]]
    result = score_predictions(
        predictions=[["1.0+2.0j"]], test_cases=tcs, task_ids=["a"]
    )
    assert result.per_problem_accuracy == 1.0


def test_score_predictions_raises_on_missing_expected() -> None:
    tcs = [[TestCase(input={"x": 1}, output=None)]]
    import pytest

    with pytest.raises(ValueError):
        score_predictions(predictions=[[1.0]], test_cases=tcs, task_ids=["a"])


def test_isclose_semantics_match_math_module() -> None:
    assert math.isclose(1.0, 1.0 + 1e-7, rel_tol=1e-6)
