"""End-to-end eval harness tests on a small in-memory fixture."""

from __future__ import annotations

from math2code.evaluation.eval import gold_check
from math2code.sandbox import SandboxPool
from math2code.schemas import MathCodePair, TestCase


def _fixture_pair(tid: str, coeff: int) -> MathCodePair:
    return MathCodePair(
        task_id=tid,
        latex_expression=f"{coeff} * x",
        solution=f"def calculate(x):\n    return {coeff} * x",
        test_cases=[
            TestCase(input={"x": 1.0}, output=coeff * 1.0),
            TestCase(input={"x": 2.0}, output=coeff * 2.0),
            TestCase(input={"x": 3.0}, output=coeff * 3.0),
        ],
    )


def test_gold_check_scores_perfect() -> None:
    pairs = [_fixture_pair("a", 2), _fixture_pair("b", 5)]
    with SandboxPool(n_workers=2) as pool:
        result = gold_check(pairs, n=None, pool=pool)
    assert result.per_problem_accuracy == 1.0
    assert result.per_case_accuracy == 1.0


def test_gold_check_catches_wrong_code() -> None:
    pairs = [_fixture_pair("a", 2), _fixture_pair("b", 5)]
    # corrupt the gold solution for 'b'
    pairs[1].solution = "def calculate(x):\n    return x"  # wrong on purpose
    with SandboxPool(n_workers=2) as pool:
        result = gold_check(pairs, n=None, pool=pool)
    assert result.per_problem_accuracy < 1.0
    assert result.per_case_accuracy < 1.0
