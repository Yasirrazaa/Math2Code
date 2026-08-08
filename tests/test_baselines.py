"""Baseline tests: scoring pipeline + floor/parse baselines on a tiny fixture."""

from __future__ import annotations

from math2code.evaluation.baselines import (
    latex_parse_baseline,
    run_baseline,
    trivial_floor,
)
from math2code.schemas import MathCodePair, TestCase


def _pair(
    tid: str, outputs: list[float], inputs: list[dict] | None = None
) -> MathCodePair:
    inputs = inputs or [{"x": float(i + 1)} for i in range(len(outputs))]
    return MathCodePair(
        task_id=tid,
        latex_expression="x + 1",
        solution="def calculate(x):\n    return x + 1",
        test_cases=[
            TestCase(input=inp, output=out) for inp, out in zip(inputs, outputs)
        ],
    )


def test_trivial_floor_zero_when_answers_nonzero() -> None:
    pairs = [_pair("a", [1.0, 2.0, 3.0]), _pair("b", [5.0, 6.0, 7.0])]
    preds = trivial_floor(pairs)
    assert preds == [["0.0", "0.0", "0.0"], ["0.0", "0.0", "0.0"]]


def test_trivial_floor_scores(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pairs = [_pair("a", [0.0, 0.0, 0.0]), _pair("b", [5.0, 6.0, 7.0])]
    result, _ = run_baseline(pairs, "test_floor", trivial_floor, results_dir=tmp_path)
    assert result.per_problem_accuracy == 0.5  # only 'a' is all-zero
    assert result.per_case_accuracy == 0.5  # 3/6 cases


def test_latex_parse_baseline_never_crashes() -> None:
    """parse_latex may fail on exotic latex; the baseline must produce None."""
    pairs = [_pair("a", [1.0, 2.0, 3.0])]
    preds = latex_parse_baseline(pairs)
    assert len(preds) == 1 and len(preds[0]) == 3
