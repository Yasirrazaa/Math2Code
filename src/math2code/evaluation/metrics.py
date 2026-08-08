"""Competition scoring metric: exact number match with numeric tolerance.

Reproduces the bootcamp competition evaluation:
  - parse each submitted output (stringified number, possibly complex)
  - compare to the expected value with `math.isclose`
  - aggregate per-case and per-problem accuracy
"""

from __future__ import annotations

import ast
import math
from collections.abc import Sequence
from dataclasses import dataclass

from math2code.schemas import TestCase


def parse_number(value: object) -> complex:
    """Robustly parse a submitted output into a complex number.

    Handles: floats, ints, complex, np types, and strings such as
    ``'144328315.93417865'`` or ``'-10.096475+88.331647j'``.
    """
    if isinstance(value, complex):
        return value
    if isinstance(value, bool):
        return complex(int(value))
    if isinstance(value, (int, float)):
        return complex(value)
    if isinstance(value, str):
        s = value.strip()
        # try literal eval first (covers 'a+bj' and '1e-6')
        try:
            return complex(ast.literal_eval(s))
        except (ValueError, SyntaxError):
            try:
                return complex(float(s))
            except ValueError as exc:
                raise ValueError(f"unparseable output: {value!r}") from exc
    # numpy / sympy numeric types
    try:
        return complex(value)  # type: ignore[no-any-return, call-overload]
    except TypeError as exc:
        raise ValueError(f"unparseable output: {value!r}") from exc


def outputs_match(
    got: object,
    expected: object,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-9,
) -> bool:
    """Numeric equality with tolerance; complex compared on both parts.

    Missing (None) or unparseable outputs are mismatches, never exceptions.
    """
    if got is None:
        return False
    try:
        g = parse_number(got)
        e = parse_number(expected)
    except ValueError:
        return False
    # non-finite equality: +/-inf must equal +/-inf; NaN never matches
    real_ok = (
        math.isclose(g.real, e.real, rel_tol=rel_tol, abs_tol=abs_tol)
        if math.isfinite(g.real) and math.isfinite(e.real)
        else g.real == e.real
    )
    imag_ok = (
        math.isclose(g.imag, e.imag, rel_tol=rel_tol, abs_tol=abs_tol)
        if math.isfinite(g.imag) and math.isfinite(e.imag)
        else g.imag == e.imag
    )
    return real_ok and imag_ok


@dataclass
class ScoredCase:
    task_id: str
    case_index: int
    got: object
    expected: object
    passed: bool


@dataclass
class ScoreResult:
    n_cases: int
    n_correct_cases: int
    n_problems: int
    n_correct_problems: int
    per_case_accuracy: float
    per_problem_accuracy: float
    details: list[ScoredCase]

    def summary(self) -> str:
        return (
            f"per-case accuracy:   {self.per_case_accuracy:.4f} "
            f"({self.n_correct_cases}/{self.n_cases})\n"
            f"per-problem accuracy: {self.per_problem_accuracy:.4f} "
            f"({self.n_correct_problems}/{self.n_problems})"
        )


def score_predictions(
    predictions: Sequence[Sequence[object]],
    test_cases: Sequence[Sequence[TestCase]],
    task_ids: Sequence[str] | None = None,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-9,
) -> ScoreResult:
    """Score `predictions[i]` (list of outputs) against `test_cases[i]`.

    A problem counts as correct only if **all** its test cases pass.
    """
    assert len(predictions) == len(test_cases), "predictions/test_cases length mismatch"
    details: list[ScoredCase] = []
    n_correct_cases = 0
    n_correct_problems = 0
    n_cases = 0
    for i, (pred, cases) in enumerate(zip(predictions, test_cases)):
        tid = task_ids[i] if task_ids else str(i)
        assert len(pred) == len(cases), (
            f"task {tid}: got {len(pred)} outputs for {len(cases)} test cases"
        )
        n_cases += len(cases)
        problem_ok = True
        for j, (got, case) in enumerate(zip(pred, cases)):
            if case.output is None:
                raise ValueError(
                    f"task {tid} case {j} has no expected output; use a scored split"
                )
            ok = outputs_match(got, case.output, rel_tol=rel_tol, abs_tol=abs_tol)
            details.append(ScoredCase(tid, j, got, case.output, ok))
            n_correct_cases += int(ok)
            problem_ok = problem_ok and ok
        n_correct_problems += int(problem_ok)
    return ScoreResult(
        n_cases=n_cases,
        n_correct_cases=n_correct_cases,
        n_problems=len(test_cases),
        n_correct_problems=n_correct_problems,
        per_case_accuracy=n_correct_cases / n_cases,
        per_problem_accuracy=n_correct_problems / len(test_cases),
        details=details,
    )


def format_output(value: object) -> str:
    """Canonical string form for a computed output (matches submission format)."""
    c = parse_number(value)
    if c.imag != 0:
        return f"{c.real}+{c.imag}j"
    return str(c.real)
