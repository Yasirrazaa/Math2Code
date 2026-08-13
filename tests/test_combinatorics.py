"""Tests for the combinatorics family (exact-integer counting queries).

Generation is the slow part (parameterized rows run committed outputs through
the sandbox, ~5 subprocess spawns per row), so the heavier tests share ONE
module-scoped generation and ONE sandbox pool.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import sympy as sp

from math2code.data.oracle import oracle_verify
from math2code.data.synthesizer.families.combinatorics import CombinatoricsFamily
from math2code.evaluation.metrics import parse_number
from math2code.sandbox import SandboxPool
from math2code.schemas import MathCodePair


@pytest.fixture(scope="module")
def pool() -> Iterator[SandboxPool]:
    with SandboxPool(n_workers=2, timeout_s=10, memory_mb=2048) as p:
        yield p


@pytest.fixture(scope="module")
def rows() -> list[MathCodePair]:
    return CombinatoricsFamily().generate(42, "t", 3)


def test_determinism() -> None:
    a = CombinatoricsFamily().generate(42, "t", 2)
    b = CombinatoricsFamily().generate(42, "t", 2)
    assert [(r.task_id, r.latex_expression) for r in a] == [
        (r.task_id, r.latex_expression) for r in b
    ]
    assert a  # non-empty


def test_contract_shape(rows: list[MathCodePair]) -> None:
    assert rows
    for r in rows:
        assert r.latex_expression
        assert "def calculate" in (r.solution or "")
        assert len(r.test_cases) == 5
        assert r.equation_type == "combinatorics"
        assert r.domain == "Mathematics_General"
        assert r.synthetic is True
        assert r.metadata["slice"] == "combinatorics"
        assert r.metadata["vocab"] in {
            "bell",
            "catalan",
            "derangement",
            "stirling",
            "binomial",
            "binomial_sum",
        }
        assert r.complexity in (1, 2, 3, 4, 5)
        for tc in r.test_cases:
            assert parse_number(str(tc.output)) is not None


def test_truth_correctness_exact(rows: list[MathCodePair], pool: SandboxPool) -> None:
    """Sandbox solution output == committed output == exact sympy count."""
    for r in rows:
        inputs = [tc.input for tc in r.test_cases]
        outs, errors = pool.run_solution_on_cases(r.solution or "", inputs)
        assert all(o is not None for o in outs), errors
        for got, tc in zip(outs, r.test_cases):
            assert parse_number(got) == tc.output  # exact (float of same int)
        if r.truth_code:
            # parameterized rows: truth_code recomputation identity
            touts, _ = pool.run_solution_on_cases(r.truth_code, inputs)
            assert all(o is not None for o in touts)
            for got, tc in zip(touts, r.test_cases):
                assert parse_number(got) == tc.output
        else:
            # concrete rows: sympify constant is the exact positive count
            expr = sp.sympify(r.sympy_exp or "0")
            assert expr.is_integer is True and int(expr) > 0
            for tc in r.test_cases:
                assert sp.N(expr) == sp.N(sp.Float(tc.output or 0))


def test_magnitude_guard(rows: list[MathCodePair]) -> None:
    for r in rows:
        for tc in r.test_cases:
            assert 0 < abs(float(tc.output or 0)) <= 10**9  # type: ignore[arg-type]


def test_oracle_verify(pool: SandboxPool) -> None:
    rows = CombinatoricsFamily().generate(42, "t", 2)
    assert rows
    for r in rows[:2]:
        ok, reasons = oracle_verify(r, r.solution or "", pool=pool)
        assert ok, reasons
