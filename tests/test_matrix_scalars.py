"""Tests for the matrix scalar-invariant family (exact Rational outputs)."""

from __future__ import annotations

from collections import Counter

import pytest
import sympy as sp

from math2code.data.oracle import oracle_verify
from math2code.data.synthesizer.families.matrix_scalars import MatrixScalarsFamily
from math2code.evaluation.metrics import parse_number
from math2code.sandbox import SandboxPool
from math2code.schemas import MathCodePair

FAM = MatrixScalarsFamily()


def _gen(count: int, seed: int = 42) -> list[MathCodePair]:
    return FAM.generate(seed=seed, prefix="t", count=count)


def test_determinism() -> None:
    a = _gen(6)
    b = _gen(6)
    assert [(r.task_id, r.latex_expression) for r in a] == [
        (r.task_id, r.latex_expression) for r in b
    ]


def test_contract_shape() -> None:
    rows = _gen(8)
    assert rows, "family produced no rows"
    for r in rows:
        assert isinstance(r, MathCodePair)
        assert r.latex_expression.startswith("\\")
        assert r.latex_expression.count("begin{pmatrix}") == 1
        assert r.solution and "def calculate" in r.solution
        assert len(r.test_cases) == 5
        assert all(tc.input == {} for tc in r.test_cases)  # constant rows
        assert r.equation_type == "matrix"
        assert r.domain == "Mathematics_General"
        assert r.synthetic is True
        assert r.metadata["slice"] == "matrix"
        assert r.metadata["size"] in (2, 3, 4)
        assert r.metadata["query"] in ("det", "trace", "inverse", "charpoly", "frobenius")
        assert r.metadata["coefficient_kind"] in ("integer", "rational")
        for tc in r.test_cases:
            assert tc.output is not None
            parse_number(tc.output)  # parseable by the competition metric


def test_ground_truth_exact() -> None:
    """sympy_exp is the exact truth; recompute it and compare with the row."""
    rows = _gen(10)
    for r in rows:
        truth = sp.sympify(r.sympy_exp)  # type: ignore[arg-type]
        for tc in r.test_cases:
            assert sp.N(truth) == pytest.approx(complex(tc.output).real, rel=1e-9)


def test_oracle_verification() -> None:
    rows = _gen(4)
    with SandboxPool(n_workers=2, timeout_s=10, memory_mb=1024) as pool:
        for r in rows[:2]:
            ok, reasons = oracle_verify(r, r.solution or "", pool=pool)
            assert ok, f"row {r.task_id} failed oracle: {reasons}"


def test_query_diversity() -> None:
    rows = _gen(60)
    queries = Counter(r.metadata["query"] for r in rows)
    assert set(queries) == {"det", "trace", "inverse", "charpoly", "frobenius"}
    sizes = Counter(r.metadata["size"] for r in rows)
    assert set(sizes) <= {2, 3, 4}


def test_magnitude_and_finiteness() -> None:
    rows = _gen(60)
    for r in rows:
        truth = sp.sympify(r.sympy_exp)  # type: ignore[arg-type]
        assert truth.is_finite
        assert abs(truth) <= 1_000_000  # magnitude guard
        if r.metadata["query"] == "det":
            assert truth != 0  # trivial dets are skipped
