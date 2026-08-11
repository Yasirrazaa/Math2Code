"""Tests for the C1-as-input differential family (general-solution ODEs)."""

from __future__ import annotations

import sympy as sp

from math2code.data.synthesizer.families.differential_c1 import (
    _C1,
    _C2,
    _YF,
    DifferentialC1Family,
)
from math2code.data.synthesizer.verify import verify_generated
from math2code.evaluation.metrics import parse_number
from math2code.sandbox import SandboxPool
from math2code.schemas import MathCodePair


def _rows(count: int = 5, seed: int = 42, n_variants: int = 1) -> list[MathCodePair]:
    return DifferentialC1Family().generate(
        seed, prefix="t", count=count, n_variants=n_variants
    )


def test_determinism() -> None:
    a = _rows(5, seed=42)
    b = _rows(5, seed=42)
    assert [(r.task_id, r.latex_expression) for r in a] == [
        (r.task_id, r.latex_expression) for r in b
    ]
    assert len(a) == 5


def test_contract_shape() -> None:
    rows = _rows(6)
    assert rows
    for r in rows:
        assert isinstance(r, MathCodePair)
        assert r.latex_expression
        assert r.solution and "def calculate" in r.solution
        assert len(r.test_cases) == 5
        assert r.equation_type == "differential"
        assert r.domain == "Mathematics_Calculus"
        assert r.synthetic is True
        assert r.metadata["slice"] == "ode"
        assert r.metadata["ode"]
        for tc in r.test_cases:
            assert set(tc.input) >= {"x", "C1"}
            assert parse_number(tc.output) is not None  # type: ignore[arg-type]
        # general solution carries its free constant(s) as inputs
        n_const = r.metadata["n_constants"]
        assert n_const in (1, 2)
        expected = {_C1, _C2} if n_const == 2 else {_C1}
        rhs = sp.sympify(r.sympy_exp)  # type: ignore[arg-type]
        assert expected.issubset(rhs.free_symbols)


def test_gate_holds() -> None:
    """checkodesol must hold for every emitted row (re-asserted independently)."""
    for r in _rows(8):
        ode = sp.sympify(r.metadata["ode"])
        rhs = sp.sympify(r.sympy_exp)  # type: ignore[arg-type]
        ok, residual = sp.checkodesol(ode, sp.Eq(_YF, rhs))
        assert ok, f"checkodesol failed for {r.task_id}: {residual}"


def test_oracle_verify() -> None:
    rows = _rows(3, n_variants=2)
    assert len(rows) == 6
    with SandboxPool(n_workers=2, timeout_s=10) as pool:
        outcome = verify_generated(rows, pool, n_points=10)
    assert len(outcome.kept) == len(rows), outcome.rejected[:3]


def test_robustness() -> None:
    rows = _rows(10)
    assert len(rows) == 10  # attempt cap never starves count=10
    for r in rows:
        assert 1 <= int(r.complexity) <= 5  # type: ignore[arg-type]
        for tc in r.test_cases:
            out = tc.output
            assert isinstance(out, (int, float))  # real outputs only
            assert abs(float(out)) < 1e12  # finiteness guard held at build time
