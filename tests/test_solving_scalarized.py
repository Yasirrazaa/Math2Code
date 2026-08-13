"""Tests for the solving_scalarized family (gated, constant-output)."""

from __future__ import annotations

import sympy as sp

from math2code.data.oracle import oracle_verify
from math2code.data.synthesizer.families.solving_scalarized import (
    SolvingScalarizedFamily,
)
from math2code.evaluation.metrics import parse_number
from math2code.sandbox import SandboxPool, execute_code
from math2code.schemas import MathCodePair

_FAMILY = SolvingScalarizedFamily()
_NAME = "SolvingScalarizedFamily"


def _gen(count: int = 5, seed: int = 42) -> list[MathCodePair]:
    return _FAMILY.generate(seed, prefix="t", count=count)


def test_determinism() -> None:
    a = _gen(5, seed=42)
    b = _gen(5, seed=42)
    assert [(r.task_id, r.latex_expression) for r in a] == [
        (r.task_id, r.latex_expression) for r in b
    ]
    assert a


def test_contract_shape() -> None:
    rows = _gen(5)
    assert rows, "generate must produce rows"
    assert len(rows) == 10  # 5 objects * 2 latex variants
    for r in rows:
        assert isinstance(r, MathCodePair)
        assert r.latex_expression
        assert r.solution and "def calculate" in r.solution
        assert len(r.test_cases) == 5
        assert all(tc.input == {} for tc in r.test_cases), "constant-row inputs"
        assert r.equation_type == "algebraic"
        assert r.domain == "Mathematics_General"
        assert r.synthetic is True
        assert r.metadata["slice"] == "algebraic"
        assert r.metadata["kind"] in {"linear", "quadratic", "system"}
        for tc in r.test_cases:
            assert tc.output is not None
            assert abs(complex(parse_number(str(tc.output))).imag) < 1e-12


def test_gated_flag() -> None:
    rows = _gen(10)
    assert rows
    assert all(r.metadata["gated"] is True for r in rows)


def test_gate_holds_substitution_residual() -> None:
    """Committed root plugged back into the equation(s) gives exact zero."""
    rows = _gen(10)
    for r in rows:
        truth = sp.sympify(r.sympy_exp or "")
        assert truth.is_finite, r.task_id
        assert isinstance(truth, (sp.Rational, sp.Integer)), type(truth)
        sol = {sp.Symbol(k): sp.sympify(v) for k, v in r.metadata["solution"].items()}
        eqs = r.metadata["eqs"]
        assert eqs, r.task_id
        for lhs_s, rhs_s in eqs:
            residual = sp.simplify(sp.sympify(lhs_s).subs(sol) - sp.sympify(rhs_s))
            assert residual == 0, (r.task_id, lhs_s, rhs_s, truth)
        # the committed output must equal the exact truth
        expected = float(sp.N(truth))
        for tc in r.test_cases:
            assert abs(float(tc.output) - expected) < 1e-12  # type: ignore[arg-type]


def test_all_kinds_present() -> None:
    rows = _gen(12)
    kinds = {r.metadata["kind"] for r in rows}
    assert {"linear", "quadratic", "system"} <= kinds, kinds


def test_solution_matches_truth() -> None:
    """Executing each solution must reproduce the committed output."""
    for r in _gen(5):
        res = execute_code(r.solution or "", inputs={})
        assert res.ok, res.stderr
        assert abs(float(res.stdout) - float(r.test_cases[0].output)) < 1e-9  # type: ignore[arg-type]


def test_oracle_verification() -> None:
    rows = _gen(3)
    assert rows
    with SandboxPool(n_workers=2, timeout_s=10, memory_mb=2048) as pool:
        for r in rows:
            ok, reasons = oracle_verify(r, r.solution or "", pool=pool)
            assert ok, f"{r.task_id}: {reasons}"
