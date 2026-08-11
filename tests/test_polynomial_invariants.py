"""Polynomial-invariants family tests.

Each emitted value is independently recomputed from the committed metadata
(`poly`/`poly2`/`k`) through SymPy and compared against the committed ground
truth — a second, symbolic proof layer on top of the sandbox oracle.
"""

from __future__ import annotations

import json

import sympy as sp

from math2code.data.synthesizer.families.polynomial_invariants import (
    PolynomialInvariantsFamily,
    coeff_of,
    discriminant_of,
    resultant_of,
    vieta_product,
    vieta_sum,
)
from math2code.evaluation.metrics import parse_number
from math2code.sandbox import SandboxPool
from math2code.schemas import MathCodePair

_X = sp.Symbol("x")


def _rows(n: int = 8, seed: int = 42) -> list[MathCodePair]:
    return PolynomialInvariantsFamily().generate(seed, "t", n)


def test_generation_deterministic() -> None:
    a = [(r.task_id, r.latex_expression) for r in _rows()]
    b = [(r.task_id, r.latex_expression) for r in _rows()]
    assert a == b  # byte-identical regeneration


def test_contract_shape() -> None:
    rows = _rows()
    assert rows
    for r in rows:
        assert isinstance(r, MathCodePair)
        assert r.latex_expression
        assert r.solution and "def calculate" in r.solution
        assert len(r.test_cases) == 5
        assert r.equation_type == "polynomial"
        assert r.domain == "Mathematics_General"
        assert r.synthetic is True
        assert r.metadata["slice"] == "polynomial"
        assert r.metadata["coefficient_kind"] == "integer"
        parse_number(str(r.test_cases[0].output))  # committed output parseable
        assert isinstance(r.complexity, int) and 1 <= r.complexity <= 5


def test_invariant_correctness_recomputed() -> None:
    rows = _rows(24)
    assert rows
    for r in rows:
        meta = r.metadata
        kind = meta["kind"]
        poly = sp.sympify(meta["poly"])
        if kind == "vieta_sum":
            expected = vieta_sum(poly)
        elif kind == "vieta_prod":
            expected = vieta_product(poly)
        elif kind == "coeff":
            expected = coeff_of(poly, int(meta["k"]))
        elif kind == "disc":
            expected = discriminant_of(poly)
        elif kind == "res":
            expected = resultant_of(poly, sp.sympify(meta["poly2"]))
        else:  # pragma: no cover
            raise AssertionError(f"unknown kind {kind}")
        got = sp.sympify(r.sympy_exp or "0")
        assert sp.simplify(expected - got) == 0, (kind, r.task_id)
        # exactness: integer/rational ground truth, no floats, no symbols
        assert expected.is_finite
        assert expected.free_symbols == set()


def test_vieta_gate_against_actual_roots() -> None:
    """Vieta rows must agree with the roots the polynomial was built from."""
    seen = 0
    for r in _rows(40):
        if r.metadata["kind"].startswith("vieta"):
            seen += 1
            poly = sp.sympify(r.metadata["poly"])
            actual = list(sp.roots(poly).keys())
            value = sp.sympify(r.sympy_exp or "0")
            if r.metadata["kind"] == "vieta_sum":
                assert sp.simplify(value - sp.Add(*actual)) == 0
            else:
                prod: sp.Expr = sp.Integer(1)
                for root in actual:
                    prod *= root
                assert sp.simplify(value - prod) == 0
    assert seen > 0  # the family must actually emit vieta rows


def test_oracle_acceptance() -> None:
    rows = _rows(6)
    assert rows
    with SandboxPool(n_workers=2, timeout_s=10, memory_mb=2048) as pool:
        from math2code.data.oracle import oracle_verify

        for r in rows:
            ok, reasons = oracle_verify(r, r.solution or "", pool=pool)
            assert ok, (r.task_id, reasons)


def test_no_zero_or_nonfinite_invariants() -> None:
    rows = _rows(40)
    assert rows
    for r in rows:
        got = sp.sympify(r.sympy_exp or "0")
        assert got != 0, r.task_id  # degenerate invariants are rejected
        assert got.is_finite
        assert not got.has(sp.oo, sp.nan)


def test_metadata_is_json_serializable() -> None:
    """Regression: metadata must round-trip through json.dumps (the generation
    pipeline writes rows as JSONL — non-serializable values silently break it).
    """
    rows = _rows(40)
    assert rows
    for r in rows:
        try:
            json.dumps(r.metadata)
        except TypeError as exc:
            raise AssertionError(f"{r.task_id}: {exc}") from exc
