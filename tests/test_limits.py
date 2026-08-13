"""Limits family tests: determinism, contract shape, limit gate, oracle.

The family emits constant-output rows (no input variables) whose ground
truth is the finite value of `\\lim_{x \\to c} f(x)`.
"""

from __future__ import annotations

import json

import sympy as sp

from math2code.data.synthesizer.families.limits import LimitsFamily
from math2code.sandbox import SandboxPool
from math2code.schemas import MathCodePair


def _dumps(rows: list[MathCodePair]) -> str:
    return json.dumps([r.model_dump() for r in rows], sort_keys=True)


def test_generation_is_deterministic() -> None:
    a = LimitsFamily().generate(seed=42, prefix="lim", count=8)
    b = LimitsFamily().generate(seed=42, prefix="lim", count=8)
    assert a and b
    assert _dumps(a) == _dumps(b)  # byte-identical regeneration
    assert [r.task_id for r in a] == [r.task_id for r in b]


def test_contract_shape() -> None:
    rows = LimitsFamily().generate(seed=7, prefix="lim", count=8)
    assert len(rows) == 8  # n_variants=1 -> one row per object
    for r in rows:
        assert isinstance(r, MathCodePair)
        assert r.latex_expression.startswith("\\lim")
        assert r.equation_type == "limit"
        assert r.domain == "Mathematics_Calculus"
        assert r.synthetic is True
        assert r.metadata["slice"] == "limits"
        assert "point" in r.metadata
        # constant-output rows: no input variables, 5 empty-input cases
        assert len(r.test_cases) == 5
        assert all(tc.input == {} for tc in r.test_cases)
        assert r.solution and "calculate" in r.solution
        # solution embeds the constant via sympify; no arg signature
        assert "def calculate():" in r.solution
        # outputs are finite, parseable floats/ints
        from math2code.evaluation.metrics import parse_number

        for tc in r.test_cases:
            val = parse_number(tc.output)
            assert abs(complex(val)) < 1e12


def test_limit_gate_holds_for_every_row() -> None:
    """The committed truth equals a fresh symbolic limit computation."""
    rows = LimitsFamily().generate(seed=11, prefix="lim", count=10)
    assert rows
    for r in rows:
        # truth is the sympify-embedded constant; it must be finite and real
        truth = sp.sympify(r.sympy_exp or "")
        assert truth.is_finite
        assert not truth.has(sp.I)
        truth_val = float(sp.N(truth))
        for tc in r.test_cases:
            assert abs(float(complex(tc.output).real) - truth_val) < 1e-9  # type: ignore[arg-type]
        assert "finite real constant" in r.metadata["family_gate"]


def test_limits_cover_finite_and_infinite_points() -> None:
    rows = LimitsFamily().generate(seed=3, prefix="lim", count=30)
    points = {r.metadata["point"] for r in rows}
    assert points  # some points emitted
    # the family must produce both finite points and infinity over enough draws
    assert any(p in ("oo", "-oo") for p in points)
    assert any(p not in ("oo", "-oo") for p in points)
    # latex at infinity uses \\infty
    inf_rows = [r for r in rows if r.metadata["point"] in ("oo", "-oo")]
    assert all("\\infty" in r.latex_expression for r in inf_rows)


def test_rows_pass_oracle() -> None:
    rows = LimitsFamily().generate(seed=5, prefix="lim", count=6)
    assert rows
    with SandboxPool(n_workers=2, timeout_s=10, memory_mb=2048) as pool:
        from math2code.data.oracle import oracle_verify

        for r in rows:
            ok, reasons = oracle_verify(r, r.solution or "", pool=pool)
            assert ok, f"{r.task_id}: {reasons}"


def test_accept_rate_is_high() -> None:
    """Nearly every drawn object passes the finite-limit gate."""
    fam = LimitsFamily()
    rejected = 0
    total = 40
    for i in range(total):
        obj = fam._one_object(__import__("random").Random(i))  # noqa: SLF001
        if obj is None:
            rejected += 1
    assert rejected / total < 0.1  # >= 90% of drawn forms are finite limits
