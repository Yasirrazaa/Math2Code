"""Tests for the extended geometry family (concrete-parameter scalars)."""

from __future__ import annotations

import math

from math2code.data.oracle import oracle_verify
from math2code.data.synthesizer.families.geometry_ext import GeometryExtFamily
from math2code.evaluation.metrics import parse_number
from math2code.sandbox import SandboxPool
from math2code.schemas import MathCodePair


def _gen(count: int, seed: int = 7) -> list[MathCodePair]:
    return GeometryExtFamily().generate(seed=seed, prefix="g", count=count)


def _expected(kind: str, p: dict[str, int]) -> float:
    """Independent pure-Python recompute of the ground truth from params."""
    if kind in ("dist", "dist_sq"):
        d2 = (p["x2"] - p["x1"]) ** 2 + (p["y2"] - p["y1"]) ** 2
        return float(d2) if kind == "dist_sq" else math.sqrt(d2)
    if kind == "tri_coord":
        return abs(
            p["x1"] * (p["y2"] - p["y3"])
            + p["x2"] * (p["y3"] - p["y1"])
            + p["x3"] * (p["y1"] - p["y2"])
        ) / 2.0
    if kind == "tri_heron":
        a, b, c = p["a"], p["b"], p["c"]
        s = (a + b + c) / 2.0
        return math.sqrt(s * (s - a) * (s - b) * (s - c))
    if kind == "circumference":
        return 2.0 * math.pi * p["r"]
    if kind == "circle_area":
        return math.pi * p["r"] ** 2
    if kind == "perimeter":
        return 2.0 * (p["a"] + p["b"])
    if kind == "rect_area":
        return float(p["a"] * p["b"])
    if kind == "angle":
        dot = p["u1"] * p["v1"] + p["u2"] * p["v2"]
        return math.acos(dot / (math.hypot(p["u1"], p["u2"]) * math.hypot(p["v1"], p["v2"])))
    raise AssertionError(kind)


def test_deterministic() -> None:
    a = _gen(5, seed=42)
    b = _gen(5, seed=42)
    assert [r.task_id for r in a] == [r.task_id for r in b]
    assert [r.latex_expression for r in a] == [r.latex_expression for r in b]
    assert [r.test_cases[0].output for r in a] == [
        r.test_cases[0].output for r in b
    ]


def test_contract_shape() -> None:
    rows = _gen(6)
    assert rows
    for r in rows:
        assert isinstance(r, MathCodePair)
        assert r.latex_expression
        assert r.solution and "def calculate" in r.solution
        assert len(r.test_cases) == 5
        assert r.domain == "Mathematics_Geometry"
        assert r.equation_type == "geometry"
        assert r.synthetic is True
        assert r.metadata["slice"] == "geometry"
        assert r.metadata["coefficient_kind"] == "integer"
        assert isinstance(r.metadata["params"], dict)
        # every override encodes every concrete parameter
        for v in r.metadata["params"].values():
            assert str(v) in r.latex_expression, (r.task_id, r.latex_expression)
        # constant-output: all five committed outputs are identical + parseable
        outs = [tc.output for tc in r.test_cases]
        assert len(set(map(float, outs))) == 1  # type: ignore[arg-type]
        for o in outs:
            c = parse_number(o)  # type: ignore[arg-type]
            assert c.imag == 0 and math.isfinite(c.real)


def test_value_correctness_and_gate() -> None:
    """Committed output == independent recompute; no degenerate values."""
    rows = _gen(12)
    kinds = {r.metadata["kind"] for r in rows}
    for r in rows:
        kind = r.metadata["kind"]
        p = r.metadata["params"]
        expected = _expected(kind, p)
        for tc in r.test_cases:
            got = float(tc.output)  # type: ignore[arg-type]
            assert abs(got - expected) < 1e-8 * max(1.0, abs(expected)), (
                r.task_id,
                got,
                expected,
            )
        if kind in ("dist", "dist_sq", "tri_coord", "tri_heron"):
            assert expected > 0.0, (r.task_id, kind)  # non-degenerate
        if kind == "angle":
            assert 0.0 <= expected <= math.pi, (r.task_id, expected)
    # a random draw of 12 objects need not hit all 9 kinds; require broad
    # coverage plus the gate-heavy (degenerate-filtered) kinds specifically
    assert len(kinds) >= 6, kinds
    assert {"dist", "tri_coord", "tri_heron"}.issubset(kinds), kinds


def test_oracle_verification() -> None:
    rows = _gen(3)
    assert rows
    with SandboxPool(n_workers=2, timeout_s=10, memory_mb=2048) as pool:
        for r in rows:
            ok, reasons = oracle_verify(r, r.solution or "", pool=pool)
            assert ok, f"{r.task_id}: {reasons}"


def test_solution_matches_truth() -> None:
    from math2code.sandbox import execute_code

    for r in _gen(4):
        res = execute_code(r.solution or "", inputs={})
        assert res.ok, res.stderr
        got = float(res.stdout)
        expected = float(r.test_cases[0].output)  # type: ignore[arg-type]
        assert abs(got - expected) < 1e-9 * max(1.0, abs(expected)), r.task_id
