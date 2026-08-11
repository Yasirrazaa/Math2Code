"""Tests for the set-cardinality family (constant-output, exact integers).

Verifies determinism, contract shape, exact recomputation of every committed
output from the endpoints/sets in metadata, the gated flag, solution-truth
agreement, and oracle acceptance in the sandbox.
"""

from __future__ import annotations

import sympy as sp

from math2code.data.oracle import oracle_verify
from math2code.data.synthesizer.families.sets_cardinality import (
    SetsCardinalityFamily,
)
from math2code.evaluation.metrics import parse_number
from math2code.sandbox import SandboxPool, execute_code
from math2code.schemas import MathCodePair

_FAMILY = SetsCardinalityFamily()
_NAME = "SetsCardinalityFamily"
_KINDS = {
    "interval_intersect",
    "interval_union",
    "set_union",
    "set_intersect",
    "membership",
}


def _gen(count: int = 6, seed: int = 42) -> list[MathCodePair]:
    return _FAMILY.generate(seed, prefix="t", count=count)


def _recompute(meta: dict) -> int:
    """Independently recompute the ground truth from the row's metadata."""
    kind = meta["kind"]
    if kind == "interval_intersect":
        return max(0, min(meta["b"], meta["d"]) - max(meta["a"], meta["c"]))
    if kind == "interval_union":
        return max(meta["b"], meta["d"]) - min(meta["a"], meta["c"])
    if kind in ("set_union", "set_intersect"):
        sa, sb = set(meta["set_a"]), set(meta["set_b"])
        return len(sa | sb) if kind == "set_union" else len(sa & sb)
    if kind == "membership":
        return 1 if meta["a"] <= meta["x"] <= meta["b"] else 0
    raise AssertionError(f"unknown kind {kind!r}")


def test_determinism() -> None:
    a = _gen(6, seed=42)
    b = _gen(6, seed=42)
    assert [(r.task_id, r.latex_expression) for r in a] == [
        (r.task_id, r.latex_expression) for r in b
    ]
    assert a


def test_contract_shape() -> None:
    rows = _gen(25)
    assert rows, "generate must produce rows"
    assert len(rows) == 25  # 25 objects * 1 latex variant
    kinds = set()
    for r in rows:
        assert isinstance(r, MathCodePair)
        assert r.latex_expression
        assert r.solution and "def calculate" in r.solution
        assert len(r.test_cases) == 5
        assert all(tc.input == {} for tc in r.test_cases), "constant-row inputs"
        assert r.equation_type == "sets"
        assert r.domain == "Mathematics_General"
        assert r.synthetic is True
        assert r.metadata["slice"] == "sets"
        assert r.metadata["gated"] is True, "gated portfolio family"
        kinds.add(r.metadata["kind"])
        assert r.metadata["kind"] in _KINDS
        assert r.output_type == "real"
        for tc in r.test_cases:
            assert tc.output is not None
            assert abs(complex(parse_number(str(tc.output))).imag) < 1e-12
            assert float(tc.output) == int(float(tc.output)), "exact integer output"
    assert kinds == _KINDS, "all five kinds must appear across a larger draw"


def test_committed_outputs_recomputed() -> None:
    """Every committed output must equal the exact recomputation from metadata."""
    rows = _gen(25, seed=42)
    assert rows
    kinds = {r.metadata["kind"] for r in rows}
    assert kinds == _KINDS, kinds
    for r in rows:
        expected = _recompute(r.metadata)
        assert expected >= 0, r.task_id
        for tc in r.test_cases:
            assert float(tc.output) == float(expected), r.task_id


def test_gate_and_latex_surfaces() -> None:
    rows = _gen(12)
    assert rows
    for r in rows:
        assert "recomputed from the set/interval endpoints" in r.metadata[
            "family_gate"
        ]
        # ground truth sympy_exp must be an exact finite Integer
        truth = sp.sympify(r.sympy_exp or "")
        assert isinstance(truth, sp.Integer), r.sympy_exp
        assert truth.is_finite
    surfaces = {r.latex_expression for r in rows}
    assert len(surfaces) == len(rows), "all rows must have distinct latex"
    assert any("\\cap" in s for s in surfaces), "intersection surfaces present"
    assert any("\\cup" in s for s in surfaces), "union surfaces present"
    assert any("\\in" in s for s in surfaces), "membership surfaces present"


def test_solution_matches_truth() -> None:
    """Executing each solution must reproduce the committed output."""
    for r in _gen(6):
        res = execute_code(r.solution or "", inputs={})
        assert res.ok, res.stderr
        assert abs(float(res.stdout) - float(r.test_cases[0].output)) < 1e-9


def test_oracle_verification() -> None:
    rows = _gen(5)
    assert rows
    with SandboxPool(n_workers=2, timeout_s=10, memory_mb=2048) as pool:
        for r in rows:
            ok, reasons = oracle_verify(r, r.solution or "", pool=pool)
            assert ok, f"{r.task_id}: {reasons}"


def test_interval_union_is_single_interval() -> None:
    """Union rows must satisfy the overlap gate (c <= b and a <= d)."""
    for r in _gen(12, seed=3):
        if r.metadata["kind"] != "interval_union":
            continue
        m = r.metadata
        assert m["c"] <= m["b"] and m["a"] <= m["d"], r.task_id
