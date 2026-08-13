"""elementary_ext family tests: determinism, contract shape, domain safety,
committed-output ground truth, and oracle acceptance.
"""

from __future__ import annotations

import cmath
import json

import sympy as sp

from math2code.data.synthesizer.families.elementary_ext import (
    NEW_VOCAB,
    ElementaryExtFamily,
)
from math2code.evaluation.metrics import parse_number
from math2code.sandbox import SandboxPool
from math2code.schemas import MathCodePair

_X = sp.Symbol("x")


def _rows(count: int = 6, seed: int = 42) -> list[MathCodePair]:
    return ElementaryExtFamily().generate(seed=seed, prefix="elem", count=count)


def test_generation_is_deterministic() -> None:
    a = _rows(count=8)
    b = _rows(count=8)
    assert [r.task_id for r in a] == [r.task_id for r in b]
    assert [r.latex_expression for r in a] == [r.latex_expression for r in b]
    assert json.dumps([r.model_dump() for r in a], sort_keys=True) == json.dumps(
        [r.model_dump() for r in b], sort_keys=True
    )


def test_contract_shape() -> None:
    rows = _rows(count=6)
    assert rows
    for r in rows:
        assert r.latex_expression
        assert r.solution and "def calculate" in r.solution
        assert r.sympy_exp and sp.sympify(r.sympy_exp) is not None
        assert len(r.test_cases) == 5
        assert r.equation_type == "functions"
        assert r.domain == "Mathematics_General"
        assert r.synthetic is True
        assert r.metadata["slice"] == "vocab"
        assert r.metadata["coefficient_kind"] == "integer"
        assert 1 <= r.metadata["depth"] <= 3
        assert r.metadata["vocab"], "vocab metadata must be non-empty"
        for tc in r.test_cases:
            parse_number(tc.output)  # must not raise


def test_new_vocabulary_is_covered() -> None:
    # the family's whole point: functions the competition train never sees
    rows = _rows(count=60, seed=7)
    used = {name for r in rows for name in r.metadata["vocab"]}
    assert used & NEW_VOCAB
    for required in ("sech", "csch", "coth", "asinh", "acosh", "atanh"):
        assert required in used, f"{required} never sampled in seed 7"


def test_committed_outputs_match_ground_truth() -> None:
    rows = _rows(count=6)
    for r in rows:
        expr = sp.sympify(r.sympy_exp)
        for tc in r.test_cases:
            subs = {sp.Symbol(k): sp.Float(v) for k, v in tc.input.items()}
            truth = complex(sp.N(expr.subs(subs)))
            got = parse_number(tc.output)
            assert cmath.isclose(got, truth, rel_tol=1e-6, abs_tol=1e-9), (
                f"{r.task_id}: {got} != {truth}"
            )


def test_domain_safety() -> None:
    rows = _rows(count=40, seed=3)
    for r in rows:
        vocab = set(r.metadata["vocab"])
        x0 = r.test_cases[0].input.get("x", 0.0)
        if {"asin", "acos", "atanh"} & vocab:
            # principal-branch args stay strictly inside (-1, 1)
            assert abs(x0) <= 0.8, f"{r.task_id}: branch arg |x|={abs(x0)}"
        if "factorial" in vocab:
            assert isinstance(x0, int) and 3 <= x0 <= 10
        assert r.output_type == "real"
        for tc in r.test_cases:
            assert not isinstance(tc.output, str)  # no 're+imj' complex strings
            assert abs(complex(tc.output)) < 1e10  # type: ignore[arg-type]


def test_rows_pass_oracle() -> None:
    rows = _rows(count=4)
    with SandboxPool(n_workers=2, timeout_s=10, memory_mb=2048) as pool:
        from math2code.data.oracle import oracle_verify

        for r in rows:
            ok, reasons = oracle_verify(r, r.solution or "", pool=pool)
            assert ok, f"{r.task_id}: {reasons}"
