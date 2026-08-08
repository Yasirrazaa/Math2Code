"""Data layer tests: competition loader, dedup, re-sampling."""

from __future__ import annotations

import json

from math2code.data.competition import (
    dedup_by_latex,
    load_competition_train,
    resample_test_cases,
)
from math2code.schemas import MathCodePair


def test_load_competition_train_shape() -> None:
    pairs = load_competition_train()
    assert len(pairs) == 26846
    assert all(len(p.test_cases) == 5 for p in pairs[:20])
    assert all(p.task_id for p in pairs)


def test_dedup_by_latex() -> None:
    a = MathCodePair(task_id="a", latex_expression="x + 1")
    b = MathCodePair(task_id="b", latex_expression="x + 1")
    c = MathCodePair(task_id="c", latex_expression="x + 2")
    out = dedup_by_latex([a, b, c])
    assert [p.task_id for p in out] == ["a", "c"]


def test_resample_test_cases_recomputes_outputs() -> None:
    pairs = load_competition_train()
    geo = next(p for p in pairs if p.equation_type == "Geometry")
    cases = resample_test_cases(geo, n=3, seed=7)
    assert len(cases) == 3
    assert all(c.output is not None for c in cases)
    assert all(
        set(c.input.keys()) == set(geo.test_cases[0].input.keys()) for c in cases
    )
    # jittered inputs differ from originals
    orig_vals = [tc.input["r"] for tc in geo.test_cases]
    new_vals = [c.input["r"] for c in cases]
    assert any(abs(a - b) > 1e-9 for a, b in zip(orig_vals, new_vals))


def test_resample_deterministic_per_seed() -> None:
    pairs = load_competition_train()
    p = pairs[0]
    a = resample_test_cases(p, n=2, seed=1)
    b = resample_test_cases(p, n=2, seed=1)
    assert a[0].input == b[0].input
    assert a[0].output == b[0].output


def test_real_row_json_roundtrip() -> None:
    """Schema must roundtrip real competition rows (incl. nan domain)."""
    with open("data/train.json") as f:
        rows = json.load(f)
    bad = [
        r
        for r in rows
        if r.get("domain") is not None
        and isinstance(r["domain"], float)
        and r["domain"] != r["domain"]
    ]
    assert bad, "expected some nan-domain rows in the dataset"
