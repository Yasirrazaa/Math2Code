"""Benchmark runner tests with stub backends (no GPU/API needed)."""

from __future__ import annotations

from math2code.evaluation.metrics import bootstrap_ci
from math2code.evaluation.runner import _run_problem, benchmark_model
from math2code.sandbox import SandboxPool
from math2code.schemas import MathCodePair, TestCase


class GoldStub:
    """Backend that emits the gold solution -> must score 1.0."""

    def __init__(self, pairs: list[MathCodePair]) -> None:
        self.by_latex = {p.latex_expression: p.solution for p in pairs}

    def complete(self, prompt: str) -> str:
        # find the latex in the prompt and look up the gold solution
        import re

        m = re.search(r"\n(.*?)\n\nFollow these guidelines", prompt, re.DOTALL)
        if not m:
            return "no latex found"
        return "```python\n" + (
            (self.by_latex.get(m.group(1).strip()) or "# unknown") + "\n```"
        )


class EmptyStub:
    def complete(self, prompt: str) -> str:
        return "I don't know how to solve this."


def _pair(tid: str, coeff: int) -> MathCodePair:
    return MathCodePair(
        task_id=tid,
        latex_expression=f"{coeff} \\cdot x + 1",
        sympy_exp=f"{coeff}*x + 1",
        solution=f"def calculate(x):\n    return {coeff} * x + 1",
        test_cases=[
            TestCase(input={"x": 1.0}, output=coeff * 1.0 + 1),
            TestCase(input={"x": 2.0}, output=coeff * 2.0 + 1),
            TestCase(input={"x": 3.0}, output=coeff * 3.0 + 1),
        ],
    )


def test_gold_stub_scores_perfect(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import json

    pairs = [_pair("a", 2), _pair("b", 5)]
    split = tmp_path / "split.json"
    split.write_text(json.dumps([p.model_dump() for p in pairs]))

    from math2code.evaluation.eval import load_split

    loaded = load_split(str(split))
    backend = GoldStub(loaded)
    result = benchmark_model(
        str(split), "hf:stub", "out.json", backend=backend, results_dir=tmp_path
    )
    assert result.per_problem_accuracy == 1.0


def test_empty_backend_scores_zero(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import json

    pairs = [_pair("a", 2)]
    split = tmp_path / "split.json"
    split.write_text(json.dumps([p.model_dump() for p in pairs]))

    from math2code.evaluation.runner import load_split, score_predictions  # noqa: F401

    pairs = load_split(str(split))
    backend = EmptyStub()
    with SandboxPool(n_workers=2, timeout_s=20) as pool:
        preds = [_run_problem(p, backend, pool) for p in pairs]
    assert all(p is None for p in preds[0])


def test_bootstrap_ci_deterministic() -> None:
    lo, hi = bootstrap_ci([True] * 90 + [False] * 10, n_resamples=500, seed=1)
    assert lo <= 0.9 <= hi
    assert 0.8 <= lo <= hi <= 1.0
    lo2, hi2 = bootstrap_ci([True] * 90 + [False] * 10, n_resamples=500, seed=1)
    assert (lo, hi) == (lo2, hi2)


def test_bootstrap_ci_perfect() -> None:
    lo, hi = bootstrap_ci([True] * 50, n_resamples=200)
    assert lo == 1.0 and hi == 1.0
