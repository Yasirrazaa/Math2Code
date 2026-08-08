"""Canonical data contract for Math2Code.

Every stage of the pipeline (generation, curation, training, evaluation, GRPO
rewards) speaks the same schema. This is the single source of truth that fixes
the historical `python_code` vs `solution` key drift.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Number = int | float | complex
InputMap = dict[str, int | float | complex]


class TestCase(BaseModel):
    """One execution probe: an input dict and (optionally) the expected output.

    `output` is `None` for public-test rows where answers are withheld.
    """

    input: InputMap = Field(default_factory=dict)
    output: Number | str | None = None


class MathCodePair(BaseModel):
    """A single LaTeX -> code training/evaluation sample."""

    task_id: str
    latex_expression: str
    solution: str | None = None  # canonical key: runnable python code
    sympy_exp: str | None = None  # ground-truth sympy expression (when available)
    test_cases: list[TestCase] = Field(default_factory=list)
    domain: str | None = None
    equation_type: str | None = None
    complexity: int | str | None = None
    output_type: Literal["real", "complex", "unknown", None] = None
    synthetic: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def from_competition_row(row: dict[str, Any]) -> MathCodePair:
    """Normalize a competition train.json / public_test row into the schema."""

    def _clean(v: Any) -> Any:
        # JSON NaN (some rows have domain: nan) -> None
        if isinstance(v, float) and v != v:
            return None
        return v

    test_cases = [
        TestCase(
            input={k: v for k, v in tc.get("input", {}).items()},
            output=tc.get("output"),
        )
        for tc in row.get("test_cases", [])
    ]
    return MathCodePair(
        task_id=row["task_id"],
        latex_expression=row.get("latex_expression", ""),
        solution=_clean(row.get("solution")),
        sympy_exp=_clean(row.get("sympy_exp")),
        test_cases=test_cases,
        domain=_clean(row.get("domain")),
        equation_type=_clean(row.get("equation_type")),
        complexity=_clean(row.get("complexity")),
        output_type=_clean(row.get("output_type", "unknown")) or "unknown",
        synthetic=row.get("synthetic"),
        metadata={
            k: v
            for k, v in row.items()
            if k
            not in {
                "task_id",
                "latex_expression",
                "solution",
                "sympy_exp",
                "test_cases",
            }
        },
    )
