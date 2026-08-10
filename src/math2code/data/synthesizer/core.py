"""Core of the code-first synthesizer.

A `SynthFamily` builds the ground truth as a SymPy AST, renders the problem in
multiple LaTeX notations, emits executable solution code, and samples
domain-aware test cases. Rows are `MathCodePair`s that satisfy the same
contract as the competition data — the oracle and the competition metric can
score them unmodified.

Anti-bug gates (applied at build time, before a row can be emitted):

1. **sympify round-trip** — the solution code evaluates
   `sp.sympify('<sympy-str>')`; rows whose sympy-str does not round-trip
   through `sympify` (Piecewise, Eq, non-elementary results) are rejected:
   bare `I`/`pi`/`sqrt` are not defined in the sandbox namespace, so the
   expression must be constructed via sympify's parser.
2. **family-specific symbolic identity** — e.g. `diff(F) - f == 0` for
   integrals. Families override `_gate`.
3. **finiteness + pole avoidance** — test inputs are sampled where the ground
   truth evaluates to a finite number, away from denominator roots.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import sympy as sp

from math2code.data.synthesizer.printer import int_seed, render_variants
from math2code.data.synthesizer.sampler import sample_inputs
from math2code.schemas import MathCodePair, TestCase


def complexity_of(expr: sp.Expr) -> int:
    """Map sympy op-count to the competition's c1..c5 scale."""
    ops = expr.count_ops()
    for level, cap in enumerate((2, 6, 12, 20), start=1):
        if ops <= cap:
            return level
    return 5


def _to_complex(val: sp.Expr) -> complex:
    """Evalf'd numeric sympy expr -> python complex (safe for I-containing)."""
    try:
        return complex(val)
    except Exception:
        return float(sp.re(val)) + float(sp.im(val)) * 1j


def roundtrips(expr: sp.Expr) -> bool:
    """Can `sp.sympify(str(expr))` reconstruct the expression?"""
    try:
        return bool(sp.sympify(sp.sstr(expr)) == expr)
    except Exception:
        return False


def solution_code(
    result_expr: sp.Expr, variables: list[sp.Symbol], complex_out: bool
) -> str:
    """Emit a `calculate(...)` function that evaluates `result_expr` at inputs.

    The expression is embedded as a string parsed by `sp.sympify` at import
    time — the only way to reference constants (I, pi, E, sqrt) that are not
    defined in the sandbox namespace. Mirrors the competition gold style.
    """
    sig = ", ".join(map(str, variables))
    syms = "\n".join(f"    {v}_s = sp.Symbol('{v}')" for v in variables)
    subs = ", ".join(f"{v}_s: {v}" for v in variables)
    cast = "complex" if complex_out else "float"
    lines = [
        "import sympy as sp",
        f"_expr = sp.sympify({sp.sstr(result_expr)!r})",
        f"def calculate({sig}):",
    ]
    if variables:
        lines.append(syms)
        lines.append(f"    return {cast}(_expr.subs({{{subs}}}))")
    else:
        lines.append(f"    return {cast}(_expr)")
    return "\n".join(lines)


class SynthFamily(ABC):
    """Base for programmatic families. `generate` must be seed-deterministic."""

    domain: str = "Mathematics_General"
    equation_type: str = "general"

    def _build_pair(
        self,
        task_id: str,
        problem_expr: sp.Expr,
        result_expr: sp.Expr,
        variables: list[sp.Symbol],
        seed: int,
        n_variants: int = 2,
        n_cases: int = 5,
        meta: dict[str, Any] | None = None,
    ) -> list[MathCodePair]:
        """One math object -> one row per LaTeX notation variant.

        Each variant shares the task base id (suffixed `:v0`, `:v1`, ...) so
        variant families stay traceable. Expected outputs come from the
        ground-truth AST at sampled inputs.
        """
        if not roundtrips(result_expr):
            return []  # gate 1: solution code would not construct the expr
        variants = render_variants(problem_expr, seed, n_variants=n_variants)
        inputs = sample_inputs(
            result_expr, variables, n_cases, int_seed(f"pair:{task_id}")
        )
        if not inputs:
            return []

        expected = []
        for inp in inputs:
            subs = {v: sp.Float(inp[str(v)]) for v in variables}
            expected.append(_to_complex(sp.N(result_expr.subs(subs))))

        complex_out = any(abs(c.imag) > 1e-12 for c in expected)
        code = solution_code(result_expr, variables, complex_out)
        gate = self._gate(result_expr)

        # complex is not JSON-serializable: real outputs stay floats (competition
        # convention), complex outputs become the canonical 're+imj' string that
        # `parse_number` round-trips (schema allows `Number | str`).
        def _out(c: complex) -> int | float | str:
            if abs(c.imag) <= 1e-12:
                return float(c.real)
            from math2code.evaluation.metrics import format_output

            return format_output(c)

        pairs: list[MathCodePair] = []
        for i, tex in enumerate(variants):
            pairs.append(
                MathCodePair(
                    task_id=f"{task_id}:v{i}",
                    latex_expression=tex,
                    solution=code,
                    sympy_exp=sp.sstr(result_expr),
                    test_cases=[
                        TestCase(input=inp, output=_out(out))
                        for inp, out in zip(inputs, expected)
                    ],
                    domain=self.domain,
                    equation_type=self.equation_type,
                    complexity=complexity_of(result_expr),
                    output_type="complex" if complex_out else "real",
                    synthetic=True,
                    metadata={
                        "source": "code_first_synth",
                        "family": self.equation_type,
                        "variant": i,
                        "n_variants": len(variants),
                        "family_gate": gate,
                        **(meta or {}),
                    },
                )
            )
        return pairs

    def _gate(self, result_expr: sp.Expr) -> str:
        """Describe the symbolic identity asserted about the ground truth."""
        return "sympify round-trip"

    @abstractmethod
    def generate(
        self, seed: int, prefix: str, count: int, **opts: Any
    ) -> list[MathCodePair]:
        """Deterministically produce `count` math objects (each with variants)."""
