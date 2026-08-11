r"""Seeded LaTeX rendering with semantic notation mutation.

Code-first synthesis builds the ground truth as a SymPy AST, then renders the
*problem* as LaTeX. SymPy's default `LatexPrinter` is rigid — it renders every
expression the same way. To teach the model that notation is surface and math is
deep, we render the same AST in multiple notational forms (`e^{x}` vs `\exp(x)`,
`\\frac{d}{dx}` vs `y'`, `x y` vs `x \\cdot y`, ...).

Design rules (audited against sympy 1.14):

- `sp.latex(expr, printer=...)` is broken in sympy 1.14 (``TypeError: Unknown
  setting 'printer'``) — call ``Printer(settings).doprint(expr)`` directly.
- `sp.Integer` subclasses `sp.Rational`: any ``_print_Rational`` override must
  guard ``q == 1`` or integers render as ``\\frac{2}{1}``.
- Style choice is a **pure function of the seed** (sha256-derived int, so
  regeneration is byte-identical across processes — string seeds are
  PYTHONHASHSEED-random and must never be used for persisted rows).
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

import sympy as sp
from sympy.printing.latex import LatexPrinter


def int_seed(key: str) -> int:
    """Deterministic int seed from a string (hash-stable across processes)."""
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


class VariantPrinter(LatexPrinter):
    """LatexPrinter that chooses notational variants via a seeded RNG.

    Every choice point calls ``self.rng.random()`` so the whole render is a
    deterministic function of the rng (which callers seed per row).
    """

    def __init__(self, rng: random.Random, settings: dict[str, Any] | None = None):
        super().__init__(settings or {})
        self.rng = rng

    # -- guards ---------------------------------------------------------
    def _print_Rational(self, expr: sp.Rational) -> str:  # noqa: N802 (sympy dispatch)
        if expr.q == 1:  # sp.Integer subclasses sp.Rational — keep ints intact
            return str(self._print(expr.p))
        return str(super()._print_Rational(expr))

    # -- notation variants ----------------------------------------------
    def _print_exp(self, expr: sp.exp) -> str:
        arg = self._print(expr.args[0])
        roll = self.rng.random()
        if roll < 0.45:
            return f"e^{{{arg}}}"
        if roll < 0.8:
            return f"\\exp\\left({arg}\\right)"
        return f"\\operatorname{{exp}}\\left({arg}\\right)"

    def _print_Derivative(self, expr: sp.Derivative) -> str:  # noqa: N802 (sympy dispatch)
        # expr.args == (f, (x, n), ...) for higher orders
        func = expr.expr
        var, n = expr.variable_count[0]
        roll = self.rng.random()
        f_tex = self._print(func)
        # Lagrange prime notation uses the function's own name: y' for y(x),
        # \sin' for sin(x). Fall back to f' for nameless composites.
        fname = getattr(getattr(func, "func", None), "__name__", None)
        if roll < 0.35:
            if fname:
                return f"{fname}'" if n == 1 else f"{fname}^{{({n})}}"
            return "f'" if n == 1 else f"f^{{({n})}}"
        if roll < 0.6 and n == 1:
            return f"\\frac{{d}}{{\\,d {self._print(var)}}} {f_tex}"
        if n == 1:
            return f"\\frac{{d {f_tex}}}{{d {self._print(var)}}}"
        return f"\\frac{{d^{{{n}}} {f_tex}}}{{d {self._print(var)}^{{{n}}}}}"

    def _print_log(self, expr: sp.log) -> str:
        arg = expr.args[0]
        base = expr.args[1] if len(expr.args) > 1 else None
        if base is not None and base.is_integer and base.is_positive:
            if self.rng.random() < 0.5:
                return f"\\log_{{{self._print(base)}}}\\left({self._print(arg)}\\right)"
        if self.rng.random() < 0.5:
            return f"\\ln\\left({self._print(arg)}\\right)"
        return f"\\log\\left({self._print(arg)}\\right)"

    def _print_sqrt(self, expr: sp.sqrt) -> str:
        if self.rng.random() < 0.3:
            return f"{self._print(expr.args[0])}^{{\\frac{{1}}{{2}}}}"
        return str(super()._print_sqrt(expr))


# Settings variants for the whole render (mul_symbol is a sympy setting).
_SETTINGS = [
    {},
    {"mul_symbol": "dot"},
    {"mul_symbol": "times"},
]


def repr_wrapped_tex(expr: sp.Expr) -> str:
    """Competition repr-wrapped surface: ``\\mathtt{{\\text{{Integral(f, x)}}}}``.

    The frozen test renders calculus questions as a wrapped python repr of the
    sympy AST string (``\\mathtt{{\\text{{Integral(7*x + ...)}}}}``)
    rather than ``\\int``/Leibniz notation. `sp.sstr` gives exactly that repr
    text; the wrapper matches the competition surface token-for-token.
    """
    return rf"\mathtt{{\text{{{sp.sstr(expr)}}}}}"


def render_variants(expr: sp.Expr, seed: int, n_variants: int = 2) -> list[str]:
    """Deterministic notation variants of `expr`.

    Returns up to `n_variants` distinct LaTeX strings. Variant 0 is sympy's
    default render; the rest come from a seeded VariantPrinter. Regeneration
    with the same seed is byte-identical (pure function of seed + expr).
    """
    rng = random.Random(int_seed(f"printer:{seed}"))
    out: list[str] = []
    out.append(LatexPrinter().doprint(expr))
    for i in range(1, max(1, n_variants)):
        settings = _SETTINGS[i % len(_SETTINGS)]
        tex = VariantPrinter(rng, settings).doprint(expr)
        if tex not in out:
            out.append(tex)
    return out
