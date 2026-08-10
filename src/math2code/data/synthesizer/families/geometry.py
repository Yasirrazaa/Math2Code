"""Geometry families (docs/DATA_STRATEGY.md §3 P1; 10% of mix).

The competition train has only 31 geometry rows. Ground truth is the formula;
inputs are positive dimensions (sampler domain low > 0). Area/perimeter/
volume formulas in 1-3 variables; `\\pi` appears in the circle/sphere/cone
family (roundtrip: sympify knows `pi`).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import sympy as sp

from math2code.data.synthesizer.core import SynthFamily, int_seed
from math2code.schemas import MathCodePair

_L = sp.Symbol("l")
_W = sp.Symbol("w")
_H = sp.Symbol("h")
_R = sp.Symbol("r")
_B = sp.Symbol("b")
_S = sp.Symbol("s")
_A = sp.Symbol("a")


def _dec(rng: random.Random) -> sp.Float:
    return sp.Float(round(rng.uniform(0.5, 4.0), 2))


_TEMPLATES: list[
    tuple[str, str, Callable[[random.Random], sp.Expr], list[sp.Symbol]]
] = [
    ("rectangle_area", "geometry", lambda r: _dec(r) * _L * _W, [_L, _W]),
    ("rectangle_perimeter", "geometry", lambda r: 2 * (_L + _W), [_L, _W]),
    ("triangle_area", "geometry", lambda r: _dec(r) * _B * _H / 2, [_B, _H]),
    ("trapezoid_area", "geometry", lambda r: (_A + _B) * _H / 2, [_A, _B, _H]),
    ("square_perimeter", "geometry", lambda r: 4 * _dec(r) * _S, [_S]),
    ("box_volume", "geometry", lambda r: _dec(r) * _L * _W * _H, [_L, _W, _H]),
    ("circle_area", "geometry", lambda r: sp.pi * _dec(r) * _R**2, [_R]),
    ("circle_circumference", "geometry", lambda r: 2 * sp.pi * _R, [_R]),
    ("cylinder_volume", "geometry", lambda r: sp.pi * _R**2 * _H, [_R, _H]),
    ("cone_volume", "geometry", lambda r: sp.pi * _R**2 * _H / 3, [_R, _H]),
    ("sphere_volume", "geometry", lambda r: 4 * sp.pi * _R**3 / 3, [_R]),
]


class GeometryFamily(SynthFamily):
    """Positive-domain geometry formulas."""

    domain = "Mathematics_General"

    def _gate(self, result_expr: sp.Expr) -> str:
        return "ground-truth AST is the truth (numeric oracle cross-check)"

    def generate(
        self, seed: int, prefix: str, count: int, **opts: Any
    ) -> list[MathCodePair]:
        n_variants = int(opts.get("n_variants", 2))
        rng = random.Random(int_seed(f"geo:{seed}"))
        out: list[MathCodePair] = []
        i = 0
        while len(out) < count * 2 and i < count * 60:
            i += 1
            name, etype, builder, vars_ = _TEMPLATES[
                int(rng.randrange(len(_TEMPLATES)))
            ]
            expr = builder(rng)
            rows = self._build_pair(
                f"{prefix}_{name}_{i}",
                expr,
                expr,
                vars_,
                int_seed(f"geo:{seed}:{i}"),
                n_variants=n_variants,
                meta={"vocab": name, "slice": "geometry"},
                sample_kwargs={"low": 0.5, "high": 8.0},
                equation_type=etype,
            )
            out.extend(rows)
        return out[: count * 2]
