"""Extended geometry family (docs/SYNTHETIC_EXPANSION.md §Tier 2).

Constant-output rows (pattern B) for scalar geometric quantities computed
from CONCRETE integer parameters: Euclidean distance between two 2D points
(and its square), triangle area by vertex coordinates (shoelace) and by side
lengths (Heron), circle circumference/area (pi-floats), rectangle
perimeter/area, and the angle between two vectors (radians).

Every query is rendered via `latex_override` — the notation (named operators
like ``\\operatorname{{dist}}``, norms, concrete Heron/shoelace formulas) is
not derivable from a single AST, and the override strings fully encode the
concrete parameters so each row is self-contained.

Ground truth is a constant AST; degenerate objects (zero distance/area,
collinear vertices, non-triangle side triples) are rejected by the gate.
"""

from __future__ import annotations

import math
import random
from typing import Any

import sympy as sp

from math2code.data.synthesizer.core import SynthFamily, int_seed
from math2code.schemas import MathCodePair

# kind -> params sampled per object (all ints, small, positive unless noted)
_KINDS = (
    "dist",
    "dist_sq",
    "tri_coord",
    "tri_heron",
    "circumference",
    "circle_area",
    "perimeter",
    "rect_area",
    "angle",
)


def _ints(rng: random.Random, n: int, lo: int = 1, hi: int = 10) -> list[int]:
    return [int(rng.randint(lo, hi)) for _ in range(n)]


def _recompute(kind: str, p: dict[str, int]) -> float:
    """Independent pure-Python recompute of the scalar from raw params.

    Deliberately a different code path from the SymPy ground truth so the
    generate-time gate is a real cross-check, not a tautology.
    """
    if kind in ("dist", "dist_sq"):
        d2 = (p["x2"] - p["x1"]) ** 2 + (p["y2"] - p["y1"]) ** 2
        return float(d2) if kind == "dist_sq" else math.sqrt(d2)
    if kind == "tri_coord":
        return (
            abs(
                p["x1"] * (p["y2"] - p["y3"])
                + p["x2"] * (p["y3"] - p["y1"])
                + p["x3"] * (p["y1"] - p["y2"])
            )
            / 2.0
        )
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
        nu = math.hypot(p["u1"], p["u2"])
        nv = math.hypot(p["v1"], p["v2"])
        return math.acos(max(-1.0, min(1.0, dot / (nu * nv))))
    raise AssertionError(f"unknown kind {kind}")


def _build_ast(kind: str, p: dict[str, int]) -> sp.Expr:
    """SymPy ground-truth AST for the concrete parameters."""
    if kind == "dist":
        return sp.sqrt((p["x2"] - p["x1"]) ** 2 + (p["y2"] - p["y1"]) ** 2)
    if kind == "dist_sq":
        return sp.Integer((p["x2"] - p["x1"]) ** 2 + (p["y2"] - p["y1"]) ** 2)
    if kind == "tri_coord":
        return (
            sp.Abs(
                p["x1"] * (p["y2"] - p["y3"])
                + p["x2"] * (p["y3"] - p["y1"])
                + p["x3"] * (p["y1"] - p["y2"])
            )
            / 2
        )
    if kind == "tri_heron":
        a, b, c = p["a"], p["b"], p["c"]
        s = sp.Rational(a + b + c, 2)
        return sp.sqrt(s * (s - a) * (s - b) * (s - c))
    if kind == "circumference":
        return 2 * sp.pi * p["r"]
    if kind == "circle_area":
        return sp.pi * p["r"] ** 2
    if kind == "perimeter":
        return sp.Integer(2 * (p["a"] + p["b"]))
    if kind == "rect_area":
        return sp.Integer(p["a"] * p["b"])
    if kind == "angle":
        dot = p["u1"] * p["v1"] + p["u2"] * p["v2"]
        nu = sp.sqrt(p["u1"] ** 2 + p["u2"] ** 2)
        nv = sp.sqrt(p["v1"] ** 2 + p["v2"] ** 2)
        return sp.acos(sp.Rational(dot, 1) / (nu * nv))
    raise AssertionError(f"unknown kind {kind}")


def _frac_tex(q: sp.Rational) -> str:
    """Render a rational as TeX (plain int when denominator is 1)."""
    if q.denominator == 1:
        return str(q.numerator)
    return rf"\frac{{{q.numerator}}}{{{q.denominator}}}"


def _overrides(kind: str, p: dict[str, int]) -> list[str]:
    """Two self-contained LaTeX surfaces per kind (encode all parameters)."""
    if kind in ("dist", "dist_sq"):
        sup = "" if kind == "dist" else "^{2}"
        return [
            rf"\operatorname{{dist}}{sup}\left(({p['x1']},{p['y1']}),\;"
            rf"({p['x2']},{p['y2']})\right)",
            rf"\left\|({p['x2']},{p['y2']})-({p['x1']},{p['y1']})\right\|{sup}",
        ]
    if kind == "tri_coord":
        return [
            rf"\operatorname{{area}}_{{\triangle}}\left(({p['x1']},{p['y1']}),\;"
            rf"({p['x2']},{p['y2']}),\;({p['x3']},{p['y3']})\right)",
            rf"\frac{{1}}{{2}}\left|{p['x1']}({p['y2']}-{p['y3']})"
            rf"+{p['x2']}({p['y3']}-{p['y1']})"
            rf"+{p['x3']}({p['y1']}-{p['y2']})\right|",
        ]
    if kind == "tri_heron":
        a, b, c = p["a"], p["b"], p["c"]
        s = sp.Rational(a + b + c, 2)
        st = _frac_tex(s)
        return [
            rf"\operatorname{{area}}_{{\triangle}}\left({a},\;{b},\;{c}\right)",
            rf"\sqrt{{{st}({st}-{a})({st}-{b})({st}-{c})}}",
        ]
    if kind == "circumference":
        return [
            rf"\operatorname{{circumference}}\left({p['r']}\right)",
            rf"2\pi\cdot{p['r']}",
        ]
    if kind == "circle_area":
        return [
            rf"\operatorname{{area}}_{{\circ}}\left({p['r']}\right)",
            rf"\pi\cdot{p['r']}^{{2}}",
        ]
    if kind == "perimeter":
        return [
            rf"\operatorname{{perimeter}}\left({p['a']},\;{p['b']}\right)",
            rf"2\left({p['a']}+{p['b']}\right)",
        ]
    if kind == "rect_area":
        return [
            rf"\operatorname{{area}}_{{\square}}\left({p['a']},\;{p['b']}\right)",
            rf"{p['a']}\cdot{p['b']}",
        ]
    if kind == "angle":
        return [
            rf"\operatorname{{angle}}\left(({p['u1']},{p['u2']}),\;"
            rf"({p['v1']},{p['v2']})\right)",
            rf"\arccos\left(\frac{{{p['u1']}\cdot{p['v1']}+{p['u2']}\cdot{p['v2']}}}"
            rf"{{\sqrt{{{p['u1']}^{{2}}+{p['u2']}^{{2}}}}"
            rf"\sqrt{{{p['v1']}^{{2}}+{p['v2']}^{{2}}}}}}\right)",
        ]
    raise AssertionError(f"unknown kind {kind}")


class GeometryExtFamily(SynthFamily):
    """Concrete-parameter scalar geometry queries (constant-output rows)."""

    domain = "Mathematics_Geometry"
    equation_type = "geometry"

    def _gate(self, result_expr: sp.Expr) -> str:
        return (
            "value recomputed from the concrete coordinates/parameters "
            "(construction identity; independent pure-Python recompute at "
            "generate time must match the SymPy ground truth)"
        )

    def _one_object(
        self, rng: random.Random
    ) -> tuple[str, dict[str, int], sp.Expr, list[str]] | None:
        """Sample one (kind, params, result AST, overrides); None when gated."""
        kind = _KINDS[int(rng.randrange(len(_KINDS)))]
        if kind in ("dist", "dist_sq"):
            x1, y1, x2, y2 = _ints(rng, 4)
            if (x1, y1) == (x2, y2):
                return None  # zero distance is degenerate
            p: dict[str, int] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        elif kind == "tri_coord":
            x1, y1, x2, y2, x3, y3 = _ints(rng, 6)
            p = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "x3": x3, "y3": y3}
            if _recompute(kind, p) == 0.0:
                return None  # collinear vertices -> zero area
        elif kind == "tri_heron":
            a, b, c = _ints(rng, 3)
            if not (a + b > c and a + c > b and b + c > a):
                return None  # strict triangle inequality (real positive area)
            p = {"a": a, "b": b, "c": c}
        elif kind in ("circumference", "circle_area"):
            p = {"r": int(rng.randint(1, 10))}
        elif kind in ("perimeter", "rect_area"):
            p = {"a": int(rng.randint(1, 10)), "b": int(rng.randint(1, 10))}
        elif kind == "angle":
            u1, u2, v1, v2 = _ints(rng, 4)
            p = {"u1": u1, "u2": u2, "v1": v1, "v2": v2}
        else:  # pragma: no cover - _KINDS guards this
            raise AssertionError(f"unknown kind {kind}")

        result = _build_ast(kind, p)
        # gate: finite, real, non-negative; defensive cross-check vs the
        # independent pure-Python recompute; angle in [0, pi]
        try:
            val = float(sp.N(result))
        except Exception:
            return None
        if not result.is_number or result.has(sp.I) or not result.is_finite:
            return None
        if val < 0.0 or not math.isfinite(val):
            return None
        if kind == "angle" and not (0.0 <= val <= math.pi + 1e-12):
            return None
        if abs(val - _recompute(kind, p)) > 1e-8 * max(1.0, abs(val)):
            return None
        return (kind, p, result, _overrides(kind, p))

    def generate(
        self, seed: int, prefix: str, count: int, **opts: Any
    ) -> list[MathCodePair]:
        """Deterministically produce `count` objects (2 rows each)."""
        rng = random.Random(int_seed(f"geometry_ext:{seed}"))
        out: list[MathCodePair] = []
        i = 0
        while len(out) < count * 2 and i < count * 60:  # cap attempts for gates
            i += 1
            obj = self._one_object(rng)
            if obj is None:
                continue
            kind, p, result, overrides = obj
            rows = self._build_pair(
                f"{prefix}_{kind}_{i}",
                result,  # unused for rendering; latex_override wins
                result,
                [],  # constant-output rows (no input variables)
                int_seed(f"geometry_ext:{seed}:{i}"),
                n_variants=1,  # variant list is provided via latex_override
                meta={
                    "slice": "geometry",
                    "kind": kind,
                    "vocab": kind,
                    "coefficient_kind": "integer",
                    "params": p,
                },
                equation_type="geometry",
                latex_override=overrides,
            )
            out.extend(rows)
        return out[: count * 2]
