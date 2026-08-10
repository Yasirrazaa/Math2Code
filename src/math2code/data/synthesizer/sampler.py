"""Domain-aware test-case sampling for generated rows.

Extends the competition's `jitter_inputs` with the guarantees the strategy
doc requires:

- pole avoidance: exclude inputs within `pole_margin` of any denominator root
- finiteness: only emit inputs where the ground truth evaluates to a finite
  number (rejects singularity/branch-cut hits)
- int preservation: integer-only variables stay integers (diophantine-style)
- magnitude control: optional log-uniform range for numeric-stability rows
- coupling: `x`/`x_val` key coupling is preserved by the caller (families emit
  clean variable names; the pool's `_match_inputs` handles `_val` variants)

All draws come from an int seed — byte-identical regeneration.
"""

from __future__ import annotations

import random
from typing import Any

import sympy as sp

from math2code.data.synthesizer.printer import int_seed

MAX_ATTEMPTS = 200


def denominator_roots(expr: sp.Expr) -> list[sp.Expr]:
    """Approximate real roots of the expression's denominator (for poles)."""
    denom = sp.denom(expr)
    if denom in (1, sp.Integer(1)):
        return []
    roots: list[sp.Expr] = []
    for r in sp.solve(denom, sp.Symbol("x")):
        if r.is_real or r.is_extended_real:
            roots.append(r)
        elif r.is_complex and r.has(sp.I):
            pass  # complex poles don't matter for real-input sampling
    return roots


def _near_pole(value: float, poles: list[sp.Expr], margin: float) -> bool:
    for p in poles:
        try:
            pv = float(p.evalf())
        except Exception:
            continue
        if abs(value - pv) < margin:
            return True
    return False


def _draw(rng: random.Random, low: float, high: float, log_scale: bool) -> float:
    if log_scale:
        lo, hi = 10.0**low, 10.0**high
        return rng.uniform(lo, hi) * rng.choice([-1.0, 1.0])
    return rng.uniform(low, high)


def sample_inputs(
    expr: sp.Expr,
    variables: list[sp.Symbol],
    n: int,
    seed: int,
    ints_only: bool = False,
    low: float = -5.0,
    high: float = 5.0,
    log_scale: bool = False,
    pole_margin: float = 0.2,
    allow_complex: bool = False,
    eval_gate: bool = True,
) -> list[dict[str, Any]]:
    """Sample `n` input dicts where `expr` is defined and finite.

    Inputs are the ground-truth variable names (competition-style). Values may
    be int/float; complex values are not sampled here (complex slice later).
    Points where the ground truth evaluates to a complex number are rejected
    unless `allow_complex=True` (keeps real-output rows honest). Families
    whose truth is not an evaluable AST (truth_code families like gcd/lcm)
    declare `eval_gate=False` and guarantee definedness themselves.
    """
    rng = random.Random(int_seed(f"sampler:{seed}"))
    poles = denominator_roots(expr)
    out: list[dict[str, Any]] = []
    attempts = 0
    while len(out) < n and attempts < MAX_ATTEMPTS:
        attempts += 1
        point: dict[str, Any] = {}
        ok = True
        for var in variables:
            v: int | float
            if ints_only or (var.is_integer if var.is_symbol else False):
                v = int(rng.randint(int(low), int(high)))
            else:
                v = _draw(rng, low, high, log_scale)
                if abs(v) < 1e-9:
                    v = 1e-6  # avoid degenerate 0 for division/log/sqrt
            if _near_pole(v, poles, pole_margin):
                ok = False
                break
            point[str(var)] = v
        if not ok:
            continue
        if eval_gate:
            # finiteness + realness gate: ground truth must evaluate to a finite
            # (and, unless allow_complex, real) number at the point
            subs = {var: sp.Float(point[str(var)]) for var in variables}
            try:
                val = sp.N(expr.subs(subs))
            except Exception:
                continue
            if not val.is_finite:
                continue
            if val.has(sp.I) and not allow_complex:
                continue
            if val.has(sp.I) is False and abs(complex(val)) > 1e12:
                continue  # numeric-stability guard for real outputs
        out.append(point)
    return out
