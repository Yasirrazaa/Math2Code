"""Elementary-vocabulary composition family (vocabulary-expansion slice).

Extends the operator vocabulary beyond the competition train's seven functions
(log, tan, cos, sin, sec, csc, cot) AND beyond the single-operator
`FunctionVocabFamily`: every row is a NESTED COMPOSITION (depth 1-3) mixing
hyperbolics (incl. sech/csch/coth/asinh/acosh/atanh), inverse trig, abs,
factorial/binomial, and odd/even roots.

Domain discipline (two layers):
1. Each base atom carries an explicit sampling domain derived from its branch
   constraints (asin/acos |x|<1, acosh arg>=1.05, atanh |x|<1, log/sqrt arg>0,
   even roots x>=0, factorial/binomial ints, csch/coth away from the pole).
2. A probe gate (`_probe_safe`) numerically verifies the final composition is
   finite, real, and magnitude-bounded at a deterministic probe grid over the
   domain BEFORE `_build_pair` — so a depth-2/3 composition can never blow up
   the sampler or the sandbox. Outer wrappers are TOTAL functions (sin, cos,
   tanh, sinh, cosh, atan, abs, sech, asinh), so the base domain is the
   composition's domain.

   Note on odd roots: sympy evalfs a NEGATIVE float to a rational power as the
   complex principal root (and float() then fails on the residual imag noise),
   so odd-root bases are restricted to positive domains.
"""

from __future__ import annotations

import random
import signal
from collections.abc import Callable
from typing import Any

import sympy as sp

from math2code.data.synthesizer.core import SynthFamily, int_seed
from math2code.schemas import MathCodePair

_X = sp.Symbol("x")
_Y = sp.Symbol("y")

# Per-iteration wall-clock cap: a few nested compositions hit pathological
# mpmath argument reduction (sin of huge args -> Chudnovsky pi digits) and can
# take minutes; skip instead of hanging the generator.
_ITER_TIMEOUT_S = 3
_MAX_SSTR_LEN = 300
_MAX_OPS = 40


class _IterTimeoutError(Exception):
    """Raised by the SIGALRM handler when one composition exceeds the cap."""


def _on_alarm(signum: int, frame: Any) -> None:
    del signum, frame
    raise _IterTimeoutError()

# ---------------------------------------------------------------------------
# coefficients
# ---------------------------------------------------------------------------


def _c(rng: random.Random) -> tuple[int, int, int]:
    """Small integer coefficients for base atoms (avoid identity forms)."""
    return int(rng.randint(2, 7)), int(rng.randint(1, 6)), int(rng.randint(2, 9))


def _cw(rng: random.Random) -> tuple[int, int, int]:
    """Smaller coefficients for wrappers (keep nested ranges tame)."""
    return int(rng.randint(1, 3)), int(rng.randint(1, 2)), int(rng.randint(0, 5))


# ---------------------------------------------------------------------------
# probe gate: the composition must be finite + real + bounded on its domain
# ---------------------------------------------------------------------------


def _probe_safe(
    expr: sp.Expr, variables: list[sp.Symbol], sopt: dict[str, Any]
) -> bool:
    """Deterministic numeric probe over the sampling domain.

    Rejects compositions that are non-finite, complex, or magnitude-explosive
    (|val| > 1e10) anywhere on a coarse grid + random points — the family's own
    gate, independent of the sampler's rejection sampling.
    """
    low = float(sopt.get("low", -5.0))
    high = float(sopt.get("high", 5.0))
    ints = bool(sopt.get("ints_only", False))
    vals: list[float]
    if ints:
        lo_i, hi_i = int(low), int(high)
        vals = [float(lo_i + k) for k in range(5) if lo_i + k <= hi_i] or [
            float(lo_i)
        ]
    else:
        vals = [low + (high - low) * k / 4.0 for k in range(5)]
    rng = random.Random(int_seed("probe"))
    for _ in range(3):
        vals.append(float(rng.randint(lo_i, hi_i)) if ints else rng.uniform(low, high))
    for v0 in vals:
        for v1 in vals:
            point: dict[str, sp.Float] = {str(variables[0]): sp.Float(v0)}
            if len(variables) > 1:
                point[str(variables[1])] = sp.Float(v1)
            try:
                val = sp.N(expr.subs(point))
            except Exception:
                return False
            if not val.is_finite or val.has(sp.I) or abs(complex(val)) > 1e10:
                return False
    return True


# ---------------------------------------------------------------------------
# base atoms: (name, builder(a,b,c) -> expr, sopt(a,b,c) -> dict, n_vars)
# ---------------------------------------------------------------------------

def _sopt(*, low: float = -5.0, high: float = 5.0, **kw: Any) -> dict[str, Any]:
    return {"low": low, "high": high, **kw}


def _sopt_log(c: int, b: int, margin: float = 2.0) -> dict[str, Any]:
    """arg = b*x + c > margin  ->  x > (margin - c)/b.

    margin=2.0 is jitter-robust: the oracle jitters committed inputs by
    +/-15% relative, so arg stays > 0.85*2 = 1.7 even for the worst draw.
    """
    return _sopt(low=(margin - c) / b, high=(margin - c) / b + 6.0)


def _sopt_branch(b: int) -> dict[str, Any]:
    """|x| <= 0.7/b for asin/acos/atanh principal real branches.

    0.7 (not 0.9) survives the oracle's +/-15% jitter: worst case 0.7*1.15/b
    = 0.805/b < 1/b, keeping the principal branch real.
    """
    return _sopt(low=-0.7 / b, high=0.7 / b)


def _sopt_acosh(c: int, b: int) -> dict[str, Any]:
    """arg = b*x + c >= 2.5  ->  x >= (2.5 - c)/b.

    2.5 survives +/-15% jitter: worst-case arg = 0.85*2.5 = 2.125 > 1.
    """
    return _sopt(low=(2.5 - c) / b, high=(2.5 - c) / b + 6.0)


BaseSpec = tuple[
    Callable[..., sp.Expr],
    Callable[..., dict[str, Any]],
    int,
]

_BASES: dict[str, BaseSpec] = {
    "poly2": (
        lambda a, b, c: a * _X**2 + b * _X + c,
        lambda a, b, c: _sopt(low=-3.0, high=3.0),
        1,
    ),
    "poly3": (
        lambda a, b, c: a * _X**3 + b * _X + c,
        lambda a, b, c: _sopt(low=-3.0, high=3.0),
        1,
    ),
    "sin": (lambda a, b, c: a * sp.sin(b * _X) + c, lambda a, b, c: {}, 1),
    "cos": (lambda a, b, c: a * sp.cos(b * _X) + c, lambda a, b, c: {}, 1),
    "exp": (
        lambda a, b, c: a * sp.exp(b * _X) + c,
        lambda a, b, c: _sopt(low=-3.0, high=3.0),
        1,
    ),
    "log": (
        lambda a, b, c: a * sp.log(b * _X + c) + 1,
        lambda a, b, c: _sopt_log(c, b),
        1,
    ),
    "asin": (
        lambda a, b, c: a * sp.asin(b * _X) + c,
        lambda a, b, c: _sopt_branch(b),
        1,
    ),
    "acos": (
        lambda a, b, c: a * sp.acos(b * _X) + c,
        lambda a, b, c: _sopt_branch(b),
        1,
    ),
    "atan": (lambda a, b, c: a * sp.atan(b * _X) + c, lambda a, b, c: {}, 1),
    "sinh": (lambda a, b, c: a * sp.sinh(b * _X) + c, lambda a, b, c: {}, 1),
    "cosh": (lambda a, b, c: a * sp.cosh(b * _X) + c, lambda a, b, c: {}, 1),
    "tanh": (lambda a, b, c: a * sp.tanh(b * _X) + c, lambda a, b, c: {}, 1),
    "sech": (lambda a, b, c: a * sp.sech(b * _X) + c, lambda a, b, c: {}, 1),
    "csch": (
        lambda a, b, c: a * sp.csch(b * _X) + c,
        lambda a, b, c: _sopt(low=0.5, high=5.0),  # pole at 0 excluded
        1,
    ),
    "coth": (
        lambda a, b, c: a * sp.coth(b * _X) + c,
        lambda a, b, c: _sopt(low=0.5, high=5.0),  # pole at 0 excluded
        1,
    ),
    "asinh": (lambda a, b, c: a * sp.asinh(b * _X) + c, lambda a, b, c: {}, 1),
    "acosh": (
        lambda a, b, c: a * sp.acosh(b * _X + c) + 1,
        lambda a, b, c: _sopt_acosh(c, b),
        1,
    ),
    "atanh": (
        lambda a, b, c: a * sp.atanh(b * _X) + c,
        lambda a, b, c: _sopt_branch(b),
        1,
    ),
    "abs": (lambda a, b, c: a * sp.Abs(b * _X) + c, lambda a, b, c: {}, 1),
    "sqrt": (
        lambda a, b, c: a * sp.sqrt(b * _X + c) + 1,
        lambda a, b, c: _sopt_log(c, b),
        1,
    ),
    "root3": (
        lambda a, b, c: a * sp.root(_X, 3) + c,
        # sympy evalfs a negative Float to a rational power as the COMPLEX
        # principal root (x**(1/3) of -2.96 -> 0.72+1.24j), and float() then
        # fails on the 1e-30 residual imag noise; honest family = positive only
        lambda a, b, c: _sopt(low=0.05, high=8.0),
        1,
    ),
    "root4": (
        lambda a, b, c: a * sp.root(_X, 4) + c,
        lambda a, b, c: _sopt(low=0.05, high=8.0),  # even root: x >= 0
        1,
    ),
    "root5": (
        lambda a, b, c: a * sp.root(_X, 5) + c,
        lambda a, b, c: _sopt(low=0.05, high=8.0),  # see root3: positive only
        1,
    ),
    "factorial": (
        lambda a, b, c: sp.factorial(_X) + a * _X + b,
        # int jitter is +/-3: low=3 keeps jittered x >= 0 (factorial defined)
        lambda a, b, c: _sopt(low=3, high=10, ints_only=True),
        1,
    ),
    "binomial": (
        lambda a, b, c: a * sp.binomial(_X, 3) + b,
        lambda a, b, c: _sopt(low=3, high=12, ints_only=True),
        1,
    ),
    "atan2": (
        lambda a, b, c: sp.atan2(a * _Y, b * _X + c),
        lambda a, b, c: {},
        2,
    ),
}

# total outer wrappers: (name, builder(inner, a, b, c) -> expr)
_WRAPPERS: dict[str, Callable[[sp.Expr, int, int, int], sp.Expr]] = {
    "sin": lambda inner, a, b, c: a * sp.sin(b * inner) + c,
    "cos": lambda inner, a, b, c: a * sp.cos(b * inner) + c,
    "tanh": lambda inner, a, b, c: a * sp.tanh(b * inner) + c,
    "sinh": lambda inner, a, b, c: a * sp.sinh(b * inner) + c,
    "cosh": lambda inner, a, b, c: a * sp.cosh(b * inner) + c,
    "atan": lambda inner, a, b, c: a * sp.atan(b * inner) + c,
    "abs": lambda inner, a, b, c: a * sp.Abs(b * inner) + c,
    "sech": lambda inner, a, b, c: a * sp.sech(b * inner) + c,
    "asinh": lambda inner, a, b, c: a * sp.asinh(b * inner) + c,
}

# the vocabulary this family adds beyond the train's {log,tan,cos,sin,sec,csc,cot}
NEW_VOCAB = {
    "asin",
    "acos",
    "atan",
    "atan2",
    "sinh",
    "cosh",
    "tanh",
    "sech",
    "csch",
    "coth",
    "asinh",
    "acosh",
    "atanh",
    "abs",
    "factorial",
    "binomial",
    "root3",
    "root4",
    "root5",
}


class ElementaryExtFamily(SynthFamily):
    """Nested-composition evaluation family (vocabulary-expansion slice)."""

    domain = "Mathematics_General"
    equation_type = "functions"

    def _gate(self, result_expr: sp.Expr) -> str:
        return (
            "ground-truth AST is the truth; input domain enforced + probe gate "
            "(finite, real, |val|<=1e10 over the sampling domain)"
        )

    def generate(
        self, seed: int, prefix: str, count: int, **opts: Any
    ) -> list[MathCodePair]:
        rng = random.Random(int_seed(f"elementary_ext:{seed}"))
        n_variants = int(opts.get("n_variants", 2))
        max_depth = int(opts.get("max_depth", 3))
        out: list[MathCodePair] = []
        i = 0
        while len(out) < count * 2 and i < count * 60:
            i += 1
            base_name = rng.choice(list(_BASES))
            builder, sopt_fn, n_vars = _BASES[base_name]
            a, b, c = _c(rng)
            expr = builder(a, b, c)
            sopt = sopt_fn(a, b, c)
            vocab = [base_name]
            variables = [_X, _Y] if n_vars == 2 else [_X]
            depth = 1
            n_wraps = int(rng.choices([0, 1, 2], weights=[30, 45, 25])[0])
            for _ in range(min(n_wraps, max_depth - 1)):
                wname = rng.choice(list(_WRAPPERS))
                wa, wb, wc = _cw(rng)
                expr = _WRAPPERS[wname](expr, wa, wb, wc)
                vocab.append(wname)
                depth += 1
            # family gate: the composition must be finite+real+bounded on-domain
            # (bounded work: size caps + wall-clock timeout, skip on any hang)
            if len(sp.sstr(expr)) > _MAX_SSTR_LEN or expr.count_ops() > _MAX_OPS:
                continue
            signal.signal(signal.SIGALRM, _on_alarm)
            signal.alarm(_ITER_TIMEOUT_S)
            try:
                if not _probe_safe(expr, variables, sopt):
                    signal.alarm(0)
                    continue
                rows = self._build_pair(
                    f"{prefix}_{base_name}_{i}",
                    expr,
                    expr,  # evaluation family: problem AST == ground truth AST
                    variables,
                    int_seed(f"elementary_ext:{seed}:{i}"),
                    n_variants=n_variants,
                    meta={
                        "slice": "vocab",
                        "vocab": sorted(set(vocab)),
                        "n_vars": len(variables),
                        "depth": depth,
                        "coefficient_kind": "integer",
                    },
                    sample_kwargs=sopt,
                )
                signal.alarm(0)
            except _IterTimeoutError:
                signal.alarm(0)
                sp.core.cache.clear_cache()  # discard half-built sympy cache state
                continue
            out.extend(rows)
        return out[: count * 2]
