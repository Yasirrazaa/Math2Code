"""Complex-output evaluation family (docs/SYNTHETIC_EXPANSION.md Tier 2).

Real inputs, complex ground truth -> the canonical `'re+imj'` string outputs
(the public-probe complex slice: 98 rows). Pattern A family: the ground truth
is a sympy AST that AUTO-EVALUATES to a complex Number when substituted with
Float inputs (`sp.sqrt(-5.3)` -> `2.3*I`, `sp.log(-5.3)` -> `log(5.3) + I*pi`,
`sp.asin(2.5)` -> `1.57 - 1.44*I`, `sp.acosh(-2)` -> `1.317 + I*pi`), so the
standard `solution_code` path (`complex(_expr.subs(...))`) works in the
sandbox without a truth_code contract.

Every sampling domain is chosen so the WHOLE domain produces a complex truth
(e.g. `sqrt(-(x**2 + b))` for |x| <= 3, `asin(b*x + c)` with `b*x + c > 1`):
rows whose committed outputs are not genuinely complex (|imag| <= 1e-6) are
rejected before emission, keeping the `output_type == "complex"` contract
honest. Safety rails: composition depth <= 2, sstr length <= 250, count_ops
<= 35, and a per-iteration SIGALRM guard against pathological sympy evalf
hangs (shared lesson from the elementary_ext family).
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

_MAX_STR_LEN = 250
_MAX_OPS = 35
_HANG_TIMEOUT_S = 5.0
_MIN_IMAG = 1e-6  # committed outputs must be genuinely complex


def _p(rng: random.Random) -> tuple[int, int, int]:
    """Small int params (a scale, b frequency/shift, c offset)."""
    a = int(rng.randint(2, 6))
    b = int(rng.randint(1, 3))
    c = int(rng.randint(2, 6))
    return a, b, c


def _mk(
    rng: random.Random,
    fn: Callable[[int, int, int], sp.Expr],
    sopt: Callable[[int, int, int], dict[str, Any]] | None = None,
) -> tuple[sp.Expr, dict[str, Any]]:
    """Draw params, build the expression, derive the sampling domain."""
    a, b, c = _p(rng)
    return fn(a, b, c), (sopt(a, b, c) if sopt else {})


_COMPLEX_OPTS = {"allow_complex": True}


# Each template: (name, builder(rng) -> (expr, sampler_kwargs)). Domains are
# constrained so the ground truth is complex for EVERY point in the domain.
_TEMPLATES: list[
    tuple[str, Callable[[random.Random], tuple[sp.Expr, dict[str, Any]]]]
] = [
    # sqrt of negative quadratic: -x^2 - b < 0 for all |x| <= 3
    (
        "sqrt_neg_quad",
        lambda r: _mk(
            r,
            lambda a, b, c: a * sp.sqrt(-(_X**2) - b) + c,
            lambda a, b, c: {"low": -3.0, "high": 3.0, **_COMPLEX_OPTS},
        ),
    ),
    # sqrt of negative linear: -(x + b) < 0 requires x > -b
    (
        "sqrt_neg_lin",
        lambda r: _mk(
            r,
            lambda a, b, c: a * sp.sqrt(-(_X + b)) + c,
            lambda a, b, c: {"low": -b + 0.5, "high": 8.0, **_COMPLEX_OPTS},
        ),
    ),
    # log of negative: log(-(x + b)) = log(x + b) + I*pi for x > -b
    (
        "log_neg_lin",
        lambda r: _mk(
            r,
            lambda a, b, c: a * sp.log(-(_X + b)) + c,
            lambda a, b, c: {"low": -b + 0.5, "high": 8.0, **_COMPLEX_OPTS},
        ),
    ),
    # e^{i k x} = cos(kx) + i sin(kx)
    (
        "exp_imag_lin",
        lambda r: _mk(
            r,
            lambda a, b, c: sp.exp(sp.I * b * _X) + c,
            lambda a, b, c: {"low": -4.0, "high": 4.0, **_COMPLEX_OPTS},
        ),
    ),
    # e^{i sin(bx)} — composition, still unit-circle complex
    (
        "exp_imag_sin",
        lambda r: _mk(
            r,
            lambda a, b, c: sp.exp(sp.I * sp.sin(b * _X)) + c,
            lambda a, b, c: {"low": -4.0, "high": 4.0, **_COMPLEX_OPTS},
        ),
    ),
    # asin of a shifted argument > 1: asin(b*x + c) complex for b*x + c > 1
    (
        "asin_shift",
        lambda r: _mk(
            r,
            lambda a, b, c: a * sp.asin(b * _X + c) + a,
            lambda a, b, c: {"low": 0.2, "high": 4.0, **_COMPLEX_OPTS},
        ),
    ),
    # acos of a shifted argument < -1 (symmetric complex branch)
    (
        "acos_shift",
        lambda r: _mk(
            r,
            lambda a, b, c: a * sp.acos(-(b * _X + c)) + a,
            lambda a, b, c: {"low": 0.2, "high": 4.0, **_COMPLEX_OPTS},
        ),
    ),
    # acosh of a negative argument <= -1: acosh(-y) = acosh(y) + I*pi
    (
        "acosh_neg",
        lambda r: _mk(
            r,
            lambda a, b, c: a * sp.acosh(-(b * _X + c)) + a,
            lambda a, b, c: {
                "low": max(0.3, (1 - c) / b + 0.3),
                "high": 5.0,
                **_COMPLEX_OPTS,
            },
        ),
    ),
    # tanh(i b x) = i tan(b x) — purely imaginary
    (
        "tanh_imag",
        lambda r: _mk(
            r,
            lambda a, b, c: a * sp.tanh(sp.I * b * _X) + c,
            lambda a, b, c: {"low": -4.0, "high": 4.0, **_COMPLEX_OPTS},
        ),
    ),
    # asin(1/(b x)) complex for |x| < 1/b (argument > 1), away from x = 0
    (
        "asin_recip",
        lambda r: _mk(
            r,
            lambda a, b, c: a * sp.asin(1 / (b * _X)) + c,
            lambda a, b, c: {"low": -0.9 / b, "high": 0.9 / b, **_COMPLEX_OPTS},
        ),
    ),
]

_VOCAB = [name for name, _ in _TEMPLATES]


class ComplexEvalFamily(SynthFamily):
    """Real-input evaluation whose ground truth is complex ('re+imj' outputs)."""

    domain = "Mathematics_General"
    equation_type = "complex"

    def _gate(self, result_expr: sp.Expr) -> str:
        return "ground truth complex at committed inputs ('re+imj' contract); |imag| > 1e-6"

    @staticmethod
    def _all_outputs_complex(rows: list[MathCodePair]) -> bool:
        """Every committed output parses and has a non-trivial imaginary part."""
        from math2code.evaluation.metrics import parse_number

        for row in rows:
            for tc in row.test_cases:
                try:
                    val = parse_number(tc.output)
                except (ValueError, TypeError):
                    return False
                if abs(val.imag) <= _MIN_IMAG:
                    return False
        return True

    @staticmethod
    def _inputs_real(rows: list[MathCodePair]) -> bool:
        for row in rows:
            for tc in row.test_cases:
                for v in tc.input.values():
                    if isinstance(v, complex) and abs(v.imag) > 1e-12:
                        return False
        return True

    def generate(
        self, seed: int, prefix: str, count: int, **opts: Any
    ) -> list[MathCodePair]:
        n_variants = int(opts.get("n_variants", 2))
        rng = random.Random(int_seed(f"complex_eval:{seed}"))
        out: list[MathCodePair] = []
        i = 0
        while len(out) < count * n_variants and i < count * 60:
            i += 1
            name, builder = _TEMPLATES[int(rng.randrange(len(_TEMPLATES)))]
            expr, sopt = builder(rng)
            # safety rails (shared hang lesson): bound expression size/complexity
            if len(sp.sstr(expr)) > _MAX_STR_LEN or expr.count_ops() > _MAX_OPS:
                continue
            # per-iteration hang guard around _build_pair (sympy evalf can hang
            # on pathological compositions)
            signal.setitimer(signal.ITIMER_REAL, _HANG_TIMEOUT_S)
            try:
                rows = self._build_pair(
                    f"{prefix}_{name}_{i}",
                    expr,  # problem and result are the same complex expression
                    expr,
                    [_X],
                    int_seed(f"complex_eval:{seed}:{i}"),
                    n_variants=n_variants,
                    meta={"slice": "vocab", "vocab": name, "n_vars": 1, "depth": 2},
                    sample_kwargs=sopt,
                    equation_type="complex",
                )
            except TimeoutError:
                continue  # pathological composition: drop the object, keep going
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            if not rows:
                continue
            # gate: committed outputs must be genuinely complex AND inputs real
            if not self._all_outputs_complex(rows) or not self._inputs_real(rows):
                continue
            out.extend(rows)
        return out[: count * n_variants]
