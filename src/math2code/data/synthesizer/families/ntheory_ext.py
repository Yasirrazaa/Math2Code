"""Extended number-theory families (docs/SYNTHETIC_EXPANSION.md §2 #14).

σ(n), τ(n), φ(n), μ(n), π(n) prime-count, modular inverse, modular power, and
partition count. Like gcd/lcm, these are NOT sympify-expressible as
parameterized ground truth (`sp.divisor_sigma(n)` with a symbolic `n` does not
evaluate), so they use the custom-code contract: `truth_code` executed in the
sandbox is the ground truth, and the oracle compares candidate vs truth with
exact integer equality (solution == truth for these families).

Definedness under oracle jitter (measured): the oracle's fresh inputs are
jittered by ±3 even for ints (`data/competition._jitter`), so:

- `sp.divisor_sigma/totient/mobius` RAISE for n <= 0 (measured) — the truth
  code clamps `n = max(1, int(n))` (self-consistent: solution == truth;
  committed cases have n >= 2, clamp is a no-op there).
- `sp.mod_inverse` RAISES for non-coprime (a, m) — the truth code returns a 0
  sentinel for gcd != 1, and `generate` FILTERS OUT any row whose committed
  cases are not all coprime, so committed outputs are always genuine inverses.
- `pow(a, b, m)` with m == 0 raises — clamp `m = max(2, m)`, `b = max(1, b)`.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import Any, cast

import sympy as sp

from math2code.data.synthesizer.core import SynthFamily, int_seed
from math2code.schemas import MathCodePair

_N = sp.Symbol("n")
_A = sp.Symbol("a")
_B = sp.Symbol("b")
_M = sp.Symbol("m")


def _sigma_code() -> str:
    return (
        "import sympy as sp\n"
        "def calculate(n):\n"
        "    n = max(1, int(n))\n"
        "    return sp.divisor_sigma(n)\n"
    )


def _tau_code() -> str:
    return (
        "import sympy as sp\n"
        "def calculate(n):\n"
        "    n = max(1, int(n))\n"
        "    return sp.divisor_count(n)\n"
    )


def _phi_code() -> str:
    return (
        "import sympy as sp\n"
        "def calculate(n):\n"
        "    n = max(1, int(n))\n"
        "    return sp.totient(n)\n"
    )


def _mu_code() -> str:
    return (
        "import sympy as sp\n"
        "def calculate(n):\n"
        "    n = max(1, int(n))\n"
        "    return sp.mobius(n)\n"
    )


def _primepi_code() -> str:
    return (
        "import sympy as sp\n"
        "def calculate(n):\n"
        "    n = max(1, int(n))\n"
        "    return sp.primepi(n)\n"
    )


def _modinv_code() -> str:
    return (
        "import math\n"
        "import sympy as sp\n"
        "def calculate(a, m):\n"
        "    a = int(a); m = int(m)\n"
        "    if math.gcd(a, m) != 1:\n"
        "        return 0  # undefined for non-coprime; such rows are filtered\n"
        "    return sp.mod_inverse(a, m)\n"
    )


def _powermod_code() -> str:
    return (
        "def calculate(a, b, m):\n"
        "    b = max(1, int(b)); m = max(2, int(m))\n"
        "    return pow(int(a), b, m)\n"
    )


def _partition_code() -> str:
    return (
        "import sympy as sp\n"
        "def calculate(n):\n"
        "    n = max(1, int(n))\n"
        "    return sp.partition(n)\n"
    )


def _modinv_valid(row: MathCodePair) -> bool:
    """Only keep rows whose committed inputs are all coprime (a, m)."""
    for tc in row.test_cases:
        a = int(cast(int, tc.input["a"]))
        m = int(cast(int, tc.input["m"]))
        if math.gcd(a, m) != 1:
            return False
    return True


# (vocab, input symbols, code builder, sampler kwargs, row-validity filter)
_TEMPLATES: list[
    tuple[
        str,
        list[sp.Symbol],
        Callable[[], str],
        dict[str, Any],
        Callable[[MathCodePair], bool] | None,
    ]
] = [
    ("sigma", [_N], _sigma_code, {"ints_only": True, "low": 2, "high": 200}, None),
    ("tau", [_N], _tau_code, {"ints_only": True, "low": 2, "high": 200}, None),
    ("varphi", [_N], _phi_code, {"ints_only": True, "low": 2, "high": 200}, None),
    ("mu", [_N], _mu_code, {"ints_only": True, "low": 2, "high": 200}, None),
    ("pi", [_N], _primepi_code, {"ints_only": True, "low": 2, "high": 200}, None),
    ("p", [_N], _partition_code, {"ints_only": True, "low": 2, "high": 40}, None),
    (
        "modinv",
        [_A, _M],
        _modinv_code,
        {"ints_only": True, "low": 2, "high": 60},
        _modinv_valid,
    ),
    (
        "powermod",
        [_A, _B, _M],
        _powermod_code,
        {"ints_only": True, "low": 2, "high": 60},
        None,
    ),
]


class NumberTheoryExtFamily(SynthFamily):
    """σ/τ/φ/μ/π/modinv/powermod/p(n) — exact integer equality via truth_code."""

    domain = "Mathematics_General"

    def _gate(self, result_expr: sp.Expr) -> str:
        return "truth_code executed in sandbox == candidate (exact integer equality)"

    def generate(
        self, seed: int, prefix: str, count: int, **opts: Any
    ) -> list[MathCodePair]:
        rng = random.Random(int_seed(f"ntheory_ext:{seed}"))
        out: list[MathCodePair] = []
        pool = opts.get("pool")  # shared SandboxPool -> fast truth_code evals
        i = 0
        while len(out) < count and i < count * 80:
            i += 1
            name, symbols, code_fn, sopt, valid = _TEMPLATES[
                int(rng.randrange(len(_TEMPLATES)))
            ]
            code = code_fn()
            problem = sp.Function(name)(*symbols)  # LaTeX only; truth is code
            rows = self._build_pair(
                f"{prefix}_{name}_{i}",
                problem,
                problem,  # unused for truth; kept for signature/sample domain
                symbols,
                int_seed(f"ntheory_ext:{seed}:{i}"),
                n_variants=1,
                meta={"vocab": name, "slice": "numtheory"},
                sample_kwargs=sopt,
                equation_type="number_theory",
                custom_code=(code, code),  # solution == truth for these families
                pool=pool,
            )
            for row in rows:
                if valid is None or valid(row):
                    out.append(row)
        return out[:count]
