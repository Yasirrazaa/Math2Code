"""Number-theory families (docs/DATA_STRATEGY.md §3; optional 10% bucket).

gcd/lcm are NOT sympify-expressible as parameterized ground truth
(`sp.gcd(a, b)` with symbols evaluates to 1 at construction), so these
families use the custom-code contract: `truth_code` executed in the sandbox
is the ground truth, and the oracle compares candidate vs truth with exact
integer equality. This is the first consumer of the `MathCodePair.truth_code`
path (docs/DATA_STRATEGY.md §8 corrections: only families whose truth is not
an AST may use it).
"""

from __future__ import annotations

import random
from typing import Any

import sympy as sp

from math2code.data.synthesizer.core import SynthFamily, int_seed
from math2code.schemas import MathCodePair

_A = sp.Symbol("a")
_B = sp.Symbol("b")


def _gcd_code() -> str:
    return "import math\ndef calculate(a, b):\n    return math.gcd(int(a), int(b))\n"


def _lcm_code() -> str:
    return (
        "import math\n"
        "def calculate(a, b):\n"
        "    a = int(a); b = int(b)\n"
        "    return a * b // math.gcd(a, b)\n"
    )


class NumberTheoryFamily(SynthFamily):
    """gcd / lcm parameterized by integer pairs (exact integer equality)."""

    domain = "Mathematics_General"

    def _gate(self, result_expr: sp.Expr) -> str:
        return "truth_code executed in sandbox == candidate (exact integer equality)"

    def generate(
        self, seed: int, prefix: str, count: int, **opts: Any
    ) -> list[MathCodePair]:
        rng = random.Random(int_seed(f"numtheory:{seed}"))
        templates = [("gcd", _gcd_code), ("lcm", _lcm_code)]
        out: list[MathCodePair] = []
        pool = opts.get("pool")  # shared SandboxPool -> fast truth_code evals
        i = 0
        while len(out) < count * 2 and i < count * 60:
            i += 1
            name, code_fn = templates[int(rng.randrange(len(templates)))]
            code = code_fn()
            problem = sp.Function(name)(_A, _B)  # LaTeX only; truth is code
            rows = self._build_pair(
                f"{prefix}_{name}_{i}",
                problem,
                problem,  # unused for truth; kept for signature/sample domain
                [_A, _B],
                int_seed(f"numtheory:{seed}:{i}"),
                n_variants=1,
                meta={"vocab": name, "slice": "numtheory"},
                sample_kwargs={"ints_only": True, "low": 1, "high": 60},
                equation_type="number_theory",
                custom_code=(code, code),  # solution == truth for these families
                pool=pool,
            )
            out.extend(rows)
        return out[: count * 2]
