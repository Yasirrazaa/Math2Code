"""Set-cardinality family (docs/SYNTHETIC_EXPANSION.md §Tier 3, gated).

Constant-output rows whose ground truth is an exact non-negative integer:
interval intersection/union measures (``|[a,b] ∩ [c,d]|``, ``|[a,b] ∪ [c,d]|``
for real intervals with integer endpoints), finite-set cardinalities
(``|A ∪ B|``, ``|A ∩ B|`` for subsets of {1..10}), and Iverson-bracket
membership indicators (``[x ∈ [a,b]]`` = 1 iff a ≤ x ≤ b).

SymPy cannot render any of these query surfaces from a single AST (set braces,
interval notation, `\\in`), so every row uses `latex_override` (pattern E)
with pattern-B constant-output exact `sp.Integer` truths.

Gated portfolio family: `meta["gated"] = True` — the mixture builder excludes
these rows from the default RL mixture unless `--include-gated-slices` is
passed.
"""

from __future__ import annotations

import random
from typing import Any

import sympy as sp

from math2code.data.synthesizer.core import SynthFamily, int_seed
from math2code.schemas import MathCodePair

_ENDPOINT_HIGH = 12  # interval endpoints and membership x live in 0..12
_SET_UNIVERSE = tuple(range(1, 11))  # finite sets draw from {1..10}
_SET_SIZES = (2, 3, 4)  # 2..4 elements per set (|A ∪ B| >= 2 stays nonempty)

_KINDS = (
    "interval_intersect",
    "interval_union",
    "set_union",
    "set_intersect",
    "membership",
)


def _set_tex(a: tuple[int, ...], b: tuple[int, ...], op: str) -> str:
    r"""`\left|\{1,2,5\} \cup \{2,3,7\}\right|` with concrete sorted elements."""
    elems_a = ",".join(str(v) for v in a)
    elems_b = ",".join(str(v) for v in b)
    return r"\left|\{" + elems_a + r"\} " + op + r" \{" + elems_b + r"\}\right|"


class SetsCardinalityFamily(SynthFamily):
    """Interval/set cardinalities and membership indicators (exact integers)."""

    domain = "Mathematics_General"
    equation_type = "sets"

    def _gate(self, result_expr: sp.Expr) -> str:
        return (
            "cardinality recomputed from the set/interval endpoints "
            "(len/max/min measure or membership predicate) — exact "
            "non-negative integer"
        )

    def _make(
        self, kind: str, rng: random.Random, task_id: str, i: int
    ) -> MathCodePair | None:
        """One concrete object of `kind`; None when a structural gate fails."""
        meta: dict[str, Any] = {"slice": "sets", "kind": kind, "gated": True}
        if kind == "interval_intersect":
            a = rng.randint(0, _ENDPOINT_HIGH)
            b = rng.randint(a, _ENDPOINT_HIGH)
            c = rng.randint(0, _ENDPOINT_HIGH)
            d = rng.randint(c, _ENDPOINT_HIGH)
            # |[a,b] ∩ [c,d]| = max(0, min(b,d) - max(a,c)); 0 for disjoint
            value = max(0, min(b, d) - max(a, c))
            meta.update(a=a, b=b, c=c, d=d)
            tex = rf"\left|[{a},{b}] \cap [{c},{d}]\right|"
        elif kind == "interval_union":
            a = rng.randint(0, _ENDPOINT_HIGH)
            b = rng.randint(a, _ENDPOINT_HIGH)
            c = rng.randint(0, _ENDPOINT_HIGH)
            d = rng.randint(c, _ENDPOINT_HIGH)
            if not (c <= b and a <= d):
                return None  # must overlap/touch so the union is one interval
            value = max(b, d) - min(a, c)
            meta.update(a=a, b=b, c=c, d=d)
            tex = rf"\left|[{a},{b}] \cup [{c},{d}]\right|"
        elif kind in ("set_union", "set_intersect"):
            # NOTE: sp.FiniteSet has no `.cardinality` — use len(...) directly.
            elems_a = tuple(sorted(rng.sample(_SET_UNIVERSE, rng.randint(*_SET_SIZES[:2]))))
            elems_b = tuple(sorted(rng.sample(_SET_UNIVERSE, rng.randint(*_SET_SIZES[:2]))))
            sa, sb = set(elems_a), set(elems_b)
            value = len(sa | sb) if kind == "set_union" else len(sa & sb)
            meta.update(set_a=list(elems_a), set_b=list(elems_b))
            tex = _set_tex(elems_a, elems_b, r"\cup" if kind == "set_union" else r"\cap")
        else:  # membership: Iverson bracket [x ∈ [a,b]] ∈ {0, 1}
            x = rng.randint(0, _ENDPOINT_HIGH)
            a = rng.randint(0, _ENDPOINT_HIGH)
            b = rng.randint(a, _ENDPOINT_HIGH)
            value = 1 if a <= x <= b else 0
            meta.update(x=x, a=a, b=b)
            tex = rf"\left[{x} \in [{a},{b}]\right]"
        truth = sp.Integer(value)  # exact, finite, non-negative by construction
        rows = self._build_pair(
            task_id,
            truth,  # problem AST unused for rendering (latex_override wins)
            truth,
            [],  # constant-output rows: no input variables
            int_seed(f"sets_cardinality:{task_id}:{i}"),
            n_variants=1,  # variant list is provided via latex_override
            meta=meta,
            equation_type="sets",
            latex_override=[tex],
        )
        return rows[0] if rows else None

    def generate(
        self, seed: int, prefix: str, count: int, **opts: Any
    ) -> list[MathCodePair]:
        rng = random.Random(int_seed(f"sets_cardinality:{seed}"))
        out: list[MathCodePair] = []
        i = 0
        while len(out) < count and i < count * 60:  # cap attempts for gates
            i += 1
            # round-robin kind cycling: every >=5 objects covers all five kinds
            # (deterministic, keeps the pool balanced on small draws)
            kind = _KINDS[(i - 1) % len(_KINDS)]
            row = self._make(kind, rng, f"{prefix}_{kind}_{i}", i)
            if row is not None:
                out.append(row)
        return out
