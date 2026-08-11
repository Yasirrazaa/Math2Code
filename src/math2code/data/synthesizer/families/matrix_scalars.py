r"""Matrix scalar-invariant family (docs/SYNTHETIC_EXPANSION.md §Tier 2).

Scalar queries over a CONCRETE matrix, exact Rational/Integer outputs
(pattern B — constant rows, no input variables):

- det(A)
- trace(A)
- (A^{-1})_{ij}  (1-indexed in the LaTeX; singular matrices rejected)
- characteristic polynomial evaluated at lambda: charpoly(A)(lambda)
- squared Frobenius norm ||A||_F^2 (exact sum of squared entries)

The question surface is a named-operator LaTeX with the matrix embedded as a
`pmatrix` (``\\det(...)``, ``\\operatorname{{tr}}(...)``, ...) — notation SymPy cannot
render from a single AST, so this family uses the `latex_override` hook
(consumer #2 after series_coeff).

Verification epistemology: every truth is recomputed directly from the
concrete matrix (`.det()`, `.trace()`, `.inv()[i,j]`, `.charpoly(x).eval(lam)`,
sum of squares) — exact arithmetic, never float-drifted. Rows whose result is
trivial (det == 0) or blows up (|result| > 1e6) are skipped.
"""

from __future__ import annotations

import random
from typing import Any

import sympy as sp

from math2code.data.synthesizer.core import SynthFamily, int_seed
from math2code.schemas import MathCodePair

_SIZES = (2, 2, 3, 3, 4)  # matrix sizes (2x2/3x3/4x4; 2/3 weighted)
_LAMBDAS = (0, 1, 2)  # charpoly evaluation points
_MAX_ABS = 1_000_000  # magnitude guard: skip results that would blow up
_QUERIES = ("det", "trace", "inverse", "charpoly", "frobenius")


def _num_tex(v: sp.Rational) -> str:
    """Exact LaTeX for an integer or rational matrix entry."""
    if v.q == 1:
        return str(v.p)
    num = f"\\frac{{{abs(v.p)}}}{{{v.q}}}"
    return f"-{num}" if v.p < 0 else num


def _pmatrix(mat: sp.Matrix) -> str:
    """Concrete matrix as a `\\begin{pmatrix}...\\end{pmatrix}` string."""
    rows = [" & ".join(_num_tex(v) for v in mat.row(i)) for i in range(mat.rows)]
    return "\\begin{pmatrix}" + " \\\\ ".join(rows) + "\\end{pmatrix}"


def _random_matrix(rng: random.Random, size: int, rational: bool) -> sp.Matrix:
    """Integer (-6..6) or exact-Rational (denominator 2/3/4) entries."""
    rows: list[list[sp.Rational]] = []
    for _ in range(size):
        row: list[sp.Rational] = []
        for _ in range(size):
            if rational and rng.random() < 0.5:
                num = int(rng.randint(-4, 4))
                den = int(rng.choice((2, 3, 4)))
                row.append(sp.Rational(num, den))
            else:
                row.append(sp.Integer(int(rng.randint(-6, 6))))
        rows.append(row)
    return sp.Matrix(rows)


def _det_tex(mat: sp.Matrix) -> str:
    return rf"\det\left({_pmatrix(mat)}\right)"


def _trace_tex(mat: sp.Matrix) -> str:
    return rf"\operatorname{{tr}}\left({_pmatrix(mat)}\right)"


def _inv_tex(mat: sp.Matrix, i: int, j: int) -> str:
    return rf"\left({_pmatrix(mat)}\right)^{{-1}}_{{{i + 1},{j + 1}}}"


def _charpoly_tex(mat: sp.Matrix, lam: int) -> str:
    return rf"\operatorname{{charpoly}}\left({_pmatrix(mat)}\right)\left({lam}\right)"


def _frob_tex(mat: sp.Matrix) -> str:
    return rf"\left\Vert {_pmatrix(mat)} \right\Vert_{{F}}^{{2}}"


class MatrixScalarsFamily(SynthFamily):
    """det/trace/inverse-element/charpoly/norm of a concrete matrix (exact)."""

    domain = "Mathematics_General"
    equation_type = "matrix"

    def _gate(self, result_expr: sp.Expr) -> str:
        return (
            "exact Rational/Integer recomputed from the concrete matrix "
            "(det != 0 for inverse queries; |result| <= 1e6)"
        )

    def generate(
        self, seed: int, prefix: str, count: int, **opts: Any
    ) -> list[MathCodePair]:
        rng = random.Random(int_seed(f"matrix_scalars:{seed}"))
        out: list[MathCodePair] = []
        i = 0
        while len(out) < count and i < count * 60:  # cap attempts for gates
            i += 1
            size = int(rng.choice(_SIZES))
            rational = rng.random() < 0.4
            query = _QUERIES[int(rng.randrange(len(_QUERIES)))]
            # inverse of a 4x4 rational matrix can blow up: ints only there
            if query == "inverse" and size == 4:
                rational = False
            mat = _random_matrix(rng, size, rational)
            if mat.is_zero_matrix:
                continue

            if query == "det":
                result: sp.Expr = mat.det()
                if result == 0 or abs(result) > _MAX_ABS:  # trivial / blowup
                    continue
                tex = _det_tex(mat)
            elif query == "trace":
                result = mat.trace()
                if abs(result) > _MAX_ABS:
                    continue
                tex = _trace_tex(mat)
            elif query == "inverse":
                det = mat.det()
                if det == 0 or abs(det) > _MAX_ABS:
                    continue
                ii = int(rng.randrange(size))
                jj = int(rng.randrange(size))
                result = mat.inv()[ii, jj]
                if abs(result) > _MAX_ABS:
                    continue
                tex = _inv_tex(mat, ii, jj)
            elif query == "charpoly":
                lam = int(rng.choice(_LAMBDAS))
                result = sp.Poly(mat.charpoly(sp.Symbol("x")), sp.Symbol("x")).eval(lam)
                if abs(result) > _MAX_ABS:
                    continue
                tex = _charpoly_tex(mat, lam)
            else:  # frobenius norm squared: exact sum of squared entries
                result = sum(v**2 for v in mat)
                if abs(result) > _MAX_ABS:
                    continue
                tex = _frob_tex(mat)

            rows = self._build_pair(
                f"{prefix}_{query}_{i}",
                mat,  # unused for rendering; latex_override wins
                result,
                [],  # constant-output rows (no input variables)
                int_seed(f"matrix_scalars:{seed}:{i}"),
                n_variants=1,  # variant list is provided via latex_override
                meta={
                    "slice": "matrix",
                    "size": size,
                    "query": query,
                    "coefficient_kind": "rational" if rational else "integer",
                },
                equation_type="matrix",
                latex_override=[tex],
            )
            out.extend(rows)
        return out[:count]
