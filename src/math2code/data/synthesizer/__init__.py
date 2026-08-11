"""Code-first synthetic data synthesis (strategy: docs/DATA_STRATEGY.md §8).

Programmatic, seed-deterministic families build ground truth as SymPy ASTs,
render notation variants, and emit rows that satisfy the same contract as the
competition data. Rows are verified by the oracle (sandbox execution on fresh
inputs) before entering the training pool; the frozen split is never touched.
"""

from math2code.data.synthesizer.core import SynthFamily
from math2code.data.synthesizer.families.combinatorics import CombinatoricsFamily
from math2code.data.synthesizer.families.complex_eval import ComplexEvalFamily
from math2code.data.synthesizer.families.derivative import DerivativeFamily
from math2code.data.synthesizer.families.differential_c1 import DifferentialC1Family
from math2code.data.synthesizer.families.edge import EdgeCaseFamily
from math2code.data.synthesizer.families.elementary_ext import ElementaryExtFamily
from math2code.data.synthesizer.families.functions import FunctionVocabFamily
from math2code.data.synthesizer.families.geometry import GeometryFamily
from math2code.data.synthesizer.families.geometry_ext import GeometryExtFamily
from math2code.data.synthesizer.families.integral import IntegralFamily
from math2code.data.synthesizer.families.limits import LimitsFamily
from math2code.data.synthesizer.families.matrix_scalars import MatrixScalarsFamily
from math2code.data.synthesizer.families.multivariate import MultivariateFamily
from math2code.data.synthesizer.families.ntheory_ext import NumberTheoryExtFamily
from math2code.data.synthesizer.families.numtheory import NumberTheoryFamily
from math2code.data.synthesizer.families.ode import ODEFamily
from math2code.data.synthesizer.families.polynomial_invariants import (
    PolynomialInvariantsFamily,
)
from math2code.data.synthesizer.families.sequences import SequenceFamily
from math2code.data.synthesizer.families.series_coeff import SeriesCoefficientFamily
from math2code.data.synthesizer.families.sets_cardinality import SetsCardinalityFamily
from math2code.data.synthesizer.families.solving_scalarized import (
    SolvingScalarizedFamily,
)
from math2code.data.synthesizer.families.special_functions import SpecialFunctionsFamily
from math2code.data.synthesizer.families.stats_moments import StatsMomentsFamily
from math2code.data.synthesizer.families.summation import SummationFamily
from math2code.data.synthesizer.printer import VariantPrinter, render_variants
from math2code.data.synthesizer.sampler import sample_inputs

__all__ = [
    "SynthFamily",
    "DerivativeFamily",
    "EdgeCaseFamily",
    "IntegralFamily",
    "FunctionVocabFamily",
    "GeometryFamily",
    "MultivariateFamily",
    "NumberTheoryFamily",
    "ODEFamily",
    "SequenceFamily",
    "VariantPrinter",
    "render_variants",
    "sample_inputs",
]

FAMILIES = {
    "derivative": DerivativeFamily,
    "edge": EdgeCaseFamily,
    "elementary_ext": ElementaryExtFamily,
    "integration": IntegralFamily,
    "functions": FunctionVocabFamily,
    "geometry": GeometryFamily,
    "geometry_ext": GeometryExtFamily,
    "multivariate": MultivariateFamily,
    "numtheory": NumberTheoryFamily,
    "ode": ODEFamily,
    "sequences": SequenceFamily,
    "differential_c1": DifferentialC1Family,
    "limits": LimitsFamily,
    "series_coeff": SeriesCoefficientFamily,
    "summation": SummationFamily,
    "combinatorics": CombinatoricsFamily,
    "complex_eval": ComplexEvalFamily,
    "matrix_scalars": MatrixScalarsFamily,
    "polynomial_invariants": PolynomialInvariantsFamily,
    "ntheory_ext": NumberTheoryExtFamily,
    "sets_cardinality": SetsCardinalityFamily,
    "solving_scalarized": SolvingScalarizedFamily,
    "special_functions": SpecialFunctionsFamily,
    "stats_moments": StatsMomentsFamily,
}
