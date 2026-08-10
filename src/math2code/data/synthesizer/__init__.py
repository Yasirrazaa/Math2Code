"""Code-first synthetic data synthesis (strategy: docs/DATA_STRATEGY.md §8).

Programmatic, seed-deterministic families build ground truth as SymPy ASTs,
render notation variants, and emit rows that satisfy the same contract as the
competition data. Rows are verified by the oracle (sandbox execution on fresh
inputs) before entering the training pool; the frozen split is never touched.
"""

from math2code.data.synthesizer.core import SynthFamily
from math2code.data.synthesizer.families.derivative import DerivativeFamily
from math2code.data.synthesizer.families.edge import EdgeCaseFamily
from math2code.data.synthesizer.families.functions import FunctionVocabFamily
from math2code.data.synthesizer.families.geometry import GeometryFamily
from math2code.data.synthesizer.families.integral import IntegralFamily
from math2code.data.synthesizer.families.multivariate import MultivariateFamily
from math2code.data.synthesizer.families.numtheory import NumberTheoryFamily
from math2code.data.synthesizer.families.ode import ODEFamily
from math2code.data.synthesizer.families.sequences import SequenceFamily
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
    "integration": IntegralFamily,
    "functions": FunctionVocabFamily,
    "geometry": GeometryFamily,
    "multivariate": MultivariateFamily,
    "numtheory": NumberTheoryFamily,
    "ode": ODEFamily,
    "sequences": SequenceFamily,
}
