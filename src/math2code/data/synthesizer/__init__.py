"""Code-first synthetic data synthesis (strategy: docs/DATA_STRATEGY.md §8).

Programmatic, seed-deterministic families build ground truth as SymPy ASTs,
render notation variants, and emit rows that satisfy the same contract as the
competition data. Rows are verified by the oracle (sandbox execution on fresh
inputs) before entering the training pool; the frozen split is never touched.
"""

from math2code.data.synthesizer.core import SynthFamily
from math2code.data.synthesizer.families.derivative import DerivativeFamily
from math2code.data.synthesizer.families.functions import FunctionVocabFamily
from math2code.data.synthesizer.families.integral import IntegralFamily
from math2code.data.synthesizer.families.multivariate import MultivariateFamily
from math2code.data.synthesizer.families.ode import ODEFamily
from math2code.data.synthesizer.printer import VariantPrinter, render_variants
from math2code.data.synthesizer.sampler import sample_inputs

__all__ = [
    "SynthFamily",
    "DerivativeFamily",
    "IntegralFamily",
    "FunctionVocabFamily",
    "MultivariateFamily",
    "ODEFamily",
    "VariantPrinter",
    "render_variants",
    "sample_inputs",
]

FAMILIES = {
    "derivative": DerivativeFamily,
    "integration": IntegralFamily,
    "functions": FunctionVocabFamily,
    "multivariate": MultivariateFamily,
    "ode": ODEFamily,
}
