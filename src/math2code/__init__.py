"""Math2Code: LaTeX -> executable SymPy code (TIR + RLVR)."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("math2code")
except PackageNotFoundError:  # source-tree runs without installed metadata
    __version__ = "0.3.0"
