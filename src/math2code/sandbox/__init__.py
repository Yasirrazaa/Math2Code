"""Sandbox backends: local subprocess (default), nsjail, Docker, E2B.

Default local backend is subprocess + rlimits + AST allowlist (fast, cheap,
no root). Stronger backends (nsjail, Docker, E2B) are added behind the same
small interface — see base.execute_code.
"""

from math2code.sandbox.base import (
    ALLOWED_IMPORT_ROOTS,
    DENIED_BUILTINS,
    ExecutionResult,
    SafetyError,
    analyze_code,
    execute_code,
    run_solution_on_cases,
)
from math2code.sandbox.pool import SandboxPool

__all__ = [
    "ALLOWED_IMPORT_ROOTS",
    "DENIED_BUILTINS",
    "ExecutionResult",
    "SafetyError",
    "SandboxPool",
    "analyze_code",
    "execute_code",
    "run_solution_on_cases",
]
