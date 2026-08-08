"""Sandbox interface + reference implementation.

Defense in depth for executing LLM-generated code:
  1. static AST analysis (import/builtin allowlist)  -> reject early
  2. subprocess isolation with rlimits (CPU time, memory) + timeout
  3. (optional, stronger) nsjail / Docker per-batch  -> see nsjail.py, docker.py

The interface is deliberately small so the pipeline can swap backends
(local subprocess at RL train time, E2B/Docker at final eval time).
"""

from __future__ import annotations

import ast
import base64
import json
import os
import resource
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Static analysis (layer 1)
# ---------------------------------------------------------------------------

ALLOWED_IMPORT_ROOTS = {
    "sympy",
    "numpy",
    "math",
    "cmath",
    "itertools",
    "functools",
    "collections",
    "fractions",
    "statistics",
    "decimal",
}
DENIED_BUILTINS = {
    "eval",
    "exec",
    "open",
    "compile",
    "__import__",
    "input",
    "breakpoint",
    "globals",
    "locals",
    "vars",
    "memoryview",
    "help",
    "exit",
    "quit",
}


class SafetyError(Exception):
    """Raised when code fails static analysis."""


def analyze_code(code: str) -> None:
    """Reject code that imports disallowed modules or uses dangerous builtins."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise SafetyError(f"SyntaxError: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                raise SafetyError("relative imports are not allowed")
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                raise SafetyError(f"import not allowed: {node.module}")
        elif isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    raise SafetyError(f"import not allowed: {a.name}")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in DENIED_BUILTINS:
                raise SafetyError(f"builtin not allowed: {func.id}")
            if isinstance(func, ast.Attribute):
                # deny attr access on suspicious bases is handled by import allowlist;
                # but deny dunder chains like os.system (os can't be imported anyway)
                if func.attr in {"system", "popen", "subprocess", "connect"}:
                    raise SafetyError(f"attribute not allowed: {func.attr}")
        elif isinstance(node, (ast.Attribute, ast.Name)):
            name = node.attr if isinstance(node, ast.Attribute) else node.id
            if name.startswith("__") and name not in {"__name__", "__file__"}:
                raise SafetyError(f"dunder access not allowed: {name}")


# ---------------------------------------------------------------------------
# Execution result + reference subprocess sandbox (layers 2-3)
# ---------------------------------------------------------------------------


@dataclass
class ExecutionResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    safety_error: str | None = None

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and self.safety_error is None


_RUNNER_TEMPLATE = r"""
import sys, json, math, cmath, inspect, base64

code = base64.b64decode(sys.argv[2]).decode()

def _find_function(ns):
    if "calculate" in ns and callable(ns["calculate"]):
        return ns["calculate"]
    for name, obj in ns.items():
        if name.startswith("_"):
            continue
        if callable(obj) and getattr(obj, "__module__", None) == "__main__":
            return obj
    return None

def _match_inputs(func, inputs):
    sig = inspect.signature(func)
    args = {}
    for pname, param in sig.parameters.items():
        val = inputs.get(pname)
        if val is None:
            base = pname[:-len("_val")] if pname.endswith("_val") else pname
            if base.endswith("_value"):
                base = base[:-len("_value")]
            val = inputs.get(base)
        if val is None and param.default is not inspect.Parameter.empty:
            val = param.default
        if val is None:
            raise TypeError(f"no value for parameter '{pname}'")
        args[pname] = val
    return args

def _format(v):
    if isinstance(v, complex):
        return f"{v.real}+{v.imag}j"
    try:
        c = complex(v)   # int, float, sympy numeric, numpy scalar
    except (TypeError, ValueError):
        return str(v)     # symbolic -> will fail the numeric metric (correct)
    if c.imag != 0:
        return f"{c.real}+{c.imag}j"
    return str(c.real)

def _main():
    try:
        ns = {"__name__": "__main__", "math": math, "cmath": cmath}
        exec(compile(code, "<generated>", "exec"), ns)
        func = _find_function(ns)
        if func is None:
            print("ERROR: no function found in generated code", file=sys.stderr)
            sys.exit(2)
        inputs = json.loads(sys.argv[1])
        result = func(**_match_inputs(func, inputs))
        print(_format(result))
    except SystemExit:
        raise
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

_main()
"""


def _limits(memory_mb: int, cpu_seconds: float) -> Callable[[], None]:
    def _apply() -> None:
        resource.setrlimit(
            resource.RLIMIT_CPU, (max(1, int(cpu_seconds)), max(1, int(cpu_seconds)))
        )
        if memory_mb:
            resource.setrlimit(
                resource.RLIMIT_AS, (memory_mb * 1024 * 1024, memory_mb * 1024 * 1024)
            )
        os.setsid()

    return _apply


def execute_code(
    code: str,
    inputs: dict[str, Any] | None = None,
    timeout_s: float = 2.0,
    memory_mb: int = 512,
    cpu_s: float | None = None,
) -> ExecutionResult:
    """Execute generated code in an isolated subprocess and return the result.

    Args:
        code: the generated Python source (a function definition).
        inputs: kwargs for the generated function.
        timeout_s: wall-clock timeout; the process group is killed on expiry.
        memory_mb: RLIMIT_AS cap in MB (0 = no cap).
        cpu_s: RLIMIT_CPU cap in seconds (defaults to timeout_s).
    """
    try:
        analyze_code(code)
    except SafetyError as exc:
        return ExecutionResult(exit_code=-1, safety_error=str(exc))

    runner = _RUNNER_TEMPLATE
    cpu = cpu_s or timeout_s
    argv_input = json.dumps(inputs or {})
    code_b64 = base64.b64encode(code.encode()).decode()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", runner, argv_input, code_b64],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            preexec_fn=_limits(int(memory_mb), float(cpu)),
            cwd=tempfile.gettempdir(),
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            exit_code=-1, timed_out=True, stderr=f"timed out after {timeout_s}s"
        )
    except OSError as exc:
        return ExecutionResult(exit_code=-1, stderr=f"subprocess error: {exc}")

    # Negative returncode = killed by a signal (SIGXCPU from RLIMIT_CPU, SIGKILL,
    # SIGSEGV, ...) -> treat as a resource/time limit violation.
    if proc.returncode < 0:
        return ExecutionResult(
            exit_code=proc.returncode,
            timed_out=True,
            stderr=f"killed by signal {-proc.returncode} (CPU/time/memory limit)",
        )

    return ExecutionResult(
        exit_code=proc.returncode,
        stdout=proc.stdout.strip(),
        stderr=proc.stderr.strip(),
    )


def run_solution_on_cases(
    code: str,
    test_cases: list[dict[str, Any]],
    timeout_s: float = 2.0,
    memory_mb: int = 512,
) -> tuple[list[str | None], list[str]]:
    """Run `code` on a list of input dicts; returns (outputs, errors)."""
    outputs: list[str | None] = []
    errors: list[str] = []
    for tc in test_cases:
        res = execute_code(code, inputs=tc, timeout_s=timeout_s, memory_mb=memory_mb)
        if res.ok:
            outputs.append(res.stdout or "0")
        else:
            outputs.append(None)
            errors.append(
                res.stderr
                or res.safety_error
                or ("timeout" if res.timed_out else f"exit {res.exit_code}")
            )
    return outputs, errors
