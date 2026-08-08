"""Shared child-side runtime helpers (no heavy imports; safe to inline/import)."""

from __future__ import annotations

import inspect
from typing import Any


def find_function(ns: dict[str, Any]) -> Any | None:
    """Locate the entry function in an executed namespace."""
    if "calculate" in ns and callable(ns["calculate"]):
        return ns["calculate"]
    for name, obj in ns.items():
        if name.startswith("_"):
            continue
        if callable(obj) and getattr(obj, "__module__", None) == "__main__":
            return obj
    return None


def match_inputs(func: Any, inputs: dict[str, Any]) -> dict[str, Any]:
    """Map test-case input keys to function parameters.

    Handles the competition's `_val` / `_value` suffix convention, e.g. a
    function parameter ``x_val`` fed by input key ``x``.
    """
    sig = inspect.signature(func)
    args: dict[str, Any] = {}
    for pname, param in sig.parameters.items():
        val = inputs.get(pname)
        if val is None and pname.endswith("_val"):
            val = inputs.get(pname[: -len("_val")])
        if val is None and pname.endswith("_value"):
            val = inputs.get(pname[: -len("_value")])
        if val is None and param.default is not inspect.Parameter.empty:
            val = param.default
        if val is None:
            raise TypeError(f"no value for parameter '{pname}'")
        args[pname] = val
    return args


def format_result(v: Any) -> str:
    """Canonical string form: floats as str(float), complex as 're+imj'."""
    if isinstance(v, complex):
        return f"{v.real}+{v.imag}j"
    try:
        c = complex(v)  # int, float, sympy numeric, numpy scalar
    except (TypeError, ValueError):
        return str(v)  # symbolic -> fails the numeric metric (correct)
    if c.imag != 0:
        return f"{c.real}+{c.imag}j"
    return str(c.real)
