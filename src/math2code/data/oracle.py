"""Verification oracle: prove a candidate solution is correct against ground truth.

Layered checks, cheapest first (used by curate.py, RL reward shaping, and eval):

1. `syntactic_check`      — AST parses, matches the safe import set, defines a
                            function (module-level analysis, no execution).
2. `numeric_check`        — Monte Carlo: run on `n` random fresh inputs (drawn
                            with pole avoidance) and compare to the ground-truth
                            sympy expression with cmath tolerance.
3. `identity_check`       — symbolic: sp.simplify(generated - ground_truth) == 0.
                            Only meaningful for pure symbolic expressions.
4. `oracle_verify`        — full pipeline with a pass/fail verdict + reasons.

Numeric tolerance: rel_tol=1e-5, abs_tol=1e-8 (looser than the competition's
1e-6/1e-9 metric: the oracle catches real bugs, the metric sets the bar).
"""

from __future__ import annotations

import cmath
import random
from typing import Any

import sympy as sp

from math2code.sandbox import SandboxPool, analyze_code, execute_code
from math2code.schemas import MathCodePair

ORACLE_REL_TOL = 1e-5
ORACLE_ABS_TOL = 1e-8


def _sample_points(
    pair: MathCodePair, n: int, rng: random.Random
) -> list[dict[str, Any]]:
    """Fresh inputs using the pair's own test-case key contract (jittered values).

    Using ground-truth free symbols would drop parameters the gold function
    actually takes (e.g. the summation index `y` in `summation_function(x, y)`).
    Jitter keeps `x`/`x_val` variants coupled (data.competition.jitter_inputs).
    """
    from math2code.data.competition import jitter_inputs

    templates = [tc.input for tc in pair.test_cases] or [{}]
    return [jitter_inputs(templates[i % len(templates)], rng) for i in range(n)]


def _parse_ground_truth(pair: MathCodePair) -> sp.Expr | None:
    if not pair.sympy_exp:
        return None
    try:
        return sp.sympify(pair.sympy_exp)
    except Exception:
        return None


def syntactic_check(code: str) -> tuple[bool, str]:
    """Module-level safety + shape check without executing."""
    try:
        analyze_code(code)
    except Exception as exc:  # SafetyError or SyntaxError
        return False, str(exc)
    if "def calculate" not in code and "def " not in code:
        return False, "no function definition found"
    return True, "ok"


def numeric_check(
    pair: MathCodePair,
    candidate_code: str,
    n: int = 8,
    seed: int = 0,
    pool: SandboxPool | None = None,
) -> tuple[bool, str]:
    """Compare candidate vs ground truth on random inputs (pole-avoiding).

    Ground truth is either the sympy_exp (symbolic path) or, when the pair
    carries `truth_code` (gcd/lcm/digit-op families), the truth CODE executed
    in the same sandbox contract — symmetric to the candidate so an honest
    family can never hide behind a wrong truth.
    """
    rng = random.Random(f"oracle:{pair.task_id}:{seed}")
    inputs = _sample_points(pair, n, rng)

    # ground-truth outputs: truth_code (sandbox) XOR sympy_exp (deterministic)
    if pair.truth_code:
        if pool is not None:
            gt_outputs, errors = pool.run_solution_on_cases(pair.truth_code, inputs)
            if any(o is None for o in gt_outputs):
                return False, f"truth code failed to run: {errors[0][:120]}"
        else:
            gt_outputs = []
            for inp in inputs:
                res = execute_code(pair.truth_code, inputs=inp)
                if not res.ok:
                    return False, f"truth code failed to run: {res.stderr[:120]}"
                gt_outputs.append(res.stdout)
        gt_complex: list[complex | None] = []
        from math2code.evaluation.metrics import parse_number

        for s in gt_outputs:
            try:
                gt_complex.append(parse_number(s))
            except (ValueError, TypeError):
                gt_complex.append(None)
    else:
        expr = _parse_ground_truth(pair)
        if expr is None:
            return False, "no ground-truth sympy_exp to compare against"
        gt_complex = []
        for inp in inputs:
            subs = {sp.Symbol(k): v for k, v in inp.items()}
            try:
                gt_complex.append(complex(expr.subs(subs).evalf()))
            except Exception:
                gt_complex.append(None)

    # candidate outputs via the sandbox
    if pool is not None:
        cand_outputs, errors = pool.run_solution_on_cases(candidate_code, inputs)
        if any(o is None for o in cand_outputs):
            return False, f"candidate failed to run: {errors[0][:120]}"
    else:
        cand_outputs = []
        for inp in inputs:
            res = execute_code(candidate_code, inputs=inp)
            if not res.ok:
                return False, f"candidate failed to run: {res.stderr[:120]}"
            cand_outputs.append(res.stdout)

    from math2code.evaluation.metrics import parse_number

    for i, (cand, gt) in enumerate(zip(cand_outputs, gt_complex)):
        if gt is None or not cmath.isfinite(gt):
            continue  # skip points where ground truth itself is undefined
        try:
            got = parse_number(cand)
        except ValueError:
            return False, f"candidate output unparseable at point {i}: {cand!r}"
        if not cmath.isclose(got, gt, rel_tol=ORACLE_REL_TOL, abs_tol=ORACLE_ABS_TOL):
            return False, (f"mismatch at point {i}: candidate={got} ground_truth={gt}")
    return True, "ok"


def identity_check(pair: MathCodePair, candidate_code: str) -> tuple[bool, str]:
    """Symbolic identity: simplify(candidate - ground_truth) == 0.

    Only attempted when the candidate function takes no arguments (numeric
    check already covered parameterized functions).
    """
    expr = _parse_ground_truth(pair)
    if expr is None:
        return False, "no ground-truth sympy_exp"
    try:
        from math2code.sandbox._runtime import find_function, format_result

        ns: dict[str, Any] = {"__name__": "__main__"}
        exec(compile(candidate_code, "<oracle>", "exec"), ns)
        func = find_function(ns)
        if func is None:
            return False, "no function found"
        import inspect

        if len(inspect.signature(func).parameters) > 0:
            return False, "identity not applicable (function takes args)"
        out = func()
        cand = sp.sympify(format_result(out))
        diff = sp.simplify(cand - expr)
        if diff == 0:
            return True, "ok"
        return False, f"simplify(candidate - truth) != 0 ({diff})"
    except Exception as exc:
        return False, f"identity check error: {exc}"


def oracle_verify(
    pair: MathCodePair,
    candidate_code: str,
    pool: SandboxPool | None = None,
) -> tuple[bool, list[str]]:
    """Full verification. Returns (passed, reasons list)."""
    reasons: list[str] = []
    ok_syntax, why = syntactic_check(candidate_code)
    reasons.append(f"syntax: {'pass' if ok_syntax else why}")
    if not ok_syntax:
        return False, reasons
    ok_num, why = numeric_check(pair, candidate_code, pool=pool)
    reasons.append(f"numeric: {'pass' if ok_num else why}")
    if ok_num:
        return True, reasons
    ok_id, why = identity_check(pair, candidate_code)
    reasons.append(f"identity: {'pass' if ok_id else why}")
    return ok_id, reasons
