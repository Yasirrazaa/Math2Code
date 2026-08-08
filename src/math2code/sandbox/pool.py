"""High-throughput persistent-worker sandbox.

The standalone `execute_code` in base.py spawns a fresh subprocess per call and
re-imports numpy/sympy every time (~300 ms/exec). RL rollouts and large eval runs
need ~10-50 ms/exec, so we keep a pool of worker processes alive; each worker
imports numpy/sympy once and then executes many snippets in fresh namespaces.

Isolation model (per worker):
  - RLIMIT_AS memory cap (set once at worker start)
  - per-task SIGALRM wall-clock timeout
  - AST allowlist analysis in the parent (same policy as base.py)
  - a task that hangs only blocks its own worker (bounded by the alarm)
"""

from __future__ import annotations

import concurrent.futures as cf
import resource
import signal
from concurrent.futures.process import BrokenProcessPool
from typing import Any

from math2code.sandbox import base
from math2code.sandbox._runtime import find_function, format_result, match_inputs

_MAX_CPU_SECONDS = 30  # generous per-process ceiling; per-task alarm is tighter


def _worker_init(memory_mb: int) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (_MAX_CPU_SECONDS, _MAX_CPU_SECONDS))
    if memory_mb:
        resource.setrlimit(
            resource.RLIMIT_AS, (memory_mb * 1024 * 1024, memory_mb * 1024 * 1024)
        )


class _TaskTimeoutError(Exception):
    pass


def _run_one(
    payload: tuple[str, dict[str, Any]], timeout_s: float
) -> tuple[int, str, str, bool]:
    """Worker entrypoint. Returns (exit_code, stdout, stderr, timed_out)."""
    code, inputs = payload

    def _alarm(_signum: int, _frame: Any) -> None:
        raise _TaskTimeoutError()

    old_handler = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        ns: dict[str, Any] = {"__name__": "__main__"}
        exec(compile(code, "<generated>", "exec"), ns)
        func = find_function(ns)
        if func is None:
            return 2, "", "ERROR: no function found in generated code", False
        result = func(**match_inputs(func, inputs))
        return 0, format_result(result), "", False
    except _TaskTimeoutError:
        return -1, "", f"timed out after {timeout_s}s", True
    except SystemExit:
        raise
    except Exception as exc:
        return 1, "", f"{type(exc).__name__}: {exc}", False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


class SandboxPool:
    """Reusable worker pool for executing generated code at RL/eval throughput."""

    def __init__(
        self,
        n_workers: int = 4,
        timeout_s: float = 2.0,
        memory_mb: int = 1536,
    ) -> None:
        self.timeout_s = timeout_s
        self.n_workers = n_workers
        self.memory_mb = memory_mb
        self._pool = cf.ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(memory_mb,),
        )

    def execute(
        self, code: str, inputs: dict[str, Any] | None = None
    ) -> base.ExecutionResult:
        try:
            base.analyze_code(code)
        except base.SafetyError as exc:
            return base.ExecutionResult(exit_code=-1, safety_error=str(exc))
        try:
            return self._submit_and_wait(code, inputs or {})
        except BrokenProcessPool:
            # a worker crashed (e.g. sympy segfault on pathological input):
            # rebuild the pool and retry once rather than killing the whole eval
            self._restart_pool()
            try:
                return self._submit_and_wait(code, inputs or {})
            except BrokenProcessPool:
                return base.ExecutionResult(
                    exit_code=-1, stderr="worker crashed twice; pool unhealthy"
                )

    def _submit_and_wait(
        self, code: str, inputs: dict[str, Any]
    ) -> base.ExecutionResult:
        fut = self._pool.submit(_run_one, (code, inputs), self.timeout_s)
        try:
            exit_code, stdout, stderr, timed_out = fut.result(
                timeout=self.timeout_s + 1.0
            )
        except cf.TimeoutError:
            # the worker may still be stuck; it can only harm itself (alarm/limits)
            return base.ExecutionResult(
                exit_code=-1, timed_out=True, stderr="pool timeout"
            )
        except BrokenProcessPool:
            # let execute() self-heal and retry once
            raise
        except (
            Exception
        ) as exc:  # worker crashed (e.g. MemoryError) -> treat as failure
            return base.ExecutionResult(exit_code=-1, stderr=f"worker error: {exc}")
        return base.ExecutionResult(
            exit_code=exit_code, stdout=stdout, stderr=stderr, timed_out=timed_out
        )

    def _restart_pool(self) -> None:
        old = self._pool
        self._pool = cf.ProcessPoolExecutor(
            max_workers=self.n_workers,
            initializer=_worker_init,
            initargs=(self.memory_mb,),
        )
        old.shutdown(wait=False, cancel_futures=True)

    def run_solution_on_cases(
        self,
        code: str,
        test_cases: list[dict[str, Any]],
    ) -> tuple[list[str | None], list[str]]:
        outputs: list[str | None] = []
        errors: list[str] = []
        for tc in test_cases:
            res = self.execute(code, inputs=tc)
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

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> SandboxPool:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
