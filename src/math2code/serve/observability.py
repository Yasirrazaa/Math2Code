"""Optional OpenTelemetry / Weave tracing for the serve API.

A thin, opt-in hook: no tracing deps are imported unless `MATH2CODE_TRACE=1`
is set AND `weave`/`openinference` are installed. Default behavior (no env
var) is zero overhead — the API proxy works on CPU-only boxes without wandb.
"""

from __future__ import annotations

import os


def maybe_register_tracing() -> bool:
    """Register Weave tracing if opted in. Returns True when active."""
    if os.environ.get("MATH2CODE_TRACE") != "1":
        return False
    try:
        import weave  # type: ignore[import-not-found]
        from openinference.instrumentation.openai import (  # type: ignore[import-not-found]
            OpenAIInstrumentor,
        )
    except ImportError:
        print("  [trace] MATH2CODE_TRACE=1 but weave/openinference missing; skipping")
        return False
    try:
        weave.init("math2code-serve")
        OpenAIInstrumentor().instrument()
        print("  [trace] Weave tracing active")
        return True
    except Exception as exc:  # network/wandb login failures must not kill serve
        print(f"  [trace] weave.init failed ({exc}); continuing untraced")
        return False
