import sys
import signal
import traceback
import resource
from dataclasses import dataclass, field
from typing import Any, Optional
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

import pandas as pd
import numpy as np

try:
    import plotly.express as px
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False
    px = None
    go = None

from analysis_engine.tools.sandbox.guard import validate_sandbox_code, get_guard_summary, GuardResult


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SandboxConfig:
    """Sandbox resource limits."""
    max_execution_time_seconds: float = 30.0
    max_memory_mb: int = 512
    max_output_chars: int = 10000
    max_dataframe_rows_for_display: int = 100
    max_series_elements_for_display: int = 50

    # Pre-imported modules (available without import)
    pre_imported: dict = field(default_factory=lambda: {
        "pd": pd,
        "pandas": pd,
        "np": np,
        "numpy": np,
        # plotly available if installed — LLM uses px.bar(), go.Figure() etc.
        **( {"px": px, "go": go} if _PLOTLY_AVAILABLE else {} ),
    })


# Default config
DEFAULT_CONFIG = SandboxConfig()


# ---------------------------------------------------------------------------
# Restricted builtins
# ---------------------------------------------------------------------------

def _create_restricted_builtins() -> dict:
    """
    Create a restricted builtins dict that removes dangerous functions.
    """
    import builtins

    safe_builtins = {}

    # Allowed builtins (explicit allowlist)
    ALLOWED = {
        # Basic types
        "int", "float", "str", "bool", "bytes", "bytearray",
        "list", "tuple", "set", "frozenset", "dict",
        "complex", "range", "enumerate", "zip",
        # None and ellipsis
        "None", "Ellipsis", "NotImplemented",
        # Truth/value testing
        "abs", "all", "any", "len", "max", "min", "sum",
        "sorted", "reversed", "iter", "next",
        # Math
        "round", "pow", "divmod", "hash",
        # String/representation
        "repr", "format", "ascii", "chr", "ord",
        # Type checking
        "isinstance", "issubclass", "callable",
        # Constants
        "True", "False",
        # Iteration helpers
        "filter", "map",
        # Collections (as constructors)
        "slice", "super", "property", "staticmethod", "classmethod",
    }

    for name in ALLOWED:
        if hasattr(builtins, name):
            safe_builtins[name] = getattr(builtins, name)

    return safe_builtins


SAFE_BUILTINS = _create_restricted_builtins()


# ---------------------------------------------------------------------------
# Execution result
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    success: bool
    output: str
    error: Optional[str] = None
    execution_time_seconds: float = 0.0
    returned_value: Any = None
    truncated: bool = False
    guard_result: Optional[GuardResult] = None
    chart_spec: Optional[dict] = None     # set when result is a plotly figure


# ---------------------------------------------------------------------------
# Timeout handler
# ---------------------------------------------------------------------------

class _TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TimeoutError("Execution timed out")


# ---------------------------------------------------------------------------
# Main executor
# ---------------------------------------------------------------------------

def execute_sandbox_code(
    code: str,
    df: pd.DataFrame,
    config: SandboxConfig = DEFAULT_CONFIG,
    extra_globals: Optional[dict] = None,
) -> ExecutionResult:
    """
    Execute code in the sandbox with the given dataframe available as 'df'.

    The code is executed with:
    - 'df' as the primary dataframe
    - pandas/numpy pre-imported as pd/np
    - Restricted builtins (no exec, eval, open, etc.)
    - Time and memory limits
    - Output captured and truncated if needed

    Args:
        code: Python code to execute (must pass validate_sandbox_code first)
        df: DataFrame to make available to the code
        config: Resource limits
        extra_globals: Additional globals to inject (use with caution)

    Returns:
        ExecutionResult with output, error, timing info
    """
    import time

    # 1. Validate code FIRST
    guard_result = validate_sandbox_code(code)
    if not guard_result.allowed:
        return ExecutionResult(
            success=False,
            output="",
            error=f"Code failed safety validation:\n{get_guard_summary(guard_result)}",
            guard_result=guard_result,
        )

    # 2. Prepare execution environment
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    # Build restricted globals
    exec_globals = {
        "__builtins__": SAFE_BUILTINS,
        "df": df,
        # Pre-imported modules
        **config.pre_imported,
    }

    # Add extra globals if provided (with warning — this can weaken sandbox)
    if extra_globals:
        exec_globals.update(extra_globals)

    # 3. Set resource limits (Unix only)
    old_limits = None
    if sys.platform != "win32":
        try:
            old_limits = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(
                resource.RLIMIT_AS,
                (config.max_memory_mb * 1024 * 1024,
                 config.max_memory_mb * 1024 * 1024)
            )
        except (ValueError, resource.error):
            old_limits = None

    # 4. Execute with timeout
    start_time = time.time()
    result_value = None
    error_msg = None
    truncated = False

    # Set signal-based timeout (Unix only)
    old_handler = None
    if sys.platform != "win32":
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(int(config.max_execution_time_seconds) + 1)

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # Compile first to catch syntax errors cleanly
            compiled = compile(code, "<sandbox>", "exec")

            # Execute
            exec(compiled, exec_globals)

            # Check if code set a 'result' variable
            result_value = exec_globals.get("result")

    except _TimeoutError:
        error_msg = f"Execution timed out after {config.max_execution_time_seconds}s"
    except MemoryError:
        error_msg = f"Memory limit exceeded ({config.max_memory_mb}MB)"
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
    finally:
        # Restore signal handler
        if sys.platform != "win32" and old_handler is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        # Restore resource limits to what they were before
        if sys.platform != "win32" and old_limits is not None:
            try:
                resource.setrlimit(resource.RLIMIT_AS, old_limits)
            except (ValueError, resource.error):
                pass

    execution_time = time.time() - start_time

    # 5. Capture and truncate output
    stdout_output = stdout_capture.getvalue()
    stderr_output = stderr_capture.getvalue()

    output = stdout_output
    if stderr_output and not error_msg:
        output += "\n[stderr]\n" + stderr_output

    if len(output) > config.max_output_chars:
        output = output[:config.max_output_chars] + f"\n... [truncated, {len(output)} total chars]"
        truncated = True

    # 6. Format return value if present
    chart_spec = None
    if result_value is not None and not error_msg:
        # detect plotly figure BEFORE formatting as text
        chart_spec = _try_extract_chart_spec(result_value)
        if chart_spec:
            result_str = f"Chart generated: {chart_spec.get('layout', {}).get('title', {}).get('text', 'untitled')}"
        else:
            result_str = _format_result_value(result_value, config)
        if output:
            output += "\n\n[result]\n" + result_str
        else:
            output = "[result]\n" + result_str

    return ExecutionResult(
        success=error_msg is None,
        output=output.strip(),
        error=error_msg,
        execution_time_seconds=round(execution_time, 3),
        returned_value=result_value,
        truncated=truncated,
        guard_result=guard_result,
        chart_spec=chart_spec,
    )


def _try_extract_chart_spec(value: Any) -> dict | None:
    """
    If value is a plotly Figure, serialize it to a JSON-compatible dict.
    Returns None for anything that isn't a plotly figure.
    """
    if not _PLOTLY_AVAILABLE:
        return None
    try:
        import plotly.graph_objects as _go
        if isinstance(value, _go.Figure):
            return value.to_dict()
    except Exception:
        pass
    return None


def _format_result_value(value: Any, config: SandboxConfig) -> str:
    """Format a return value for display, handling large dataframes safely."""
    if isinstance(value, pd.DataFrame):
        if len(value) > config.max_dataframe_rows_for_display:
            return f"DataFrame with {len(value)} rows x {len(value.columns)} columns\n[showing first {config.max_dataframe_rows_for_display} rows]\n{value.head(config.max_dataframe_rows_for_display).to_string()}"
        return value.to_string()

    if isinstance(value, pd.Series):
        if len(value) > config.max_series_elements_for_display:
            return f"Series with {len(value)} elements\n[showing first {config.max_series_elements_for_display}]\n{value.head(config.max_series_elements_for_display).to_string()}"
        return value.to_string()

    if isinstance(value, (np.ndarray, list, tuple, set)):
        s = str(value)
        if len(s) > 2000:
            return f"{type(value).__name__} (length {len(value)}) [output truncated]"
        return s

    # For other values, use repr but limit length
    s = repr(value)
    if len(s) > 2000:
        return f"{type(value).__name__}: {s[:2000]}... [truncated]"
    return s