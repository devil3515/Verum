"""
Sandbox package — exposes two things to the rest of the codebase:

  SANDBOX_TOOL   — OpenAI tool definition, appended to each agent's tool list
  run_sandbox_code() — executes code through guard → executor → audit pipeline

Nothing outside this package should import from executor/guard/audit directly.
"""
import time

from analysis_engine.tools.base import ToolResult
from analysis_engine.tools.sandbox.guard import validate_sandbox_code, get_guard_summary
from analysis_engine.tools.sandbox.executor import execute_sandbox_code, DEFAULT_CONFIG
from analysis_engine.tools.sandbox.audit import get_audit_logger


# ---------------------------------------------------------------------------
# OpenAI tool definition — appended to each agent's EXPLORE_TOOLS list
# ---------------------------------------------------------------------------

SANDBOX_TOOL = {
    "type": "function",
    "function": {
        "name": "run_code",
        "description": (
            "Execute a short pandas snippet against the dataframe (`df` is pre-loaded). "
            "Use this for computations the other tools don't cover — e.g. value_counts, "
            "custom filtering, multi-step aggregations. "
            "Set result = <value> to capture output. "
            "Do NOT use for anything the named tools already handle."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. 'df' is available. Set result = <value> to return output."
                },
                "purpose": {
                    "type": "string",
                    "description": "One sentence explaining what this code computes and why."
                }
            },
            "required": ["code", "purpose"]
        }
    }
}


# ---------------------------------------------------------------------------
# Execution entry point
# ---------------------------------------------------------------------------

def run_sandbox_code(
    df,
    code: str,
    purpose: str,
    run_id: str = "unknown",
    agent: str = "analysis",
) -> ToolResult:
    """
    Run code through the full guard → execute → audit pipeline.
    Returns a ToolResult the explore_loop feeds back to the LLM.
    """
    logger = get_audit_logger()
    start = time.time()

    # Guard first — fast AST check, no execution
    guard_result = validate_sandbox_code(code)

    if not guard_result.allowed:
        summary = get_guard_summary(guard_result)
        elapsed = time.time() - start
        logger.log_execution(
            code=code,
            purpose=_sanitize_purpose(purpose),
            validation_passed=False,
            validation_errors=summary,
            execution_success=False,
            execution_error=None,
            execution_time=elapsed,
            output="",
            run_id=run_id,
            agent=agent,
        )
        return ToolResult(
            tool_name="run_code",
            output=f"Code blocked by safety guard:\n{summary}"
        )

    # Execute in sandbox
    exec_result = execute_sandbox_code(code=code, df=df, config=DEFAULT_CONFIG)
    elapsed = time.time() - start

    logger.log_execution(
        code=code,
        purpose=_sanitize_purpose(purpose),
        validation_passed=True,
        validation_errors=None,
        execution_success=exec_result.success,
        execution_error=exec_result.error,
        execution_time=elapsed,
        output=exec_result.output,
        run_id=run_id,
        agent=agent,
    )

    if not exec_result.success:
        return ToolResult(
            tool_name="run_code",
            output=f"Execution error: {exec_result.error}"
        )

    output = exec_result.output or "(code ran successfully, no output)"
    if exec_result.truncated:
        output += "\n[output was truncated]"

    # if the code produced a plotly figure, attach the chart spec
    return ToolResult(
        tool_name="run_code",
        output=output,
        chart_spec=exec_result.chart_spec,   # None if no figure was produced
    )


def _sanitize_purpose(purpose: str) -> str:
    """
    Sanitize LLM-generated purpose string before it enters audit logs.
    Strips non-printable chars and limits length so injection attempts
    in the purpose field don't pollute the audit log.
    """
    sanitized = "".join(c for c in purpose if c.isprintable())
    return sanitized[:200]