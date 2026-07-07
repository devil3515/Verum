"""
Verification tools — Stage A: internal recomputation.
 
The LLM wrote claims with values it observed from tool outputs. We now
independently re-derive those values from the cleaned dataframe to confirm
they're actually correct — not hallucinated or misremembered across turns.
 
Same tool-loop pattern as cleaning and analysis:
  profile → tool loop → finish()
 
The LLM gets each claim's source_query and calls the matching recompute
tool. We compare the recomputed value to the claimed value within a
tolerance. Mismatch → contradicted. Match → confirmed.
"""

import json

import pandas as pd

from analysis_engine.tools.base import ToolResult

TOLERANCE = 0.02


def _pct_diff(a: float, b: float) -> float:
    if b == 0:
        return abs(a)
    return abs(a - b) / abs(b)


# ---------------------------------------------------------------------------
# Recompute tools — mirror the analysis tools but labelled as "recompute_*"
# so the LLM knows they're for verification, not new exploration
# ---------------------------------------------------------------------------

def recompute_groupby_mean( df: pd.DataFrame, value_col: str, group_col: str, group_value: str) -> ToolResult:
    for c in [value_col, group_col]:
        if c not in df.columns:
            return ToolResult("recompute_groupby_mean", f"Column '{c}' not found.")
    result = df.groupby(group_col)[value_col].mean()
    if group_value not in result.index:
        return ToolResult(
            "recompute_groupby_mean",
            f"Group '{group_value}' not found in '{group_col}'. "
            f"Available: {result.index.tolist()}"
        )
    val = round(float(result[group_value]),4)
    return ToolResult("recompute_groupby_mean", json.dumps({
        "value_col": value_col,
        "group_col": group_col,
        "group_value": group_value,
        "recomputed_mean": val,
        "all_groups": result.round(4).to_dict(),
    }, indent=2))


def recompute_column_mean(df: pd.DataFrame, column: str) -> ToolResult:
    if column not in df.columns:
        return ToolResult("recompute_column_mean", f"Column '{column}' not found.")
    val = round(float(df[column].dropna().mean()), 4)
    return ToolResult("recompute_column_mean", json.dumps({
        "column": column,
        "recomputed_mean": val,
        "count": int(df[column].dropna().count()),
    }, indent=2))
 
 
def recompute_correlation(df: pd.DataFrame, col1: str, col2: str) -> ToolResult:
    for c in [col1, col2]:
        if c not in df.columns:
            return ToolResult("recompute_correlation", f"Column '{c}' not found.")
    val = round(float(df[col1].corr(df[col2])), 4)
    return ToolResult("recompute_correlation", json.dumps({
        "col1": col1,
        "col2": col2,
        "recomputed_correlation": val,
    }, indent=2))
 
 
def recompute_count(df: pd.DataFrame, column: str, condition: str) -> ToolResult:
    """
    condition: a simple pandas-style filter string, e.g. "> 0", "== True"
    Safely evaluated against the column only — not arbitrary exec.
    """
    if column not in df.columns:
        return ToolResult("recompute_count", f"Column '{column}' not found.")
    try:
        # safe: only column series operations, no exec
        series = df[column]
        op = condition.strip()
        if op.startswith("> "):
            mask = series > float(op[2:])
        elif op.startswith("< "):
            mask = series < float(op[2:])
        elif op.startswith("== "):
            val = op[3:].strip()
            mask = series == (True if val == "True" else False if val == "False" else val)
        elif op.startswith(">= "):
            mask = series >= float(op[3:])
        elif op.startswith("<= "):
            mask = series <= float(op[3:])
        else:
            return ToolResult("recompute_count", f"Unsupported condition format: '{condition}'")
        count = int(mask.sum())
        return ToolResult("recompute_count", json.dumps({
            "column": column,
            "condition": condition,
            "recomputed_count": count,
        }, indent=2))
    except Exception as e:
        return ToolResult("recompute_count", f"Error evaluating condition: {e}")
 
 
# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
 
def dispatch_verification_tool(
    df: pd.DataFrame, tool_name: str, args: dict
) -> ToolResult:
    dispatch = {
        "recompute_groupby_mean": lambda: recompute_groupby_mean(
            df, args["value_col"], args["group_col"], args["group_value"]
        ),
        "recompute_column_mean": lambda: recompute_column_mean(df, args["column"]),
        "recompute_correlation":  lambda: recompute_correlation(df, args["col1"], args["col2"]),
        "recompute_count":        lambda: recompute_count(df, args["column"], args["condition"]),
    }
    fn = dispatch.get(tool_name)
    if not fn:
        return ToolResult(tool_name, f"Unknown verification tool: '{tool_name}'")
    try:
        return fn()
    except Exception as e:
        return ToolResult(tool_name, f"Tool error: {e}")
 
 
# ---------------------------------------------------------------------------
# OpenAI tool definitions
# ---------------------------------------------------------------------------
 
VERIFICATION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recompute_groupby_mean",
            "description": "Independently recompute the mean of a numeric column for a specific group. Use to verify claims like 'EMEA mean revenue is X'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value_col":   {"type": "string", "description": "Numeric column"},
                    "group_col":   {"type": "string", "description": "Categorical column to group by"},
                    "group_value": {"type": "string", "description": "The specific group to check"},
                },
                "required": ["value_col", "group_col", "group_value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recompute_column_mean",
            "description": "Recompute the overall mean of a numeric column to verify a claimed average.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"}
                },
                "required": ["column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recompute_correlation",
            "description": "Recompute Pearson correlation between two columns to verify a claimed correlation value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {"type": "string"},
                    "col2": {"type": "string"},
                },
                "required": ["col1", "col2"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recompute_count",
            "description": "Recompute a count matching a condition to verify claims like 'there are 150 outliers'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column":    {"type": "string"},
                    "condition": {
                        "type": "string",
                        "description": "Simple condition string: '> 0', '== True', '< 100', etc."
                    },
                },
                "required": ["column", "condition"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Call when all claims have been verified. Return verdict for each claim.",
            "parameters": {
                "type": "object",
                "properties": {
                    "verdicts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "claim_id":   {"type": "string"},
                                "status":     {
                                    "type": "string",
                                    "enum": ["confirmed", "contradicted", "unverifiable"]
                                },
                                "confidence": {"type": "number"},
                                "reasoning":  {"type": "string"},
                                "recomputed_value": {"type": "number"}
                            },
                            "required": ["claim_id", "status", "confidence", "reasoning"]
                        }
                    }
                },
                "required": ["verdicts"]
            }
        }
    }
]
 