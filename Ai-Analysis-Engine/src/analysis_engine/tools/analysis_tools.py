"""
Analysis tools for the LLM-driven explore loop.

Static plot tools (plot_histogram, plot_scatter, plot_grouped_bar) have been
REMOVED. The LLM generates charts by writing plotly code in run_code.
px and go are pre-injected into the sandbox globals — no import needed.
"""
import json
import pandas as pd
from analysis_engine.tools.base import ToolResult
from analysis_engine.tools.sandbox import SANDBOX_TOOL


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

def describe_column(df: pd.DataFrame, column: str) -> ToolResult:
    if column not in df.columns:
        return ToolResult("describe_column", f"Column '{column}' not found. Available: {list(df.columns)}")
    s = df[column]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        stats = {
            "dtype": str(s.dtype),
            "count": int(s.count()),
            "null_count": int(s.isna().sum()),
            "mean":   round(float(s.mean()), 4)            if s.count() else None,
            "std":    round(float(s.std()), 4)             if s.count() > 1 else None,
            "min":    round(float(s.min()), 4)             if s.count() else None,
            "25%":    round(float(s.quantile(0.25)), 4)    if s.count() else None,
            "median": round(float(s.median()), 4)          if s.count() else None,
            "75%":    round(float(s.quantile(0.75)), 4)    if s.count() else None,
            "max":    round(float(s.max()), 4)             if s.count() else None,
        }
    elif pd.api.types.is_datetime64_any_dtype(s):
        non_null = s.dropna()
        stats = {
            "dtype":      str(s.dtype),
            "count":      int(s.count()),
            "null_count": int(s.isna().sum()),
            "min":        str(non_null.min()) if len(non_null) else None,
            "max":        str(non_null.max()) if len(non_null) else None,
            "note":       "datetime — use run_code for trend analysis",
        }
    else:
        stats = {
            "dtype":         str(s.dtype),
            "count":         int(s.count()),
            "null_count":    int(s.isna().sum()),
            "unique_values": int(s.nunique()),
            # str(k) prevents Timestamp/Period keys from crashing json.dumps
            "top_values":    {str(k): int(v) for k, v in s.value_counts().head(10).items()},
        }
    return ToolResult("describe_column", json.dumps(stats, indent=2))


def correlation(df: pd.DataFrame, col1: str, col2: str) -> ToolResult:
    for c in [col1, col2]:
        if c not in df.columns:
            return ToolResult("correlation", f"Column '{c}' not found.")
        if not pd.api.types.is_numeric_dtype(df[c]):
            return ToolResult("correlation", f"Column '{c}' is not numeric.")
    val = df[col1].corr(df[col2])
    output = {
        "col1": col1, "col2": col2,
        "pearson_correlation": round(float(val), 4),
        "interpretation": (
            "strong positive"   if val > 0.7  else
            "moderate positive" if val > 0.4  else
            "weak positive"     if val > 0.1  else
            "strong negative"   if val < -0.7 else
            "moderate negative" if val < -0.4 else
            "weak negative"     if val < -0.1 else
            "negligible"
        )
    }
    return ToolResult("correlation", json.dumps(output, indent=2))


def groupby_mean(df: pd.DataFrame, group_col: str, value_col: str) -> ToolResult:
    for c in [value_col, group_col]:
        if c not in df.columns:
            return ToolResult("groupby_mean", f"Column '{c}' not found.")
    result = df.groupby(group_col)[value_col].agg(["mean", "count"]).round(4)
    result.columns = ["mean", "count"]
    output = {
        "value_col": value_col,
        "group_col": group_col,
        "groups": result.reset_index().to_dict(orient="records"),
    }
    return ToolResult("groupby_mean", json.dumps(output, indent=2))


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def dispatch_tool(df: pd.DataFrame, tool_name: str, args: dict, run_id: str = "unknown") -> ToolResult:
    if tool_name == "run_code":
        from analysis_engine.tools.sandbox import run_sandbox_code
        return run_sandbox_code(
            df=df,
            code=args["code"],
            purpose=args.get("purpose", ""),
            run_id=run_id,
        )
    dispatch = {
        "describe_column": lambda: describe_column(df, args["column"]),
        "correlation":     lambda: correlation(df, args["col1"], args["col2"]),
        "groupby_mean":    lambda: groupby_mean(df, group_col=args["group_col"], value_col=args["value_col"]),
    }
    fn = dispatch.get(tool_name)
    if not fn:
        return ToolResult(tool_name, f"Unknown tool: '{tool_name}'")
    try:
        return fn()
    except Exception as e:
        return ToolResult(tool_name, f"Tool error: {e}")


# ---------------------------------------------------------------------------
# OpenAI tool definitions — NO static plot tools
# ---------------------------------------------------------------------------

EXPLORE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "describe_column",
            "description": "Get detailed statistics for a single column.",
            "parameters": {
                "type": "object",
                "properties": {"column": {"type": "string"}},
                "required": ["column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "correlation",
            "description": "Compute Pearson correlation between two numeric columns.",
            "parameters": {
                "type": "object",
                "properties": {"col1": {"type": "string"}, "col2": {"type": "string"}},
                "required": ["col1", "col2"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "groupby_mean",
            "description": "Compute mean of a numeric column grouped by a categorical column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value_col": {"type": "string", "description": "Numeric column to aggregate"},
                    "group_col": {"type": "string", "description": "Categorical column to group by"},
                },
                "required": ["value_col", "group_col"]
            }
        }
    },
    SANDBOX_TOOL,
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Call this after at most 8 tool calls to submit your final claims. Do not delay.",
            "parameters": {
                "type": "object",
                "properties": {
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text":           {"type": "string"},
                                "metric":         {"type": "string"},
                                "value":          {"type": "number"},
                                "source_query":   {"type": "string"},
                                "source_columns": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["text", "metric", "value", "source_query", "source_columns"]
                        }
                    }
                },
                "required": ["claims"]
            }
        }
    },
]