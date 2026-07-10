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


TOLERANCE = 0.02   # 2% relative tolerance — allows for float rounding


def _pct_diff(a: float, b: float) -> float:
    if b == 0:
        return abs(a)
    return abs(a - b) / abs(b)


# ---------------------------------------------------------------------------
# Recompute tools — mirror the analysis tools but labelled as "recompute_*"
# so the LLM knows they're for verification, not new exploration
# ---------------------------------------------------------------------------

def recompute_groupby_mean(
    df: pd.DataFrame, value_col: str, group_col: str, group_value: str
) -> ToolResult:
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
    val = round(float(result[group_value]), 4)
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
    condition examples: "> 0", "< 100", ">= 1", "== True"
    For string equality use recompute_value_count instead.
    """
    if column not in df.columns:
        return ToolResult("recompute_count", f"Column '{column}' not found.")
    try:
        series = df[column]
        op = condition.strip()
        if op.startswith(">= "):
            mask = series >= float(op[3:])
        elif op.startswith("<= "):
            mask = series <= float(op[3:])
        elif op.startswith("> "):
            mask = series > float(op[2:])
        elif op.startswith("< "):
            mask = series < float(op[2:])
        elif op.startswith("== "):
            val = op[3:].strip().strip('"').strip("'")
            if val == "True":
                mask = series == True
            elif val == "False":
                mask = series == False
            else:
                try:
                    mask = series == float(val)
                except ValueError:
                    mask = series == val
        else:
            return ToolResult("recompute_count",
                f"Unsupported condition: '{condition}'. "
                "For string matching use recompute_value_count instead.")
        count = int(mask.sum())
        return ToolResult("recompute_count", json.dumps({
            "column": column,
            "condition": condition,
            "recomputed_count": count,
        }, indent=2))
    except Exception as e:
        return ToolResult("recompute_count", f"Error: {e}")


def recompute_value_count(df: pd.DataFrame, column: str, value: str) -> ToolResult:
    """
    Count rows where column exactly matches a string value.
    Case-insensitive — handles 'Sedan', 'SEDAN', 'sedan' all the same.
    Also returns close matches to help diagnose mismatches.
    """
    if column not in df.columns:
        return ToolResult("recompute_value_count", f"Column '{column}' not found.")

    exact = int((df[column] == value).sum())
    lower = int((df[column].astype(str).str.lower() == value.lower()).sum())

    # show top actual values so LLM knows what strings exist
    top = {str(k): int(v) for k, v in df[column].value_counts().head(10).items()}

    return ToolResult("recompute_value_count", json.dumps({
        "column": column,
        "searched_value": value,
        "exact_match_count": exact,
        "case_insensitive_count": lower,
        "top_10_actual_values": top,
    }, indent=2))


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
        "recompute_value_count":  lambda: recompute_value_count(df, args["column"], args["value"]),
    }

    #Web search needs to be handeled dfferently because it dont need dataframe.
    if tool_name == "web_search":
        from analysis_engine.tools.web_search import web_search as _web_search
        return _web_search(args["query"])

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
            "name": "recompute_value_count",
            "description": "Count rows where a column matches a specific string value. Case-insensitive. Use this for claims like 'there are 660,172 Sedans' or 'Driver Inattention is the top factor'. Returns top 10 actual values so you can see exact spellings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "value":  {"type": "string", "description": "The string value to count"},
                },
                "required": ["column", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "REQUIRED: Call this once after verifying all claims. You MUST call this even if some claims are unverifiable. Do not stop without calling finish().",
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
    },
    {
    "type": "function",
    "function":{
        "name": "web_Search",
        "descrption":(
             "Search the web to find external context for a claim that references "
            "industry trends, market conditions, or other external facts. "
            "Do NOT use this to verify internal data numbers — use recompute_* tools for those. "
            "Only use for claims like 'consistent with industry trends' or 'reflects a market-wide pattern'."
        ),
        "parameters":{
            "type": "object",
            "properties": {
                "query":{
                    "type": "string",
                    "description": "Specific search query. Be precise — include dates, industry, region if relevant."
                }
            },
            "required": ["query"]
        }
    }
}
]