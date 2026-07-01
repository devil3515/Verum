"""
Cleaning tools exposed to the LLM during the cleaning tool loop.

Same pattern as analysis_tools.py — the LLM calls these iteratively
after seeing the data profile. It decides which operations to apply
and in what order, rather than running a fixed pipeline blindly.

Every tool appends to a cleaning log so the full audit trail is preserved.
"""
import json
import pandas as pd

from analysis_engine.tools.base import ToolResult
from analysis_engine.tools.sandbox import SANDBOX_TOOL
from analysis_engine.state import CleaningLogEntry


# ---------------------------------------------------------------------------
# Tool functions — each returns (df, ToolResult) since cleaning mutates data
# ---------------------------------------------------------------------------

def profile_column(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, ToolResult]:
    """Profile a column before deciding what cleaning to apply."""
    if column not in df.columns:
        return df, ToolResult("profile_column", f"Column '{column}' not found. Available: {list(df.columns)}")

    s = df[column]
    stats: dict = {
        "column": column,
        "dtype": str(s.dtype),
        "total_rows": len(df),
        "null_count": int(s.isna().sum()),
        "null_pct": round(s.isna().mean() * 100, 2),
        "unique_count": int(s.nunique()),
        "duplicate_rows_total": int(df.duplicated().sum()),
    }

    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        non_null = s.dropna()
        if len(non_null):
            q1, q3 = non_null.quantile(0.25), non_null.quantile(0.75)
            iqr = q3 - q1
            outliers = ((non_null < q1 - 1.5 * iqr) | (non_null > q3 + 1.5 * iqr)).sum()
            stats.update({
                "min": round(float(non_null.min()), 4),
                "max": round(float(non_null.max()), 4),
                "mean": round(float(non_null.mean()), 4),
                "median": round(float(non_null.median()), 4),
                "std": round(float(non_null.std()), 4) if len(non_null) > 1 else 0,
                "iqr_outlier_count": int(outliers),
            })
    else:
        stats["top_values"] = s.value_counts().head(5).to_dict()

    return df, ToolResult("profile_column", json.dumps(stats, indent=2))


def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, ToolResult, list[CleaningLogEntry]]:
    before = len(df)
    cleaned = df.drop_duplicates()
    dropped = before - len(cleaned)
    log = []
    if dropped > 0:
        log.append(CleaningLogEntry(
            operation="drop_duplicates",
            column="(all columns)",
            rows_affected=dropped,
            rationale=f"Removed {dropped} fully duplicate row(s).",
        ))
    msg = f"Dropped {dropped} duplicate rows. Remaining: {len(cleaned)} rows."
    return cleaned, ToolResult("drop_duplicates", msg), log


def drop_nulls(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, ToolResult, list[CleaningLogEntry]]:
    if column not in df.columns:
        return df, ToolResult("drop_nulls", f"Column '{column}' not found."), []
    before = len(df)
    cleaned = df.dropna(subset=[column])
    dropped = before - len(cleaned)
    log = []
    if dropped > 0:
        log.append(CleaningLogEntry(
            operation="drop_nulls",
            column=column,
            rows_affected=dropped,
            rationale=f"Dropped {dropped} rows with null '{column}'.",
        ))
    msg = f"Dropped {dropped} null rows in '{column}'. Remaining: {len(cleaned)} rows."
    return cleaned, ToolResult("drop_nulls", msg), log


def fill_nulls(
    df: pd.DataFrame, column: str, strategy: str
) -> tuple[pd.DataFrame, ToolResult, list[CleaningLogEntry]]:
    """strategy: mean | median | mode | zero"""
    if column not in df.columns:
        return df, ToolResult("fill_nulls", f"Column '{column}' not found."), []

    null_count = int(df[column].isna().sum())
    if null_count == 0:
        return df, ToolResult("fill_nulls", f"No nulls in '{column}', nothing to fill."), []

    s = df[column]
    if strategy == "mean":
        if not pd.api.types.is_numeric_dtype(s):
            return df, ToolResult("fill_nulls", f"Cannot use mean strategy on non-numeric column '{column}'."), []
        fill_value = s.mean()
    elif strategy == "median":
        if not pd.api.types.is_numeric_dtype(s):
            return df, ToolResult("fill_nulls", f"Cannot use median strategy on non-numeric column '{column}'."), []
        fill_value = s.median()
    elif strategy == "mode":
        mode = s.mode()
        fill_value = mode.iloc[0] if len(mode) else None
    elif strategy == "zero":
        fill_value = 0
    else:
        return df, ToolResult("fill_nulls", f"Unknown strategy '{strategy}'. Use: mean | median | mode | zero"), []

    df = df.copy()
    df[column] = df[column].fillna(fill_value)
    log = [CleaningLogEntry(
        operation="fill_nulls",
        column=column,
        rows_affected=null_count,
        rationale=f"Filled {null_count} nulls in '{column}' with {strategy} ({fill_value!r}).",
    )]
    return df, ToolResult("fill_nulls", f"Filled {null_count} nulls in '{column}' with {strategy}={fill_value!r}."), log


def coerce_dtype(
    df: pd.DataFrame, column: str, dtype: str
) -> tuple[pd.DataFrame, ToolResult, list[CleaningLogEntry]]:
    """dtype: numeric | datetime | string"""
    if column not in df.columns:
        return df, ToolResult("coerce_dtype", f"Column '{column}' not found."), []

    original_dtype = str(df[column].dtype)
    df = df.copy()
    try:
        if dtype == "numeric":
            df[column] = pd.to_numeric(df[column], errors="coerce")
        elif dtype == "datetime":
            df[column] = pd.to_datetime(df[column], errors="coerce")
        elif dtype == "string":
            df[column] = df[column].astype(str)
        else:
            return df, ToolResult("coerce_dtype", f"Unknown dtype '{dtype}'. Use: numeric | datetime | string"), []
    except Exception as e:
        return df, ToolResult("coerce_dtype", f"Coercion failed for '{column}': {e}"), []

    new_dtype = str(df[column].dtype)
    log = [CleaningLogEntry(
        operation="coerce_dtype",
        column=column,
        rows_affected=len(df),
        rationale=f"Converted '{column}' from {original_dtype} → {new_dtype}.",
    )]
    return df, ToolResult("coerce_dtype", f"Converted '{column}' from {original_dtype} → {new_dtype}."), log


def flag_outliers(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, ToolResult, list[CleaningLogEntry]]:
    if column not in df.columns:
        return df, ToolResult("flag_outliers", f"Column '{column}' not found."), []
    if not pd.api.types.is_numeric_dtype(df[column]):
        return df, ToolResult("flag_outliers", f"Column '{column}' is not numeric."), []

    s = df[column].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    flag_col = f"{column}_is_outlier"

    df = df.copy()
    df[flag_col] = (df[column] < lower) | (df[column] > upper)
    count = int(df[flag_col].sum())
    log = []
    if count > 0:
        log.append(CleaningLogEntry(
            operation="flag_outliers",
            column=column,
            rows_affected=count,
            rationale=f"Flagged {count} IQR outlier(s) in '{column}' [{lower:.2f}, {upper:.2f}]. Column '{flag_col}' added.",
        ))
    return df, ToolResult("flag_outliers", f"Flagged {count} outlier(s) in '{column}'. Added '{flag_col}' column."), log


def drop_outliers(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, ToolResult, list[CleaningLogEntry]]:
    """Drop IQR outliers. Only call when you've judged them to be errors, not real spikes."""
    if column not in df.columns:
        return df, ToolResult("drop_outliers", f"Column '{column}' not found."), []
    if not pd.api.types.is_numeric_dtype(df[column]):
        return df, ToolResult("drop_outliers", f"Column '{column}' is not numeric."), []

    s = df[column].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    before = len(df)
    df = df[(df[column] >= lower) & (df[column] <= upper)].copy()
    dropped = before - len(df)
    log = [CleaningLogEntry(
        operation="drop_outliers",
        column=column,
        rows_affected=dropped,
        rationale=f"Dropped {dropped} IQR outlier row(s) from '{column}' [{lower:.2f}, {upper:.2f}].",
    )]
    return df, ToolResult("drop_outliers", f"Dropped {dropped} outlier rows from '{column}'."), log


# ---------------------------------------------------------------------------
# Dispatcher — cleaning tools mutate df so it threads through
# ---------------------------------------------------------------------------

def dispatch_cleaning_tool(
    df: pd.DataFrame,
    tool_name: str,
    args: dict,
    run_id: str = "unknown",
) -> tuple[pd.DataFrame, ToolResult, list[CleaningLogEntry]]:
    """
    Returns (updated_df, tool_result, new_log_entries).
    Profile tools don't mutate df — they return the same df back.
    """
    no_log: list[CleaningLogEntry] = []

    if tool_name == "profile_column":
        df, result = profile_column(df, args["column"])
        return df, result, no_log

    if tool_name == "drop_duplicates":
        return drop_duplicates(df)

    if tool_name == "drop_nulls":
        return drop_nulls(df, args["column"])

    if tool_name == "fill_nulls":
        return fill_nulls(df, args["column"], args["strategy"])

    if tool_name == "coerce_dtype":
        return coerce_dtype(df, args["column"], args["dtype"])

    if tool_name == "flag_outliers":
        return flag_outliers(df, args["column"])

    if tool_name == "drop_outliers":
        return drop_outliers(df, args["column"])

    if tool_name == "run_code":
        from analysis_engine.tools.sandbox import run_sandbox_code
        result = run_sandbox_code(df=df, code=args["code"], purpose=args.get("purpose", ""), run_id=run_id, agent="cleaning")
        return df, result, no_log  # sandbox can't mutate df — read-only in cleaning context

    return df, ToolResult(tool_name, f"Unknown tool: '{tool_name}'"), no_log


# ---------------------------------------------------------------------------
# OpenAI tool definitions
# ---------------------------------------------------------------------------

CLEANING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "profile_column",
            "description": "Profile a single column: null count, dtype, value range, outlier count. Call this before deciding what cleaning to apply.",
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
            "name": "drop_duplicates",
            "description": "Remove fully duplicate rows from the dataframe.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "drop_nulls",
            "description": "Drop rows where a specific column is null. Use when null % is low and the column is critical.",
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
            "name": "fill_nulls",
            "description": "Fill nulls in a column with a computed value. Use when dropping would lose too much data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "strategy": {
                        "type": "string",
                        "enum": ["mean", "median", "mode", "zero"],
                        "description": "mean/median for numeric, mode for categorical, zero as a last resort"
                    }
                },
                "required": ["column", "strategy"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "coerce_dtype",
            "description": "Convert a column to the correct type (numeric, datetime, or string).",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "dtype": {
                        "type": "string",
                        "enum": ["numeric", "datetime", "string"]
                    }
                },
                "required": ["column", "dtype"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "flag_outliers",
            "description": "Flag IQR outliers in a numeric column with a boolean column (does NOT drop them). Use when unsure if outliers are errors or real spikes.",
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
            "name": "drop_outliers",
            "description": "Drop IQR outlier rows from a numeric column. Only call when you have judged the outliers to be data errors, not real spikes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"}
                },
                "required": ["column"]
            }
        }
    },
    SANDBOX_TOOL,
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Call when cleaning is complete. Summarize what was done.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "One paragraph describing what cleaning was applied and why."
                    }
                },
                "required": ["summary"]
            }
        }
    }
]