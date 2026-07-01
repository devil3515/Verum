import uuid
import json
import pandas as pd
from dataclasses import dataclass, field

@dataclass
class ToolResult:
    tool_name: str
    output: str
    chart_spec: dict | None = None
    chart_ref: str = ""

    def __post_init__(self):
        if self.chart_spec and not self.chart_ref:
            self.chart_ref = f"chart-{uuid.uuid4()}.json"


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
            "mean": round(float(s.mean()), 4) if s.count() else None,
            "std": round(float(s.std()), 4) if s.count() > 1 else None,
            "min": round(float(s.min()), 4) if s.count() else None,
            "25%": round(float(s.quantile(0.25)), 4) if s.count() else None,
            "median": round(float(s.median()), 4) if s.count() else None,
            "75%": round(float(s.quantile(0.75)), 4) if s.count() else None,
            "max": round(float(s.max()), 4) if s.count() else None,
        }
    else:
        stats = {
            "dtype": str(s.dtype),
            "count": int(s.count()),
            "null_count": int(s.isna().sum()),
            "unique_values": int(s.nunique()),
            "top_values": s.value_counts().head(10).to_dict(),
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
        "col1": col1,
        "col2": col2,
        "pearson_correlation": round(float(val), 4),
        "interpretation": (
            "strong positive" if val > 0.7 else
            "moderate positive" if val > 0.4 else
            "weak positive" if val > 0.1 else
            "strong negative" if val < -0.7 else
            "moderate negative" if val < -0.4 else
            "weak negative" if val < -0.1 else
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


def plot_histogram(df: pd.DataFrame, column: str) -> ToolResult:
    if column not in df.columns:
        return ToolResult("plot_histogram", f"Column '{column}' not found.")

    values = df[column].dropna().tolist()
    spec = {
        "type": "histogram",
        "data": {"x": values},
        "layout": {
            "title": f"Distribution of {column}",
            "xaxis": {"title": column},
            "yaxis": {"title": "Count"},
        }
    }
    return ToolResult(
        "plot_histogram",
        f"Histogram for '{column}' generated ({len(values)} values).",
        chart_spec=spec,
    )


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str) -> ToolResult:
    for c in [x_col, y_col]:
        if c not in df.columns:
            return ToolResult("plot_scatter", f"Column '{c}' not found.")

    pairs = df[[x_col, y_col]].dropna()
    spec = {
        "type": "scatter",
        "data": {
            "x": pairs[x_col].tolist(),
            "y": pairs[y_col].tolist(),
        },
        "layout": {
            "title": f"{y_col} vs {x_col}",
            "xaxis": {"title": x_col},
            "yaxis": {"title": y_col},
        }
    }
    return ToolResult(
        "plot_scatter",
        f"Scatter plot of '{y_col}' vs '{x_col}' generated ({len(pairs)} points).",
        chart_spec=spec,
    )


def plot_grouped_bar(df: pd.DataFrame, value_col: str, group_col: str) -> ToolResult:
    for c in [value_col, group_col]:
        if c not in df.columns:
            return ToolResult("plot_grouped_bar", f"Column '{c}' not found.")

    grouped = df.groupby(group_col)[value_col].mean().reset_index()
    spec = {
        "type": "bar",
        "data": {
            "x": grouped[group_col].tolist(),
            "y": grouped[value_col].round(2).tolist(),
        },
        "layout": {
            "title": f"Mean {value_col} by {group_col}",
            "xaxis": {"title": group_col},
            "yaxis": {"title": f"Mean {value_col}"},
        }
    }
    return ToolResult(
        "plot_grouped_bar",
        f"Grouped bar chart of mean '{value_col}' by '{group_col}' generated.",
        chart_spec=spec,
    )


# ---------------------------------------------------------------------------
# Tool dispatcher — maps tool name + args → ToolResult
# ---------------------------------------------------------------------------

def dispatch_tool(df: pd.DataFrame, tool_name: str, args: dict) -> ToolResult:
    dispatch = {
        "describe_column": lambda: describe_column(df, args["column"]),
        "correlation": lambda: correlation(df, args["col1"], args["col2"]),
        "groupby_mean": lambda: groupby_mean(df, group_col=args["group_col"], value_col=args["value_col"]),
        "plot_histogram": lambda: plot_histogram(df, args["column"]),
        "plot_scatter": lambda: plot_scatter(df, args["x_col"], args["y_col"]),
        "plot_grouped_bar": lambda: plot_grouped_bar(df, args["value_col"], args["group_col"]),
    }
    fn = dispatch.get(tool_name)
    if not fn:
        return ToolResult(tool_name, f"Unknown tool: '{tool_name}'")
    try:
        return fn()
    except Exception as e:
        return ToolResult(tool_name, f"Tool error: {e}")


# ---------------------------------------------------------------------------
# OpenAI tool definitions — what the LLM sees
# ---------------------------------------------------------------------------

EXPLORE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "describe_column",
            "description": "Get detailed statistics for a single column (dtype, nulls, mean, std, min, max, top values for categoricals).",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Column name to describe"}
                },
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
                "properties": {
                    "col1": {"type": "string"},
                    "col2": {"type": "string"}
                },
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
                    "group_col": {"type": "string", "description": "Categorical column to group by"}
                },
                "required": ["value_col", "group_col"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_histogram",
            "description": "Generate a histogram for a numeric column to understand its distribution.",
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
            "name": "plot_scatter",
            "description": "Generate a scatter plot between two numeric columns to visualize correlation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x_col": {"type": "string"},
                    "y_col": {"type": "string"}
                },
                "required": ["x_col", "y_col"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_grouped_bar",
            "description": "Generate a grouped bar chart showing mean of a numeric column per category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value_col": {"type": "string"},
                    "group_col": {"type": "string"}
                },
                "required": ["value_col", "group_col"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Call this when you have gathered enough evidence. Provide your final insights.",
            "parameters": {
                "type": "object",
                "properties": {
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "One-sentence plain-language insight"
                                },
                                "metric": {
                                    "type": "string",
                                    "description": "Short metric key, e.g. revenue_by_region"
                                },
                                "value": {
                                    "type": "number",
                                    "description": "The single most representative number for this claim"
                                },
                                "source_query": {
                                    "type": "string",
                                    "description": "Which tool call produced this (e.g. groupby_mean(revenue, region))"
                                },
                                "source_columns": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            },
                            "required": ["text", "metric", "value", "source_query", "source_columns"]
                        }
                    }
                },
                "required": ["claims"]
            }
        }
    }
]
