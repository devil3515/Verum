from dataclasses import dataclass, field

import pandas as pd

@dataclass
class MetricResult:
    value: float
    source_query: str
    source_columns: list[str] = field(default_factory=list)


def compute_summary_stats(df: pd.DataFrame) -> dict[str, MetricResult]:
    results = {}
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        results[f"{col}_mean"] = MetricResult(
            value=float(series.mean()),
            source_query=f"df['{col}'].mean()",
            source_columns=[col],
        )
        results[f"{col}_std"] = MetricResult(
            value=float(series.std()) if len(series) > 1 else 0.0,
            source_query=f"df['{col}'].std()",
            source_columns=[col],
        )
        results[f"{col}_min"] = MetricResult(
            value=float(series.min()),
            source_query=f"df['{col}'].min()",
            source_columns=[col],
        )
        results[f"{col}_max"] = MetricResult(
            value=float(series.max()),
            source_query=f"df['{col}'].max()",
            source_columns=[col],
        )
    return results

def compute_correlation_matrix(df: pd.DataFrame, min_abs_corr: float = 0.5) -> dict[str, MetricResult]:
    results = {}
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return results

    corr_matrix = numeric_df.corr()
    seen = set()

    corr_matrix = numeric_df.corr()
    for col1 in corr_matrix.columns:
        for col2 in corr_matrix.columns:
            if col1 == col2 or (col1, col2) in seen:
                continue
            seen.add((col1, col2))
            corr_val = corr_matrix.loc[col1, col2]
            if pd.isna(corr_val) or abs(corr_val) < min_abs_corr:
                continue
            key = f"corr_{col1}_{col2}"
            results[key] = MetricResult(
                value=float(corr_val),
                source_query=f"df['{col1}'].corr(df['{col2}'])",
                source_columns=[col1, col2],
            )
    return results


def compute_group_comparisons(df: pd.DataFrame) -> dict[str, MetricResult]:
    results = {}
    categorical_cols = [
        c for c in df.columns
        if (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]))
        and df[c].nunique() <= 20 and df[c].nunique() > 1
    ]
    numeric_cols = df.select_dtypes(include="number").columns

    for cat_col in categorical_cols:
        for num_col in numeric_cols:
            grouped = df.groupby(cat_col)[num_col].mean().dropna()
            if len(grouped) < 2:
                continue
            max_group = grouped.idxmax()
            min_group = grouped.idxmin()
            max_value = grouped[max_group]
            min_value = grouped[min_group]
            if min_value == 0:
                continue
            pct_diff = ((max_value - min_value) / abs(min_value)) * 100

            key = f"{num_col}_by_{cat_col}_max_group_diff_pct"
            results[key] = MetricResult(
                value=float(pct_diff),
                source_query=(
                    f"df.groupby('{cat_col}')['{num_col}'].mean() -> "
                    f"'{max_group}' ({max_value:.2f}) vs '{min_group}' ({min_value:.2f})"
                ),
                source_columns=[cat_col, num_col],
            )
    return results


def run_full_analysis(df: pd.DataFrame) -> dict[str, MetricResult]:
    results = {}
    results.update(compute_summary_stats(df))
    results.update(compute_correlation_matrix(df))
    results.update(compute_group_comparisons(df))
    return results