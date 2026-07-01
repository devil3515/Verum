import pandas as pd

from analysis_engine.state import CleaningLogEntry

def drop_null_rows(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, list[CleaningLogEntry]]:
    before = len(df)
    cleaned = df.dropna(subset=[column])
    dropped = before - len(cleaned)
    log = []
    if dropped > 0:
        log.append(CleaningLogEntry(
            operation="dropped_nulls",
            column=column,
            rows_affected=dropped,
            rationale=f"{dropped} row(s) had a null '{column}' value and were removed.",
        ))
    return cleaned, log


def coerce_dtype(df: pd.DataFrame, column: str, dtype: str) -> tuple[pd.DataFrame, list[CleaningLogEntry]]:
    log = []
    if column not in df.columns:
        return df, log

    original_dtype = str(df[column].dtype)
    try:
        if dtype == "datetime":
            df[column] = pd.to_datetime(df[column], errors="coerce")
        elif dtype == "numeric":
            df[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            df[column] = df[column].astype(dtype)
    except Exception as e:
        log.append(CleaningLogEntry(
            operation="coercion_failed",
            column=column,
            rows_affected=0,
            rationale=f"Could not coerce '{column}' from {original_dtype} to {dtype}: {e}",
        ))
        return df, log

    new_dtype = str(df[column].dtype)
    if new_dtype != original_dtype:
        log.append(CleaningLogEntry(
            operation="coerced_dtype",
            column=column,
            rows_affected=len(df),
            rationale=f"Converted '{column}' from {original_dtype} to {new_dtype}.",
        ))
    return df, log


def flag_outliers_iqr(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, list[CleaningLogEntry]]:
    """
    Flags (does not drop) outliers using the IQR method - adds a
    boolean column '{column}_is_outlier'. Dropping outliers automatically
    is risky (could be a real spike) so we flag for now; an LLM-assisted
    decision on whether to drop can be layered in later if needed.
    """
    log = []
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return df, log

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outlier_mask = (df[column] < lower) | (df[column] > upper)
    df[f"{column}_is_outlier"] = outlier_mask

    count = int(outlier_mask.sum())
    if count > 0:
        log.append(CleaningLogEntry(
            operation="flagged_outliers",
            column=column,
            rows_affected=count,
            rationale=f"{count} row(s) fall outside [{lower:.2f}, {upper:.2f}] "
                      f"(IQR method) and were flagged, not dropped.",
        ))
    return df, log


def drop_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, list[CleaningLogEntry]]:
    before = len(df)
    cleaned = df.drop_duplicates()
    dropped = before - len(cleaned)
    log = []
    if dropped > 0:
        log.append(CleaningLogEntry(
            operation="dropped_duplicates",
            column="(all columns)",
            rows_affected=dropped,
            rationale=f"{dropped} fully duplicate row(s) were removed.",
        ))
    return cleaned, log
