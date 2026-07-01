import pandas as pd

from analysis_engine.state import PipelineState
from analysis_engine.tools.data_io import load_dataframe, save_dataframe
from analysis_engine.tools.cleaning_ops import (
    drop_duplicate_rows,
    coerce_dtype,
    flag_outliers_iqr,
)


def _infer_numeric_columns(df: pd.DataFrame) -> list[str]:
    candidates = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            candidates.append(col)
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        non_null_ratio = coerced.notna().mean() if len(df) else 0
        if non_null_ratio > 0.9:
            candidates.append(col)
    return candidates

def cleaning_node(state: PipelineState) -> dict:
    if not state.files:
        print("[cleaning] no files in state, skipping")
        return {"status": "analyzing"}

    all_log_entries = []
    cleaned_refs = {}

    for f in state.files:
        print(f"[cleaning] processing file_id={f.file_id} ref={f.ref}")
        df = load_dataframe(f.ref)

        df, log = drop_duplicate_rows(df)
        all_log_entries.extend(log)

        numeric_cols = _infer_numeric_columns(df)
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df, log = coerce_dtype(df, col, "numeric")
                all_log_entries.extend(log)

        for col in numeric_cols:
            df, log = flag_outliers_iqr(df, col)
            all_log_entries.extend(log)

        out_ref = save_dataframe(df, f.ref)
        cleaned_refs[f.file_id] = out_ref
        print(f"[cleaning] wrote cleaned file to {out_ref} ({len(df)} rows)")

    return {
        "cleaning_log": all_log_entries,
        "cleaned_refs": cleaned_refs,
        "status": "analyzing",
    }
