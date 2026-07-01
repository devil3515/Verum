import os
import pandas as pd

def load_dataframe(ref: str) -> pd.DataFrame:
    if ref.endswith(".csv"):
        return pd.read_csv(ref)
    if ref.endswith(".parquet"):
        return pd.read_parquet(ref)
    raise ValueError(f"Unsupported file type for ref: {ref}")

def save_dataframe(df: pd.DataFrame, original_ref: str, suffix: str = ".cleaned") -> str:
    base, ext = os.path.splitext(original_ref)
    out_ref = f"{base}{suffix}{ext}"
    if ext == ".csv":
        df.to_csv(out_ref, index=False)
    elif ext == ".parquet":
        df.to_parquet(out_ref, index=False)
    else:
        raise ValueError(f"Unsupported file type for ref: {original_ref}")
    return out_ref