"""
Phase 3 (revised) — Cleaning Agent: profile → tool loop → finish.

Same pattern as the analysis agent. The LLM profiles columns, decides
which cleaning operations to apply, calls them, sees the result, then
calls finish() when done.

Key difference from analysis tools: cleaning tools MUTATE the dataframe.
The dispatcher returns (updated_df, result, log_entries) and df is threaded
through each call so mutations accumulate correctly.
"""
import pandas as pd

from analysis_engine.state import PipelineState, CleaningLogEntry
from analysis_engine.llm.client import get_llm
from analysis_engine.tools.data_io import load_dataframe, save_dataframe
from analysis_engine.tools.cleaning_tools import dispatch_cleaning_tool, CLEANING_TOOLS
from analysis_engine.tools.base import ToolResult


SYSTEM_PROMPT = """You are a data cleaning agent. You have access to tools
that let you profile and clean a dataframe.

Your job:
1. Profile the columns to understand what cleaning is needed.
2. Apply cleaning operations based on what you find — don't apply them blindly.
3. Make judgment calls: if a column has 2% nulls in a key metric, drop them.
   If it has 40% nulls, fill with median instead of losing that data.
4. Flag outliers rather than dropping them unless you're confident they're errors.
5. When cleaning is complete, call finish() with a summary of what you did.

Be conservative — only apply operations that are clearly justified by the data.
"""


def _build_profile(df: pd.DataFrame) -> str:
    lines = [f"Shape: {df.shape[0]} rows x {df.shape[1]} columns", "", "Columns:"]
    for col in df.columns:
        dtype = str(df[col].dtype)
        nulls = int(df[col].isna().sum())
        null_pct = round(df[col].isna().mean() * 100, 1)
        if pd.api.types.is_numeric_dtype(df[col]):
            col_min = round(float(df[col].min()), 2) if df[col].count() else "N/A"
            col_max = round(float(df[col].max()), 2) if df[col].count() else "N/A"
            lines.append(f"  `{col}` ({dtype}) — nulls: {nulls} ({null_pct}%), range: [{col_min}, {col_max}]")
        else:
            unique = df[col].nunique()
            lines.append(f"  `{col}` ({dtype}) — nulls: {nulls} ({null_pct}%), unique: {unique}")
    dupes = int(df.duplicated().sum())
    lines.append(f"\nDuplicate rows: {dupes}")
    lines.append("\nSample rows (first 3):")
    lines.append(df.head(3).to_string(index=False))
    return "\n".join(lines)


def cleaning_node(state: PipelineState) -> dict:
    if not state.files:
        print("[cleaning] no files in state, skipping")
        return {"status": "analyzing"}

    all_log_entries: list[CleaningLogEntry] = []
    cleaned_refs: dict[str, str] = {}
    question = getattr(state, "question", None)
    run_id = state.run_id

    for f in state.files:
        print(f"\n[cleaning] file_id={f.file_id} ref={f.ref}")
        df = load_dataframe(f.ref)

        profile = _build_profile(df)
        print(f"[cleaning] profile built ({df.shape[0]}r x {df.shape[1]}c)")

        # thread df + log through tool calls — cleaning mutates state
        current_df = df
        file_log: list[CleaningLogEntry] = []

        def tool_dispatcher(tool_name: str, args: dict) -> ToolResult:
            nonlocal current_df, file_log
            updated_df, result, new_logs = dispatch_cleaning_tool(
                current_df, tool_name, args, run_id=run_id
            )
            current_df = updated_df
            file_log.extend(new_logs)
            return result

        focus = (
            f'\nThe user\'s question is: "{question}". '
            f'Keep it in mind when deciding what to clean — '
            f'columns relevant to that question should be cleaned carefully.\n'
            if question else ""
        )

        print("[cleaning] starting tool loop...")
        llm = get_llm()
        _, finish_args = llm.explore_loop(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=f"Here is the dataset profile:\n\n{profile}\n{focus}\nStart profiling and cleaning.",
            tools=CLEANING_TOOLS,
            tool_dispatcher=tool_dispatcher,
            finish_tool_name="finish",
            max_iterations=15,
        )

        if finish_args:
            print(f"[cleaning] done. summary: {finish_args.get('summary', '')[:120]}")
        else:
            print("[cleaning] WARNING: loop ended without finish() call")

        out_ref = save_dataframe(current_df, f.ref)
        cleaned_refs[f.file_id] = out_ref
        all_log_entries.extend(file_log)
        print(f"[cleaning] wrote {out_ref} ({len(current_df)} rows, {len(file_log)} operations)")

    return {
        "cleaning_log": all_log_entries,
        "cleaned_refs": cleaned_refs,
        "status": "analyzing",
    }