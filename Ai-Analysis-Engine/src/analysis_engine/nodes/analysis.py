import json
import uuid
import pandas as pd

from analysis_engine.state import PipelineState, Claim
from analysis_engine.llm.client import get_llm
from analysis_engine.tools.data_io import load_dataframe
from analysis_engine.tools.analysis_tools import (
    dispatch_tool,
    EXPLORE_TOOLS,
    ToolResult,
)


# ---------------------------------------------------------------------------
# Phase 1 — Profile
# ---------------------------------------------------------------------------

def _build_profile(df: pd.DataFrame) -> str:
    """
    Cheap deterministic pass. Returns a string the LLM reads as context
    before it decides which tools to call.
    """
    lines = []
    lines.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    lines.append("")
    lines.append("Columns:")
    for col in df.columns:
        dtype = str(df[col].dtype)
        nulls = int(df[col].isna().sum())
        if pd.api.types.is_numeric_dtype(df[col]):
            col_min = round(float(df[col].min()), 2) if df[col].count() else "N/A"
            col_max = round(float(df[col].max()), 2) if df[col].count() else "N/A"
            lines.append(f"  `{col}` ({dtype}) — nulls: {nulls}, range: [{col_min}, {col_max}]")
        else:
            unique = df[col].nunique()
            top = df[col].value_counts().head(3).index.tolist()
            lines.append(f"  `{col}` ({dtype}) — nulls: {nulls}, unique: {unique}, top: {top}")

    lines.append("")
    lines.append("Sample rows (first 3):")
    lines.append(df.head(3).to_string(index=False))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 2 — Explore loop
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a data analysis agent. You have access to tools
that let you explore a dataset. Your job is to:

1. Use the tools to investigate the data thoroughly.
2. Focus on findings that are genuinely interesting: strong correlations,
   notable group differences, outliers, unexpected distributions.
3. For every interesting finding, also generate an appropriate chart using
   plot_histogram, plot_scatter, or plot_grouped_bar.
4. When you have gathered enough evidence (typically 3-6 tool calls),
   call finish() with your claims. Each claim must reference a value you
   actually observed from a tool result — never invent numbers.

Be selective. 3 strong insights with charts beat 10 weak ones without.
"""


def _make_user_prompt(profile: str, question: str | None) -> str:
    focus = (
        f"\nThe user's specific question is: \"{question}\"\nFocus your "
        f"exploration on answering this question, but surface other notable "
        f"findings too.\n"
        if question else ""
    )
    return f"""Here is the dataset profile:

{profile}
{focus}
Start exploring. Use the tools to investigate, generate charts for your
findings, then call finish() with your claims."""


# ---------------------------------------------------------------------------
# Phase 3 — Build Claim objects from finish() output
# ---------------------------------------------------------------------------

def _build_claims(finish_args: dict, chart_refs: list[str]) -> list[Claim]:
    claims = []
    raw_claims = finish_args.get("claims", [])
    for rc in raw_claims:
        try:
            claim = Claim(
                id=str(uuid.uuid4()),
                text=rc["text"],
                metric=rc["metric"],
                value=float(rc["value"]),
                source_query=rc["source_query"],
                source_columns=rc.get("source_columns", []),
            )
            claims.append(claim)
        except Exception as e:
            print(f"[analysis] skipping malformed claim: {e} — {rc}")
    return claims


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def analysis_node(state: PipelineState) -> dict:
    if not state.cleaned_refs:
        print("[analysis] no cleaned files available, skipping")
        return {"status": "verifying"}

    all_claims: list[Claim] = []
    all_chart_refs: list[str] = []
    all_chart_specs: dict[str, dict] = {}  # ref -> spec, for persistence later

    # user question from state (planner could put this in state eventually,
    # for now we read it if it's there, otherwise None)
    question = getattr(state, "question", None)

    for file_id, cleaned_ref in state.cleaned_refs.items():
        print(f"\n[analysis] file_id={file_id}")

        df = load_dataframe(cleaned_ref)

        # ── Phase 1: Profile ─────────────────────────────────────────────
        profile = _build_profile(df)
        print(f"[analysis] profile built ({df.shape[0]}r x {df.shape[1]}c)")

        # ── Phase 2: Explore loop ─────────────────────────────────────────
        print("[analysis] starting explore loop...")

        # track tool results so we can collect chart specs
        file_tool_results: list[ToolResult] = []

        def tool_dispatcher_with_tracking(tool_name: str, args: dict) -> ToolResult:
            result = dispatch_tool(df, tool_name, args)
            file_tool_results.append(result)
            return result

        llm = get_llm()
        _, finish_args = llm.explore_loop(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=_make_user_prompt(profile, question),
            tools=EXPLORE_TOOLS,
            tool_dispatcher=tool_dispatcher_with_tracking,
            finish_tool_name="finish",
            max_iterations=10,
        )

        # ── collect chart specs from all tool results ─────────────────────
        for result in file_tool_results:
            if result.chart_spec and result.chart_ref:
                all_chart_refs.append(result.chart_ref)
                all_chart_specs[result.chart_ref] = result.chart_spec
                print(f"[analysis] chart generated: {result.chart_ref}")

        # ── Phase 3: Build claims ─────────────────────────────────────────
        if finish_args:
            claims = _build_claims(finish_args, all_chart_refs)
            all_claims.extend(claims)
            for c in claims:
                print(f"[analysis] claim: {c.text} (value={c.value})")
        else:
            print("[analysis] WARNING: explore loop ended without finish() call")

    return {
        "claims": all_claims,
        "chart_refs": all_chart_refs,
        "status": "verifying",
    }