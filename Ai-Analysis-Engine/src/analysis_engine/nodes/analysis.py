import json
import uuid
import os
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from analysis_engine.state import PipelineState, Claim
from analysis_engine.llm.client import get_llm
from analysis_engine.tools.data_io import load_dataframe
from analysis_engine.tools.analysis_tools import dispatch_tool, EXPLORE_TOOLS
from analysis_engine.tools.base import ToolResult

# where chart JSON files are saved — backend serves from here
CHARTS_DIR = Path(os.environ.get("VERUM_CHARTS_DIR", "charts"))

# global registry: run_id → event_callback
# The web server registers here before invoking the graph so that
# analysis_node can push live events back without any import cycles.
RUN_CALLBACKS: dict[str, Callable[[str, dict], None]] = {}


SYSTEM_PROMPT = """You are a data analysis agent with tools for exploring data and generating charts.

Rules:
- Use the dedicated chart tools first: plot_grouped_bar, plot_scatter, plot_histogram.
- Generate at least 4 charts covering different angles — distributions (histogram),
  category comparisons (bar), and correlations (scatter).
- px and go are pre-imported in run_code. Do NOT import them — just use them directly.
- Use result = fig to capture a chart from run_code (not print).
- print() works for debug output, but set result = <value> to return structured data.
- Pick the right chart type: bar for comparisons, histogram for distributions,
  scatter for correlations, line for time trends.
- Always set a descriptive title and axis labels.
- For every meaningful finding, generate a chart immediately after discovering it.
- Call finish() when you have 3-5 strong insights with charts.

Example chart via run_code:
  grouped = df.groupby('region')['revenue'].mean().reset_index()
  result = px.bar(grouped, x='region', y='revenue',
                  title='Mean Revenue by Region',
                  labels={'revenue': 'Mean Revenue ($)'})
"""


def _build_profile(df: pd.DataFrame) -> str:
    lines = [f"Shape: {df.shape[0]} rows x {df.shape[1]} columns", "", "Columns:"]
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
    lines.append(f"\nSample rows (first 3):\n{df.head(3).to_string(index=False)}")
    return "\n".join(lines)


def _make_user_prompt(profile: str, question: Optional[str]) -> str:
    focus = (
        f'\nUser question: "{question}"\nFocus on answering this, but surface other notable findings too.\n'
        if question else ""
    )
    return f"Dataset profile:\n\n{profile}\n{focus}\nStart exploring. Generate charts for every finding. Call finish() when done."


def _save_chart(spec: dict, charts_dir: Path) -> str:
    charts_dir.mkdir(parents=True, exist_ok=True)
    ref = f"chart-{uuid.uuid4()}.json"
    (charts_dir / ref).write_text(json.dumps(spec))
    return ref


def _build_claims(finish_args: dict) -> list[Claim]:
    claims = []
    for rc in finish_args.get("claims", []):
        try:
            claims.append(Claim(
                id=str(uuid.uuid4()),
                text=rc["text"],
                metric=rc["metric"],
                value=float(rc["value"]),
                source_query=rc["source_query"],
                source_columns=rc.get("source_columns", []),
            ))
        except Exception as e:
            print(f"[analysis] skipping malformed claim: {e}")
    return claims


def analysis_node(
    state: PipelineState,
    event_callback: Optional[Callable[[str, dict], None]] = None,
) -> dict:
    # If no callback supplied directly, check the global registry
    # (populated by the web server before invoking the graph)
    if event_callback is None:
        event_callback = RUN_CALLBACKS.get(state.run_id)

    if not state.cleaned_refs:
        print("[analysis] no cleaned files, skipping")
        return {"status": "verifying"}

    all_claims: list[Claim] = []
    all_chart_refs: list[str] = []
    all_chart_specs: dict[str, dict] = {}
    question = state.question
    run_id = state.run_id
    charts_dir = CHARTS_DIR

    for file_id, cleaned_ref in state.cleaned_refs.items():
        print(f"\n[analysis] file_id={file_id}")
        df = load_dataframe(cleaned_ref)
        profile = _build_profile(df)
        print(f"[analysis] profile built ({df.shape[0]}r x {df.shape[1]}c)")
        print("[analysis] starting explore loop...")

        file_tool_results: list[ToolResult] = []

        def tool_dispatcher(tool_name: str, args: dict) -> ToolResult:
            if event_callback:
                event_callback("tool_called", {"node": "analysis", "tool": tool_name, "args": {k: str(v)[:100] for k, v in args.items()}})
            result = dispatch_tool(df, tool_name, args, run_id=run_id)
            file_tool_results.append(result)
            if event_callback:
                event_callback("tool_result", {"node": "analysis", "tool": tool_name, "output": result.output[:200], "has_chart": result.chart_spec is not None})
            return result

        llm = get_llm()
        _, finish_args = llm.explore_loop(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=_make_user_prompt(profile, question),
            tools=EXPLORE_TOOLS,
            tool_dispatcher=tool_dispatcher,
            finish_tool_name="finish",
            max_iterations=12,
        )

        # save charts from tool results
        for result in file_tool_results:
            if result.chart_spec:
                ref = _save_chart(result.chart_spec, charts_dir)
                all_chart_refs.append(ref)
                all_chart_specs[ref] = result.chart_spec
                print(f"[analysis] chart saved: {ref}")
                if event_callback:
                    event_callback("chart_generated", {"ref": ref})

        if finish_args:
            claims = _build_claims(finish_args)
            all_claims.extend(claims)
            for c in claims:
                print(f"[analysis] claim: {c.text[:80]} (value={c.value})")
                if event_callback:
                    event_callback("claim_generated", {"text": c.text, "value": c.value, "metric": c.metric})
        else:
            print("[analysis] WARNING: explore loop ended without finish() call")

    return {
        "claims": all_claims,
        "chart_refs": all_chart_refs,
        "chart_specs": all_chart_specs,
        "status": "verifying",
    }