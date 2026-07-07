import json
import uuid
import os
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from plotly.utils import PlotlyJSONEncoder

from analysis_engine.state import PipelineState, Claim
from analysis_engine.llm.client import get_llm
from analysis_engine.registry import RUN_CALLBACKS
from analysis_engine.tools.data_io import load_dataframe
from analysis_engine.tools.analysis_tools import dispatch_tool, EXPLORE_TOOLS
from analysis_engine.tools.base import ToolResult

def _get_charts_dir() -> Path:
    """Resolve lazily so VERUM_CHARTS_DIR set by app.py is always picked up."""
    return Path(os.environ.get("VERUM_CHARTS_DIR", "charts"))

SYSTEM_PROMPT = """You are a data analysis agent with tools for exploring data and generating charts.

STRICT RULES — violations cause the pipeline to fail:
1. px and go are pre-imported globals. NEVER write import statements. Using import will cause an error.
2. print() does not exist. Use result = <value> for ALL output.
3. You MUST call finish() after at most 8 tool calls. Do not keep exploring indefinitely.
4. Generate charts using run_code with px or go directly. Example:
     result = px.bar(df.groupby('region')['revenue'].mean().reset_index(),
                     x='region', y='revenue', title='Revenue by Region')

WORKFLOW:
- Use describe_column, groupby_mean, correlation to discover insights (3-5 calls)
- For each insight, immediately generate a chart with run_code (1 call per chart)
- After 3-5 insights with charts, call finish() — do not delay

Pick chart types by insight:
  comparisons → px.bar()
  distributions → px.histogram()
  correlations → px.scatter()
  time trends → px.line()
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


def _save_chart(spec: dict) -> str:
    charts_dir = _get_charts_dir()
    charts_dir.mkdir(parents=True, exist_ok=True)
    ref = f"chart-{uuid.uuid4()}.json"
    (charts_dir / ref).write_text(json.dumps(spec, cls=PlotlyJSONEncoder))
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
    if event_callback is None:
        event_callback = RUN_CALLBACKS.get(state.run_id)

    if event_callback:
        event_callback("step_started", {"step": "analyzing", "message": "Analyzing data..."})

    if not state.cleaned_refs:
        print("[analysis] no cleaned files, skipping")
        if event_callback:
            event_callback("step_completed", {"step": "analyzing"})
        return {"status": "verifying"}

    all_claims: list[Claim] = []
    all_chart_refs: list[str] = []
    all_chart_specs: dict[str, dict] = {}
    question = state.question
    run_id = state.run_id

    for file_id, cleaned_ref in state.cleaned_refs.items():
        print(f"\n[analysis] file_id={file_id}")
        df = load_dataframe(cleaned_ref)
        profile = _build_profile(df)
        print(f"[analysis] profile built ({df.shape[0]}r x {df.shape[1]}c)")
        print("[analysis] starting explore loop...")

        file_tool_results: list[ToolResult] = []

        def tool_dispatcher(tool_name: str, args: dict) -> ToolResult:
            if event_callback:
                event_callback("tool_called", {
                    "node": "analysis",
                    "tool": tool_name,
                    "args": {k: str(v)[:100] for k, v in args.items()},
                })
            result = dispatch_tool(df, tool_name, args, run_id=run_id)
            file_tool_results.append(result)
            if event_callback:
                event_callback("tool_result", {
                    "node": "analysis",
                    "tool": tool_name,
                    "output": result.output[:200],
                    "has_chart": result.chart_spec is not None,
                })
            return result

        llm = get_llm()
        _, finish_args = llm.explore_loop(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=_make_user_prompt(profile, question),
            tools=EXPLORE_TOOLS,
            tool_dispatcher=tool_dispatcher,
            finish_tool_name="finish",
            max_iterations=20,
        )

        # collect charts from all tool results
        for result in file_tool_results:
            if result.chart_spec:
                ref = _save_chart(result.chart_spec)
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
                    event_callback("claim_generated", {
                        "text": c.text,
                        "value": c.value,
                        "metric": c.metric,
                        "source_columns": c.source_columns,
                    })
        else:
            # loop ended without finish() — synthesise claims from charts we did collect
            print("[analysis] WARNING: explore loop ended without finish() — generating claims from charts")
            if event_callback:
                event_callback("tool_result", {
                    "node": "analysis",
                    "tool": "warning",
                    "output": "Explore loop hit iteration limit. Claims derived from collected charts.",
                })
            # emit at least one claim per chart so verification/synthesis has something to work with
            for ref in all_chart_refs:
                import uuid as _uuid
                from analysis_engine.state import Claim as _Claim
                c = _Claim(
                    id=str(_uuid.uuid4()),
                    text=f"Chart generated: {ref}",
                    metric="chart",
                    value=0.0,
                    source_query="run_code (plotly)",
                    source_columns=[],
                    verification_status="unverifiable",
                )
                all_claims.append(c)
                if event_callback:
                    event_callback("claim_generated", {"text": c.text, "value": 0.0, "metric": "chart", "source_columns": []})

    if event_callback:
        event_callback("step_completed", {"step": "analyzing"})

    return {
        "claims": all_claims,
        "chart_refs": all_chart_refs,
        "chart_specs": all_chart_specs,
        "status": "verifying",
    }