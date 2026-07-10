"""
Phase 7 — Synthesis Agent.

Single LLM call (not a tool loop — there's nothing to iterate over).
Takes the full pipeline output and writes a structured narrative report:

  - Executive summary (direct answer to user's question)
  - Data quality notes (what cleaning found)
  - Key findings (only confirmed/unverifiable claims — contradicted ones
    get their own flagged section so they're visible, not hidden)
  - Charts referenced by finding
  - Caveats and limitations

The report is markdown. The frontend renders it.
"""
from typing import Callable, Optional

from pydantic import BaseModel, Field

from analysis_engine.state import PipelineState, Claim
from analysis_engine.llm.client import get_llm, call_structured
from analysis_engine.registry import RUN_CALLBACKS


class SynthesisOutput(BaseModel):
    executive_summary: str = Field(
        description="2-3 sentence direct answer to the user's question, or a top-level summary if no question was given."
    )
    data_quality_notes: str = Field(
        description="1-2 sentences summarizing what the cleaning agent found and fixed. Be specific about row counts and operations."
    )
    findings: list[dict] = Field(
        description=(
            "Ordered list of key findings. Each item has: "
            "'heading' (short title), "
            "'body' (2-3 sentence narrative explaining the finding and its significance), "
            "'chart_ref' (the chart filename that supports this finding, or empty string if none), "
            "'status' (confirmed/unverifiable/contradicted)."
        )
    )
    contradicted_claims: list[str] = Field(
        default_factory=list,
        description="List of claim texts that were contradicted by verification. Each should note what the contradiction was."
    )
    caveats: str = Field(
        description="1-2 sentences about limitations: data coverage, outliers kept, unverifiable claims, etc."
    )


def _build_prompt(state: PipelineState) -> str:
    # cleaning summary
    cleaning_lines = []
    for entry in state.cleaning_log:
        cleaning_lines.append(f"  - {entry.operation} on '{entry.column}': {entry.rationale}")
    cleaning_block = "\n".join(cleaning_lines) if cleaning_lines else "  - No cleaning operations applied."

    # claims grouped by status
    confirmed    = [c for c in state.claims if c.verification_status == "confirmed"]
    unverifiable = [c for c in state.claims if c.verification_status == "unverifiable"]
    contradicted = [c for c in state.claims if c.verification_status == "contradicted"]

    def fmt_claim(c: Claim) -> str:
        return (
            f"  text: {c.text}\n"
            f"  value: {c.value}\n"
            f"  status: {c.verification_status}\n"
            f"  confidence: {c.confidence}\n"
            f"  chart_ref: {c.id}\n"  # used as a placeholder — frontend matches by position
        )

    claims_block = ""
    if confirmed:
        claims_block += "CONFIRMED CLAIMS:\n" + "\n".join(fmt_claim(c) for c in confirmed) + "\n"
    if unverifiable:
        claims_block += "UNVERIFIABLE CLAIMS (show but flag):\n" + "\n".join(fmt_claim(c) for c in unverifiable) + "\n"
    if contradicted:
        claims_block += "CONTRADICTED CLAIMS (must be disclosed):\n" + "\n".join(fmt_claim(c) for c in contradicted) + "\n"

    # chart refs available
    charts_block = "\n".join(state.chart_refs) if state.chart_refs else "(no charts)"

    question_block = (
        f"User's question: {state.question}"
        if state.question
        else "No specific question — provide a general analysis summary."
    )

    return f"""You are writing the final analysis report.

{question_block}

CLEANING SUMMARY:
{cleaning_block}

VERIFIED CLAIMS:
{claims_block}

AVAILABLE CHART FILES:
{charts_block}

Write a structured report with:
1. executive_summary: 2-3 sentences directly answering the user's question
2. data_quality_notes: what cleaning found and fixed
3. findings: one entry per confirmed/unverifiable claim. For each finding,
   assign the most relevant chart_ref from the available chart files list above.
   If no chart is relevant, use an empty string.
4. contradicted_claims: list the contradicted claims with what went wrong
5. caveats: limitations of this analysis

Be specific with numbers. Do not invent any values not shown above.
"""


def _render_markdown(output: SynthesisOutput, state: PipelineState) -> str:
    """Convert SynthesisOutput to a clean markdown string the frontend renders."""
    sections = []

    if state.question:
        sections.append(f"# {state.question}\n")
    else:
        sections.append("# Analysis Report\n")

    sections.append(f"## Executive Summary\n{output.executive_summary}\n")

    sections.append(f"## Data Quality\n{output.data_quality_notes}\n")

    if output.findings:
        sections.append("## Key Findings\n")
        for f in output.findings:
            status_icon = {"confirmed": "✅", "unverifiable": "⚠️", "contradicted": "❌"}.get(
                f.get("status", "unverifiable"), "⚠️"
            )
            sections.append(f"### {status_icon} {f.get('heading', 'Finding')}")
            sections.append(f.get("body", ""))
            chart_ref = f.get("chart_ref", "")
            if chart_ref and chart_ref in state.chart_refs:
                sections.append(f"\n📊 *Chart: {chart_ref}*")
            sections.append("")

    if output.contradicted_claims:
        sections.append("## ⚠️ Contradicted Claims\n")
        sections.append("The following claims were generated but could not be verified against the data:\n")
        for c in output.contradicted_claims:
            sections.append(f"- {c}")
        sections.append("")

    sections.append(f"## Caveats\n{output.caveats}\n")

    return "\n".join(sections)


def synthesis_node(
    state: PipelineState,
    event_callback: Optional[Callable[[str, dict], None]] = None,
) -> dict:
    if event_callback is None:
        event_callback = RUN_CALLBACKS.get(state.run_id)

    if event_callback:
        event_callback("step_started", {
            "step": "report",
            "message": "Writing final report..."
        })

    print(f"[synthesis] assembling report: {len(state.claims)} claim(s), {len(state.chart_refs)} chart(s)")

    if not state.claims and not state.cleaning_log:
        # nothing to synthesize
        report = "# Analysis Report\n\n_No findings were generated._"
        if event_callback:
            event_callback("step_completed", {"step": "report"})
        return {"report": report, "status": "done"}

    try:
        llm = get_llm()
        prompt = _build_prompt(state)
        output: SynthesisOutput = call_structured(llm, prompt, SynthesisOutput)
        report = _render_markdown(output, state)
        print(f"[synthesis] report generated ({len(report)} chars)")

    except Exception as e:
        # fallback to simple bullet-point report if LLM call fails
        print(f"[synthesis] LLM call failed ({e}), falling back to simple report")
        lines = ["# Analysis Report\n"]
        if state.question:
            lines.append(f"**Question:** {state.question}\n")
        for c in state.claims:
            icon = "✅" if c.verification_status == "confirmed" else "⚠️"
            lines.append(f"{icon} {c.text} (value: {c.value}, status: {c.verification_status})")
        report = "\n".join(lines)

    if event_callback:
        event_callback("step_completed", {"step": "report"})

    return {"report": report, "status": "done"}