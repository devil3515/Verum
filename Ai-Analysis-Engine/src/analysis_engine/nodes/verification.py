import json
from typing import Callable, Optional

from analysis_engine.state import PipelineState, Claim
from analysis_engine.llm.client import get_llm
from analysis_engine.registry import RUN_CALLBACKS
from analysis_engine.tools.data_io import load_dataframe
from analysis_engine.tools.verification_tools import (
    dispatch_verification_tool,
    VERIFICATION_TOOLS,
    TOLERANCE,
    _pct_diff,
)
from analysis_engine.tools.base import ToolResult



SYSTEM_PROMPT = """You are a verification agent. Your job is to independently
verify each claim using the provided tools, then call finish() with verdicts.

STRICT RULES:
- Call finish() exactly once after all claims are evaluated.
- finish() must include a verdict for EVERY claim.
- Never stop without calling finish().

VERIFICATION STRATEGY:
1. For numeric claims (means, counts, correlations):
   → Use recompute_groupby_mean, recompute_column_mean, recompute_correlation,
     recompute_count, or recompute_value_count.
   → If recomputed value is within 2% of claimed value → confirmed.
   → If it differs by more than 2% → contradicted.

2. For string/category claims ("top factor is X", "most common value is Y"):
   → Use recompute_value_count — it handles case-insensitive matching and
     shows actual spellings so you don't waste calls guessing the exact string.

3. For externally-referenced claims ("consistent with industry trends",
   "reflects market-wide patterns", "aligns with seasonal data"):
   → Use web_search to find supporting or contradicting evidence.
   → Classify each source as supports/contradicts/neutral in your reasoning.

4. If you genuinely cannot verify a claim with any tool → unverifiable.

BUDGET: Use at most {budget} tool calls total, then call finish().
"""


def _build_prompt(claims: list[Claim], df_columns: list[str]) -> str:
    lines = [
        f"Available columns: {df_columns}",
        f"Total claims to verify: {len(claims)}",
        f"Tool budget: {len(claims) * 4} calls maximum — then you MUST call finish().",
        "",
        "Claims to verify:",
    ]
    for c in claims:
        lines.append(
            f"  id={c.id}\n"
            f"  text: {c.text}\n"
            f"  claimed_value: {c.value}\n"
            f"  source_query: {c.source_query}\n"
            f"  source_columns: {c.source_columns}\n"
        )
    lines.append(
        "\nIMPORTANT: After using your tool budget, call finish() immediately with verdicts "
        "for ALL claims. Never leave a claim without a verdict. If unsure, mark it unverifiable."
    )
    return "\n".join(lines)


def _apply_verdicts(claims: list[Claim], verdicts: list[dict]) -> list[Claim]:
    verdict_map = {v["claim_id"]: v for v in verdicts}
    updated = []
    for claim in claims:
        verdict = verdict_map.get(claim.id)
        if not verdict:
            # LLM missed this claim — mark unverifiable
            updated.append(claim.model_copy(update={
                "verification_status": "unverifiable",
                "confidence": 0.0,
            }))
            continue

        # cross-check: if LLM says confirmed but recomputed value differs too much,
        # override to contradicted — don't trust the LLM's judgment blindly
        status = verdict["status"]
        recomputed = verdict.get("recomputed_value")
        if (
            status == "confirmed"
            and recomputed is not None
            and _pct_diff(recomputed, claim.value) > TOLERANCE
        ):
            status = "contradicted"
            print(
                f"[verification] override: claimed={claim.value}, "
                f"recomputed={recomputed} → contradicted"
            )

        updated.append(claim.model_copy(update={
            "verification_status": status,
            "confidence": float(verdict.get("confidence", 0.5)),
        }))
    return updated


def verification_node(
    state: PipelineState,
    event_callback: Optional[Callable[[str, dict], None]] = None,
) -> dict:
    if event_callback is None:
        event_callback = RUN_CALLBACKS.get(state.run_id)

    if event_callback:
        event_callback("step_started", {
            "step": "verifying",
            "message": f"Verifying {len(state.claims)} claim(s)..."
        })

    if not state.claims:
        print("[verification] no claims to verify")
        if event_callback:
            event_callback("step_completed", {"step": "verifying"})
        return {"status": "synthesizing"}

    if not state.cleaned_refs:
        print("[verification] no cleaned data — marking all unverifiable")
        updated = [
            c.model_copy(update={"verification_status": "unverifiable", "confidence": 0.0})
            for c in state.claims
        ]
        if event_callback:
            event_callback("step_completed", {"step": "verifying"})
        return {"claims": updated, "status": "synthesizing"}

    # use first cleaned file for recomputation
    cleaned_ref = next(iter(state.cleaned_refs.values()))
    df = load_dataframe(cleaned_ref)
    df_columns = list(df.columns)

    print(f"[verification] verifying {len(state.claims)} claim(s) against {cleaned_ref}")

    def tool_dispatcher(tool_name: str, args: dict) -> ToolResult:
        if event_callback:
            event_callback("tool_called", {
                "node": "verification",
                "tool": tool_name,
                "args": {k: str(v)[:80] for k, v in args.items()},
            })
        result = dispatch_verification_tool(df, tool_name, args)
        if event_callback:
            event_callback("tool_result", {
                "node": "verification",
                "tool": tool_name,
                "output": result.output[:200],
            })
        return result

    budget = len(state.claims) * 5 + 4
    llm = get_llm()
    _, finish_args = llm.explore_loop(
        system_prompt=SYSTEM_PROMPT.format(budget=budget),
        user_prompt=_build_prompt(state.claims, df_columns),
        tools=VERIFICATION_TOOLS,
        tool_dispatcher=tool_dispatcher,
        finish_tool_name="finish",
        max_iterations=budget,
    )

    if finish_args and "verdicts" in finish_args:
        verdicts = finish_args["verdicts"]
        updated_claims = _apply_verdicts(state.claims, verdicts)
    else:
        print("[verification] WARNING: no verdicts returned — marking all unverifiable")
        updated_claims = [
            c.model_copy(update={"verification_status": "unverifiable", "confidence": 0.0})
            for c in state.claims
        ]

    confirmed    = sum(1 for c in updated_claims if c.verification_status == "confirmed")
    contradicted = sum(1 for c in updated_claims if c.verification_status == "contradicted")
    print(f"[verification] confirmed={confirmed} contradicted={contradicted} "
          f"unverifiable={len(updated_claims)-confirmed-contradicted}")

    if event_callback:
        event_callback("step_completed", {
            "step":          "verifying",
            "confirmed":     confirmed,
            "contradicted":  contradicted,
        })

    return {"claims": updated_claims, "status": "synthesizing"}