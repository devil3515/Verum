import uuid
from pydantic import BaseModel, Field

from analysis_engine.state import PipelineState, Claim
from analysis_engine.llm.client import get_llm, call_structured
from analysis_engine.tools.analysis_ops import run_full_analysis, MetricResult
from analysis_engine.tools.data_io import load_dataframe



class ClaimDraft(BaseModel):
    metric_key: str = Field(description="Must exactly match one of the provided metric keys")
    text: str = Field(description="One-sentence plain-language insight narrating this metric")


class ClaimSelection(BaseModel):
    claims: list[ClaimDraft] = Field(
        description="The most interesting/important metrics worth surfacing as insights. "
                    "Be selective - only pick metrics that tell a meaningful story, not "
                    "every computed number."
    )


def _build_prompt(metric_menu: dict[str, MetricResult]) -> str:
    lines = []
    for key, result in metric_menu.items():
        lines.append(f"- key=\"{key}\" value={result.value:.4f} (from: {result.source_query})")
    menu_block = "\n".join(lines) if lines else "(no metrics computed)"

    return f"""You are the analysis narration agent for a data pipeline.

Below is a menu of pre-computed statistics. Pick the ones that represent
genuinely interesting or important findings (e.g. notable group
differences, strong correlations, surprising extremes) and write a
one-sentence plain-language insight for each. Skip metrics that are
unremarkable or redundant. Do not invent numbers - reference only the
metric keys given below; the actual values will be filled in
automatically from the key you choose.

Metrics:
{menu_block}

Select at most 5 of the most noteworthy metrics.
"""


def analysis_node(state: PipelineState) -> dict:
    if not state.cleaned_refs:
        print("[analysis] no cleaned files available, skipping")
        return {"status": "verifying"}

    all_claims = []

    for file_id, cleaned_ref in state.cleaned_refs.items():
        print(f"[analysis] running stats on file_id={file_id} ref={cleaned_ref}")
        df = load_dataframe(cleaned_ref)

        metric_menu = run_full_analysis(df)
        print(f"[analysis] computed {len(metric_menu)} candidate metrics")

        if not metric_menu:
            continue

        llm = get_llm()
        prompt = _build_prompt(metric_menu)
        selection: ClaimSelection = call_structured(llm, prompt, ClaimSelection)

        for draft in selection.claims:
            if draft.metric_key not in metric_menu:
                print(f"[analysis] WARNING: LLM referenced unknown metric_key "
                      f"'{draft.metric_key}', skipping")
                continue

            result = metric_menu[draft.metric_key]
            claim = Claim(
                id=str(uuid.uuid4()),
                text=draft.text,
                metric=draft.metric_key,
                value=result.value,           # <- injected from code, never from the LLM
                source_query=result.source_query,
                source_columns=result.source_columns,
            )
            all_claims.append(claim)
            print(f"[analysis] claim: {claim.text} (value={claim.value:.2f})")

    return {
        "claims": all_claims,
        "status": "verifying",
    }

