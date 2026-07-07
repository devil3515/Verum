from typing import Callable, Optional
from pydantic import BaseModel, Field

from analysis_engine.state import PipelineState
from analysis_engine.llm.client import get_llm, call_structured
from analysis_engine.registry import RUN_CALLBACKS


class PlannerOutput(BaseModel):
    steps: list[str] = Field(
        description="Ordered list of pipeline steps to execute, e.g. "
                    "['clean_data', 'run_analysis', 'verify_claims', 'synthesize_report']"
    )
    reasoning: str = Field(
        description="Brief explanation of why this plan fits the request/dataset"
    )


def _build_prompt(state: PipelineState) -> str:
    file_summaries = []
    for f in state.files:
        file_summaries.append(
            f"- file_id={f.file_id}, rows={f.row_count}, "
            f"size_bytes={f.size_bytes}, schema={f.schema_}"
        )
    files_block = "\n".join(file_summaries) if file_summaries else "(no files attached)"

    return f"""You are the planning agent for a data analysis pipeline.

Given the dataset(s) below, decide the ordered list of pipeline steps
needed to fulfill a standard "clean -> analyze -> verify -> report" request.

Available step names: clean_data, run_analysis, verify_claims, synthesize_report.

Dataset(s):
{files_block}

Return a structured plan with the step order and a one-sentence reasoning.
"""


def planner_node(
    state: PipelineState,
    event_callback: Optional[Callable[[str, dict], None]] = None,
) -> dict:
    if event_callback is None:
        event_callback = RUN_CALLBACKS.get(state.run_id)

    if event_callback:
        event_callback("step_started", {"step": "planning", "message": "Planning analysis approach..."})

    llm = get_llm()
    prompt = _build_prompt(state)
    result: PlannerOutput = call_structured(llm, prompt, PlannerOutput)

    print(f"[planner] reasoning: {result.reasoning}")
    print(f"[planner] plan: {result.steps}")

    if event_callback:
        event_callback("step_completed", {"step": "planning", "plan": result.steps})

    return {"plan": result.steps, "status": "cleaning"}