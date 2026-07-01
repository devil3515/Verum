"""
Phase 1 — Stub graph, end to end.

Every node here is a stub: no LLM calls, no real pandas, no sandbox.
Each node just mutates state with fake data and logs what it did.
The point of this phase is to validate the GRAPH TOPOLOGY and the
STATE SCHEMA flow, not to do real work yet.

Run it directly:
    python -m analysis_engine.graph
"""
import uuid
from langgraph.graph import StateGraph, END

from analysis_engine.state import (
    PipelineState,
    Claim,
    CleaningLogEntry,
    FileMeta,
)
from analysis_engine.nodes.planner import planner_node
from analysis_engine.nodes.cleaning import cleaning_node
from analysis_engine.nodes.analysis import analysis_node


# ---------------------------------------------------------------------------
# Stub nodes
# ---------------------------------------------------------------------------


def verification_node(state: PipelineState) -> dict:
    print(f"[verification] checking {len(state.claims)} claim(s)")
    updated_claims = []
    for c in state.claims:
        c = c.model_copy(update={
            "verification_status": "confirmed",
            "confidence": 0.92,
        })
        updated_claims.append(c)
    return {
        "claims": updated_claims,
        "status": "synthesizing",
    }


def synthesis_node(state: PipelineState) -> dict:
    print(f"[synthesis] assembling final report")
    lines = ["# Report (stub)\n"]
    for c in state.claims:
        lines.append(f"- {c.text} (status: {c.verification_status})")
    return {
        "report": "\n".join(lines),
        "status": "done",
    }


# ---------------------------------------------------------------------------
# Plan-aware conditional edges
# ---------------------------------------------------------------------------

def route_after_planner(state: PipelineState) -> str:
    if "clean_data" in state.plan:
        return "cleaning"
    # no cleaning needed per the plan - go straight to analysis
    return "analysis"


def route_after_analysis(state: PipelineState) -> str:
    if "verify_claims" in state.plan:
        return "verification"
    return "synthesis"


def route_after_verification(state: PipelineState) -> str:
    """
    If any claim came back contradicted, loop back to analysis to
    regenerate it (Phase 5 will make 'contradicted' a real outcome -
    today verification_node always confirms, so this branch won't
    trigger yet, but the wiring is correct and ready for Phase 5).
    """
    has_contradicted = any(c.verification_status == "contradicted" for c in state.claims)
    if has_contradicted:
        return "analysis"
    return "synthesis"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("planner", planner_node)
    graph.add_node("cleaning", cleaning_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("verification", verification_node)
    graph.add_node("synthesis", synthesis_node)

    graph.set_entry_point("planner")

    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "cleaning": "cleaning",
            "analysis": "analysis",
        },
    )

    graph.add_edge("cleaning", "analysis")

    graph.add_conditional_edges(
        "analysis",
        route_after_analysis,
        {
            "verification": "verification",
            "synthesis": "synthesis",
        },
    )

    graph.add_conditional_edges(
        "verification",
        route_after_verification,
        {
            "analysis": "analysis",
            "synthesis": "synthesis",
        },
    )

    graph.add_edge("synthesis", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Local runnable entrypoint — no server, no Django, just a script
# ---------------------------------------------------------------------------

def main():
    import os
    import sys

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = os.environ.get(
            "ANALYSIS_ENGINE_INPUT_FILE", "test_data/complex_10k.csv"
        )

    if not os.path.exists(file_path):
        print(f"ERROR: file not found: {file_path}")
        print("Usage: python -m analysis_engine.graph <path-to-csv>")
        print("   or: ANALYSIS_ENGINE_INPUT_FILE=<path> python -m analysis_engine.graph")
        sys.exit(1)

    compiled_graph = build_graph()

    initial_state = PipelineState(
        run_id=str(uuid.uuid4()),
        question="Which region has the highest revenue and is there an outlier?",
        files=[
            FileMeta(
                file_id="input-file-1",
                ref=file_path,
                row_count=0,
            )
        ],
    )

    print("=" * 60)
    print("Starting stub pipeline run:", initial_state.run_id)
    print("=" * 60)

    final_state_dict = compiled_graph.invoke(initial_state)
    final_state = PipelineState(**final_state_dict)

    print("=" * 60)
    print("FINAL STATE")
    print("=" * 60)
    print("status:", final_state.status)
    print("cleaning_log entries:", len(final_state.cleaning_log))
    print("claims:", [c.text for c in final_state.claims])
    print("chart_refs:", final_state.chart_refs)
    print("\n--- report ---\n")
    print(final_state.report)


if __name__ == "__main__":
    main()