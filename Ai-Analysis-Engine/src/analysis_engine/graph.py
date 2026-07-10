import uuid
import os
import sys
from typing import Callable, Optional
from langgraph.graph import StateGraph, END

from analysis_engine.state import PipelineState, FileMeta
from analysis_engine.registry import RUN_CALLBACKS
from analysis_engine.nodes.planner import planner_node
from analysis_engine.nodes.cleaning import cleaning_node
from analysis_engine.nodes.analysis import analysis_node
from analysis_engine.nodes.verification import verification_node
from analysis_engine.nodes.synthesis import synthesis_node


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_planner(state: PipelineState) -> str:
    return "cleaning" if "clean_data" in state.plan else "analysis"


def route_after_analysis(state: PipelineState) -> str:
    return "verification" if "verify_claims" in state.plan else "synthesis"


def route_after_verification(state: PipelineState) -> str:
    has_contradicted = any(c.verification_status == "contradicted" for c in state.claims)
    return "analysis" if has_contradicted else "synthesis"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("planner", planner_node)
    graph.add_node("cleaning", cleaning_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("verification", verification_node)
    graph.add_node("synthesis", synthesis_node)

    graph.set_entry_point("planner")

    graph.add_conditional_edges("planner", route_after_planner,
                                {"cleaning": "cleaning", "analysis": "analysis"})
    graph.add_edge("cleaning", "analysis")
    graph.add_conditional_edges("analysis", route_after_analysis,
                                {"verification": "verification", "synthesis": "synthesis"})
    graph.add_conditional_edges("verification", route_after_verification,
                                {"analysis": "analysis", "synthesis": "synthesis"})
    graph.add_edge("synthesis", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Local dev entrypoint
# ---------------------------------------------------------------------------

def main():
    file_path = sys.argv[1] if len(sys.argv) > 1 else "test_data/sales_messy.csv"
    question  = sys.argv[2] if len(sys.argv) > 2 else "Which region has the highest revenue?"

    if not os.path.exists(file_path):
        print(f"ERROR: file not found: {file_path}")
        sys.exit(1)

    compiled = build_graph()
    state = PipelineState(
        run_id=str(uuid.uuid4()),
        question=question,
        files=[FileMeta(file_id="input-file-1", ref=file_path, row_count=0)],
    )

    print("=" * 60)
    print(f"run_id:   {state.run_id}")
    print(f"file:     {file_path}")
    print(f"question: {question}")
    print("=" * 60)

    final = compiled.invoke(state)
    fs = PipelineState(**final)
    print("\n" + "=" * 60)
    print(f"status:        {fs.status}")
    print(f"cleaning ops:  {len(fs.cleaning_log)}")
    print(f"claims:        {len(fs.claims)}")
    print(f"charts:        {len(fs.chart_refs)}")
    print(f"\n--- report ---\n{fs.report}")


if __name__ == "__main__":
    main()