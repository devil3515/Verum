import json
import os
import uuid
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel, Field

from analysis_engine.chat_state import ChatSession, ChatMessage
from analysis_engine.llm.client import get_llm ,call_structured
from analysis_engine.tools.data_io import load_dataframe
from analysis_engine.tools.analysis_tools import dispatch_tool, EXPLORE_TOOLS
from analysis_engine.tools.base import ToolResult
from analysis_engine.nodes.analysis import _save_chart, _get_charts_dir



class Citations(BaseModel):
    tool_name: str
    result_answer: str = Field(description="Which tool produced this fact, e.g. groupby_mean")


class AnswerOutput(BaseModel):
    text: str = Field(description=(
            "The full answer to the user's question in plain language. "
            "Be specific with numbers. Every number must come from a tool you called this turn "
            "or from the pipeline_context provided. Do not invent values."
        )
    )
    citations: list[Citations] = Field(
        default_factory=list,
        description="One citation per tool call that contributed a fact to the answer."
    )
    follow_up_suggestions: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Up to 3 natural follow-up questions the user might want to ask next."
    )


CHAT_SYSTEM_PROMPT = """You are Verum's data assistant. You answer questions about a
specific dataset that has already been cleaned and analyzed.

You have access to the same tools as the analysis agent. Use them to answer the
user's question with precision.

RULES:
1. Answer the specific question asked. Don't explore unrelated topics.
2. Use tools to compute any numbers in your answer — never invent figures.
3. px and go are pre-imported globals. Do NOT use import statements.
4. Use result = <value> for output, not print().
5. Generate a chart with run_code + plotly if it helps answer the question visually.
6. After at most 6 tool calls, call answer() with your response.
7. Reference the pipeline_context if it already contains the answer — don't
   re-compute what the pipeline already found.
8. Provide a detailed, clear explanation of your findings and any generated charts. When asked to compare categories/regions or analyze differences, don't just output the values — explain the insights, trends, and significance of the results in a friendly, conversational manner.

CITATION RULE: Every numeric fact in your answer text must appear in a citation
pointing to the tool call that produced it. If you can't cite a number, don't
state it.
"""


CHAT_TOOLS = [t for t in EXPLORE_TOOLS if t["function"]["name"] != "finish"] + [
    {
        "type": "function",
        "function": {
            "name": "answer",
            "description": "Call this when you have enough evidence to answer the question. Required — do not stop without calling answer().",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Full answer in plain language. Be specific with numbers. Cite tool results."
                    },
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "tool_name": {"type": "string"},
                                "result_summary": {"type": "string"}
                            },
                            "required": ["tool_name", "result_summary"]
                        },
                        "description": "One entry per tool call that contributed a fact to the answer."
                    },
                    "follow_up_suggestions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Up to 3 follow-up questions the user might ask next."
                    }
                },
                "required": ["text", "citations"]
            }
        }
    }
]


def  _build_pipeline_context(final_state: dict) -> str:
    """
    Summarise the pipeline's existing findings so the chat agent
    doesn't re-derive what was already computed.
    """
    lines = ["=== PIPELINE ANALYSIS (already completed) ==="]

    claims = final_state.get("claims", [])
    if claims:
        lines.append(f"\nKey findings ({len(claims)} confirmed claims):")
        for c in claims:
            status = c.get("verification_status", "unverified")
            lines.append(f"  [{status}] {c.get('text', '')} (value={c.get('value')})")

    report = final_state.get("report", "")
    if report:
        for line in report.split("\n"):
            if "executive summary" in line.lower():
                idx = report.lower().find("executive summary")
                snippet = report[idx:idx+500].split("##")[0].strip()
                lines.append(f"\nExecutive summary:\n{snippet}")
                break

    lines.append("\nYou may reference these findings directly in your answer.")
    return "\n".join(lines)


def run_chat_turn(
        session: ChatSession,
        user_message: str,
        final_state: dict,
        event_callback: Optional[Callable[[str, dict], None]] = None,
) -> ChatMessage:
    df = load_dataframe(session.cleaned_ref)
    run_id = session.run_id
    tool_results: list[ToolResult] = []
    chart_refs: list[str] = []

    def tool_dispatcher(tool_name: str, args: dict) -> ToolResult:
        if event_callback:
            event_callback("chat_tool_called",{
                "tool": tool_name,
                "args": {k: str(v)[:80] for k, v in args.items()},
            })
        result = dispatch_tool(df, tool_name, args, run_id=run_id)
        tool_results.append(result)

        if result.chart_spec:
            ref = _save_chart(result.chart_spec)
            chart_refs.append(ref)
            if event_callback:
                event_callback("chat_chart_generated", {"ref": ref})

        if event_callback:
            event_callback("chat_tool_result", {
                "tool": tool_name,
                "output": result.output[:200],
                "has_chart": result.chart_spec is not None,
            })
        return result

    pipeline_context = _build_pipeline_context(final_state)
    history_lines = []
    for msg in session.messages[-6:]:   # last 6 messages = 3 turns of context
        history_lines.append(f"{msg.role.upper()}: {msg.content}")

    history_block = "\n".join(history_lines) if history_lines else "(first message)"

    user_prompt = f"""{pipeline_context}
                    === CONVERSATION HISTORY ===
                    {history_block}

                    === CURRENT QUESTION ===
                    {user_message}

                    Use your tools to answer this question precisely. Call answer() when done.
                    """
    llm = get_llm()
    _, answer_args = llm.explore_loop(
        system_prompt=CHAT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tools=CHAT_TOOLS,
        tool_dispatcher=tool_dispatcher,
        finish_tool_name="answer",
        max_iterations=8,
        event_callback=event_callback,
    )
    if not answer_args:
         answer_args = {
            "text": "I wasn't able to compute a confident answer for that question. Try rephrasing or asking something more specific.",
            "citations": [],
            "follow_up_suggestions": [],
        }

    citations = [
        {"tool_name": c.get("tool_name", ""), "result_summary": c.get("result_summary", "")}
        for c in answer_args.get("citations", [])
    ]

    if event_callback:
        event_callback("chat_answer", {
            "text":                 answer_args["text"],
            "citations":            citations,
            "chart_refs":           chart_refs,
            "follow_up_suggestions": answer_args.get("follow_up_suggestions", []),
        })

    return ChatMessage(
        role="assistant",
        content=answer_args["text"],
        chart_ref=chart_refs,
        citations=citations,
        follow_up_suggestions=answer_args.get("follow_up_suggestions", []),
    )


