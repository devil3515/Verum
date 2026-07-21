"""
LLM client wrapper, OpenAI-compatible (Bedrock Mantle endpoint).

Added in this revision: explore_loop() — a multi-turn tool-calling loop
that lets the LLM call tools iteratively until it calls 'finish' or hits
max_iterations. This is what powers the analysis agent's explore phase.
"""
import json
import os
import re
from typing import Type, TypeVar, Callable, Optional

from openai import OpenAI
from pydantic import BaseModel

from analysis_engine.llm.config import load_llm_config

T = TypeVar("T", bound=BaseModel)


def extract_text_from_json_stream(accumulated_args: str) -> str:
    """
    Extracts the partial value of the "text" field from a streaming JSON string,
    supporting varying whitespace around key and colon.
    """
    match = re.search(r'"text"\s*:\s*"', accumulated_args)
    char_quote = '"'
    if not match:
        match = re.search(r"'text'\s*:\s*'", accumulated_args)
        char_quote = "'"
        if not match:
            return ""
    
    start = match.end()
    text_val = ""
    escaped = False
    for char in accumulated_args[start:]:
        if escaped:
            text_val += char
            escaped = False
        elif char == '\\':
            escaped = True
        elif char == char_quote:
            break
        else:
            text_val += char
    return text_val


class LLMClient:
    def __init__(self, config):
        self.config = config
        self._client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )

    def complete(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            extra_headers=self.config.headers,
        )
        return response.choices[0].message.content

    def structured(self, prompt: str, schema: Type[T]) -> T:
        tool_def = {
            "type": "function",
            "function": {
                "name": schema.__name__,
                "description": f"Return data matching the {schema.__name__} schema.",
                "parameters": schema.model_json_schema(),
            },
        }
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            tools=[tool_def],
            tool_choice={"type": "function", "function": {"name": schema.__name__}},
            extra_headers=self.config.headers,
        )
        message = response.choices[0].message
        if not message.tool_calls:
            raise ValueError(
                f"Model did not return a tool call for schema {schema.__name__}. "
                f"Raw content: {message.content!r}"
            )
        args = json.loads(message.tool_calls[0].function.arguments)
        return schema(**args)

    def explore_loop(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict],
        tool_dispatcher: Callable[[str, dict], object],
        finish_tool_name: str = "finish",
        max_iterations: int = 10,
        event_callback: Optional[Callable[[str, dict], None]] = None,
    ) -> tuple[list[dict], dict | None]:
        """
        Multi-turn tool-calling loop.

        The LLM calls tools from `tools` iteratively. After each call we
        execute the tool via `tool_dispatcher(tool_name, args) -> ToolResult`
        and feed the result back as a tool message. Loop ends when:
          - The LLM calls `finish_tool_name`, OR
          - max_iterations is reached

        Returns:
          (messages, finish_args)
          - messages: full conversation history
          - finish_args: the arguments passed to finish(), or None if
            max_iterations was hit without a finish call
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        finish_args = None
        budget_warning_sent = False

        for iteration in range(max_iterations):

            # at 80% of budget, inject a system message forcing finish()
            if not budget_warning_sent and iteration >= int(max_iterations * 0.8):
                messages.append({
                    "role": "user",
                    "content": (
                        f"[BUDGET WARNING] You have used {iteration} of {max_iterations} "
                        f"allowed tool calls. You MUST call {finish_tool_name}() NOW with "
                        "verdicts/claims for everything processed so far. "
                        "Do not make any more tool calls except finish()."
                    )
                })
                budget_warning_sent = True

            # Call event callback to signal thinking starting
            if event_callback:
                event_callback("chat_thinking_start", {})

            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                tools=tools,
                tool_choice="auto",
                extra_headers=self.config.headers,
                stream=True,
            )

            accumulated_content = ""
            accumulated_tool_calls = {}

            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # Stream reasoning tokens
                if delta.content:
                    accumulated_content += delta.content
                    if event_callback:
                        event_callback("chat_thinking_token", {"token": delta.content})

                # Stream tool call tokens
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in accumulated_tool_calls:
                            accumulated_tool_calls[idx] = {
                                "id": tc_delta.id,
                                "name": None,
                                "arguments": ""
                            }
                        tc = accumulated_tool_calls[idx]
                        if tc_delta.id:
                            tc["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tc["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                old_args = tc["arguments"]
                                tc["arguments"] += tc_delta.function.arguments

                                # Stream answer text tokens if calling the finish tool
                                if tc["name"] == finish_tool_name:
                                    old_text = extract_text_from_json_stream(old_args)
                                    new_text = extract_text_from_json_stream(tc["arguments"])
                                    if len(new_text) > len(old_text):
                                        diff = new_text[len(old_text):]
                                        if event_callback:
                                            event_callback("chat_answer_token", {"token": diff})

            tool_calls_list = []
            for idx, tc in sorted(accumulated_tool_calls.items()):
                if tc["name"]:
                    tool_calls_list.append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"]
                        }
                    })

            # Append assistant message to history
            messages.append({
                "role": "assistant",
                "content": accumulated_content,
                "tool_calls": tool_calls_list if tool_calls_list else None,
            })

            if not tool_calls_list:
                break

            for tc in tool_calls_list:
                tool_name = tc["function"]["name"]
                
                # Check if it is the finish tool call
                if tool_name == finish_tool_name:
                    try:
                        finish_args = json.loads(tc["function"]["arguments"])
                    except Exception:
                        finish_args = {
                            "text": extract_text_from_json_stream(tc["function"]["arguments"]),
                            "citations": [],
                            "follow_up_suggestions": []
                        }
                    return messages, finish_args

                # Execute other tools
                try:
                    args = json.loads(tc["function"]["arguments"])
                except Exception:
                    args = {}
                tool_result = tool_dispatcher(tool_name, args)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result.output if hasattr(tool_result, "output") else str(tool_result),
                })

                print(f"  [explore] {tool_name}({', '.join(f'{k}={v}' for k,v in args.items())}) -> {tool_result.output[:80] if hasattr(tool_result, 'output') else str(tool_result)[:80]}...")

        return messages, finish_args


def get_llm() -> "LLMClient":
    if os.environ.get("LLM_PROVIDER", "real").lower() == "fake":
        return _FakeLLMClient()
    config = load_llm_config()
    return LLMClient(config)


def call_structured(llm, prompt: str, schema: Type[T]) -> T:
    return llm.structured(prompt, schema)


class _FakeLLMClient:
    """Zero-network stand-in for testing graph wiring."""

    def structured(self, prompt: str, schema: Type[T]) -> T:
        if schema.__name__ == "PlannerOutput":
            return schema(
                steps=["clean_data", "run_analysis", "verify_claims", "synthesize_report"],
                reasoning="(fake LLM) standard tabular analysis request.",
            )
        if schema.__name__ == "SynthesisOutput":
            return schema(
                executive_summary="(fake LLM) The dataset was analyzed and key findings were identified.",
                data_quality_notes="(fake LLM) Minor cleaning was applied including duplicate removal.",
                findings=[{
                    "heading": "Notable variation detected",
                    "body": "(fake LLM) The primary metric showed notable variation across the dataset.",
                    "chart_ref": "",
                    "status": "confirmed",
                }],
                contradicted_claims=[],
                caveats="(fake LLM) Results based on a small sample dataset.",
            )
        raise NotImplementedError(
            f"_FakeLLMClient has no canned response for schema {schema.__name__}"
        )

    def complete(self, prompt: str) -> str:
        return "(fake LLM) no canned plain-text response."

    def explore_loop(self, system_prompt, user_prompt, tools, tool_dispatcher,
                     finish_tool_name="finish", max_iterations=10):
        """
        Fake explore loop — works for cleaning, analysis, and verification.
        Detects which agent is calling based on tool names available.
        """
        import re
        cols = re.findall(r"`([^`]+)`", user_prompt)
        first_col = next((c for c in cols if "(" not in c), None)

        tool_names = {t.get("function", {}).get("name") for t in tools}
        is_cleaning     = "drop_nulls" in tool_names
        is_verification = "recompute_groupby_mean" in tool_names
        is_chat         = "answer" in tool_names

        if is_verification:
            # extract claim ids from prompt to produce fake verdicts
            claim_ids = re.findall(r"id=([a-f0-9\-]{36})", user_prompt)
            verdicts = [
                {
                    "claim_id":        cid,
                    "status":          "confirmed",
                    "confidence":      0.9,
                    "reasoning":       "(fake LLM) recomputed value matches claimed value.",
                    "recomputed_value": 0.0,
                }
                for cid in claim_ids
            ]
            return [], {"verdicts": verdicts}

        if is_chat:
            return [], {
                "text": f"(fake LLM) Based on the data, {first_col or 'the dataset'} shows notable patterns.",
                "citations": [{"tool_name": "describe_column", "result_summary": f"(fake) stats for {first_col}"}],
                "follow_up_suggestions": ["What is the average value?", "Show me a chart of this."],
            }

        # call a real profiling tool so tests get real tool events
        if first_col:
            try:
                tool_dispatcher("profile_column", {"column": first_col})
                print(f"  [explore/fake] profile_column({first_col})")
            except Exception:
                try:
                    tool_dispatcher("describe_column", {"column": first_col})
                    print(f"  [explore/fake] describe_column({first_col})")
                except Exception:
                    pass

        if is_cleaning:
            finish_args = {"summary": f"(fake LLM) profiled {first_col or 'columns'}, no changes needed."}
        else:
            finish_args = {
                "claims": [{
                    "text":           f"(fake LLM) {first_col or 'a column'} showed notable variation.",
                    "metric":         f"{first_col or 'col'}_variation",
                    "value":          0.0,
                    "source_query":   f"describe_column({first_col or 'col'})",
                    "source_columns": [first_col] if first_col else [],
                }]
            }
        return [], finish_args