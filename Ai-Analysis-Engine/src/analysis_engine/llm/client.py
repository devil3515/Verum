"""
LLM client wrapper, OpenAI-compatible (Bedrock Mantle endpoint).

Added in this revision: explore_loop() — a multi-turn tool-calling loop
that lets the LLM call tools iteratively until it calls 'finish' or hits
max_iterations. This is what powers the analysis agent's explore phase.
"""
import json
import os
from typing import Type, TypeVar, Callable

from openai import OpenAI
from pydantic import BaseModel

from analysis_engine.llm.config import load_llm_config

T = TypeVar("T", bound=BaseModel)


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

        for iteration in range(max_iterations):
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                tools=tools,
                tool_choice="auto",
                extra_headers=self.config.headers,
            )

            message = response.choices[0].message

            # append assistant turn to history
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in (message.tool_calls or [])
                ] or None,
            })

            if not message.tool_calls:
                # model stopped without calling finish - treat as done
                break

            for tc in message.tool_calls:
                tool_name = tc.function.name
                args = json.loads(tc.function.arguments)

                if tool_name == finish_tool_name:
                    finish_args = args
                    return messages, finish_args

                # execute the tool and feed result back
                tool_result = tool_dispatcher(tool_name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result.output if hasattr(tool_result, "output") else str(tool_result),
                })

                print(f"  [explore] {tool_name}({', '.join(f'{k}={v}' for k,v in args.items())}) -> {tool_result.output[:80]}...")

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