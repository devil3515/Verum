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
        Fake explore loop: runs describe_column on first numeric column,
        then calls finish with a canned claim. Enough to validate wiring.
        """
        from analysis_engine.tools.analysis_tools import ToolResult

        # fake: call describe_column on the first numeric-looking column we find
        # by parsing column names from the user_prompt
        import re
        cols = re.findall(r"`([^`]+)`", user_prompt)
        numeric_col = next((c for c in cols if c not in ["object", "str"]), None)

        tool_results = []
        if numeric_col:
            fake_args = {"column": numeric_col}
            result = tool_dispatcher("describe_column", fake_args)
            print(f"  [explore/fake] describe_column({numeric_col}) -> {result.output[:80]}...")

        # fake finish
        finish_args = {
            "claims": [
                {
                    "text": f"(fake LLM) {numeric_col or 'a column'} showed notable variation.",
                    "metric": f"{numeric_col or 'col'}_variation",
                    "value": 0.0,
                    "source_query": f"describe_column({numeric_col or 'col'})",
                    "source_columns": [numeric_col] if numeric_col else [],
                }
            ]
        }
        return [], finish_args