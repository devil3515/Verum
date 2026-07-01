import json
import os
from typing import Type, TypeVar

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


def get_llm() -> "LLMClient":
    """
    Returns the LLM client. Set LLM_PROVIDER=fake in .env to get a
    zero-network stand-in instead (used for testing graph wiring).
    """
    if os.environ.get("LLM_PROVIDER", "real").lower() == "fake":
        return _FakeLLMClient()

    config = load_llm_config()
    return LLMClient(config)


def call_structured(llm, prompt: str, schema: Type[T]) -> T:
    return llm.structured(prompt, schema)


class _FakeLLMClient:
    """
    Zero-network stand-in for testing graph wiring. Only knows canned
    responses for schemas used so far - add a branch here as new phases
    introduce new structured schemas.
    """

    def structured(self, prompt: str, schema: Type[T]) -> T:
        if schema.__name__ == "PlannerOutput":
            return schema(
                steps=["clean_data", "run_analysis", "verify_claims", "synthesize_report"],
                reasoning="(fake LLM) standard tabular analysis request, no unusual steps.",
            )
        raise NotImplementedError(
            f"_FakeLLMClient has no canned response for schema {schema.__name__}"
        )

    def complete(self, prompt: str) -> str:
        return "(fake LLM) no canned plain-text response configured."