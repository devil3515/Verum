"""
Single place that reads .env / environment. Nothing else in the codebase
should call os.environ.get() for LLM config - it all goes through here.

.env keys (see .env.example):
  LLM_BASE_URL
  LLM_API_KEY
  LLM_MODEL
  LLM_TEMPERATURE   # optional, default 0.1
  LLM_MAX_TOKENS    # optional, default 8000
  LLM_PROJECT       # optional, default "default"

If you later want cheaper/stronger models for different nodes, that's a
later optimization (Phase 13) - add it then. One model is fine for now.
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    headers: dict


def load_llm_config() -> LLMConfig:
    return LLMConfig(
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
        model=os.environ["LLM_MODEL"],
        temperature=float(os.environ.get("LLM_TEMPERATURE", 0.1)),
        max_tokens=int(os.environ.get("LLM_MAX_TOKENS", 8000)),
        headers={"OpenAI-Project": os.environ.get("LLM_PROJECT", "default")},
    )