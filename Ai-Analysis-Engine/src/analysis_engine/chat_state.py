from typing import Literal, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    chart_ref: list[str] = Field(default_factory=list)
    citations: list[dict] = Field(default_factory=list)


class ChatSession(BaseModel):
    session_id: str
    run_id: str
    cleaned_ref: str
    messages: list[ChatMessage] = Field(default_factory=list)
    pipeline_summary: str = ""