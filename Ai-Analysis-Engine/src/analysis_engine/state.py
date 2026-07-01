"""
Core state schema for the multi-agent pipeline.
This is the single contract every node reads from and writes to.
"""
from typing import Literal, Optional
from pydantic import BaseModel, Field


class WebSource(BaseModel):
    url: str
    snippet: str
    retrieved_at: str
    stance: Literal["supports", "contradicts", "neutral"] = "neutral"


class Claim(BaseModel):
    id: str
    text: str
    metric: str
    value: float
    source_query: str
    source_columns: list[str] = Field(default_factory=list)
    verification_status: Literal[
        "unverified", "confirmed", "contradicted", "unverifiable"
    ] = "unverified"
    confidence: float = 0.0
    web_sources: list[WebSource] = Field(default_factory=list)


class CleaningLogEntry(BaseModel):
    operation: str
    column: str
    rows_affected: int
    rationale: str


class FileMeta(BaseModel):
    file_id: str
    ref: str
    schema_: dict = Field(default_factory=dict, alias="schema")
    row_count: int = 0
    size_bytes: int = 0


class PipelineState(BaseModel):
    run_id: str
    user_id: str = "local-dev"

    files: list[FileMeta] = Field(default_factory=list)
    cleaned_refs: dict[str, str] = Field(default_factory=dict)
    cleaning_log: list[CleaningLogEntry] = Field(default_factory=list)

    plan: list[str] = Field(default_factory=list)
    question: Optional[str] = None
    claims: list[Claim] = Field(default_factory=list)
    chart_refs: list[str] = Field(default_factory=list)

    status: Literal[
        "planning",
        "cleaning",
        "analyzing",
        "verifying",
        "visualizing",
        "synthesizing",
        "done",
        "failed",
    ] = "planning"
    error: Optional[str] = None
    report: Optional[str] = None

    class Config:
        populate_by_name = True