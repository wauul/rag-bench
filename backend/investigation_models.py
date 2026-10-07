"""Version one diagnostic output: hypotheses, never proven root causes."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Category = Literal[
    "retrieval_miss",
    "reranking_loss",
    "context_selection_loss",
    "unused_context",
    "unsupported_claims",
    "source_insufficient",
    "reference_ambiguity",
    "evaluation_failure",
    "insufficient_evidence",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Evidence(StrictModel):
    evidence_id: str = Field(max_length=120)
    quote: str = Field(min_length=1, max_length=1500)


class Hypothesis(StrictModel):
    category: Category
    strength: Literal["limited", "moderate", "strong"]
    rationale: str = Field(min_length=10, max_length=1500)
    supporting: list[Evidence] = Field(max_length=8)
    contradicting: list[Evidence] = Field(max_length=8)


class Diagnosis(StrictModel):
    summary: str = Field(min_length=10, max_length=1200)
    hypotheses: list[Hypothesis] = Field(min_length=1, max_length=8)
    limitations: list[str] = Field(max_length=12)


class InvestigationRequest(StrictModel):
    configuration_id: str
    question_index: int = Field(ge=0)
    search_sources: bool = False
    run_again: bool = False
    request_key: str = Field(min_length=8, max_length=80)
