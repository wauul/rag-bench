from typing import Literal
from pydantic import AliasChoices, BaseModel, Field, model_validator

MODELS = ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    reference: str = Field(min_length=3, max_length=6000)


class TestSet(BaseModel):
    name: str = Field(default="My test set", min_length=1, max_length=80)
    questions: list[Question] = Field(min_length=1, max_length=20)


class Configuration(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    # WordPiece tokens, capped below MiniLM's 256-token window to avoid silent truncation.
    chunk_size: int = Field(default=192, ge=32, le=240)
    overlap: int = Field(default=32, ge=0)
    embedding_model: Literal["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"] = MODELS[0]
    rerank: bool = False
    context_k: int = Field(default=3, ge=1, le=8,
                           validation_alias=AliasChoices("context_k", "top_k"))
    candidate_k: int | None = Field(default=None, ge=1, le=40)

    @model_validator(mode="before")
    @classmethod
    def check_legacy_top_k(cls, data):
        if isinstance(data, dict) and "top_k" in data and "context_k" in data:
            if data["top_k"] != data["context_k"]:
                raise ValueError("top_k and context_k must agree when both are supplied")
        return data

    @model_validator(mode="after")
    def check_overlap(self):
        if self.overlap >= self.chunk_size:
            raise ValueError("Overlap must be smaller than chunk size")
        if not self.name.strip():
            raise ValueError("Configuration name cannot be blank")
        self.name = self.name.strip()
        # Old stored configurations retain their original candidate pool and results.
        if self.candidate_k is None:
            self.candidate_k = self.context_k
        if self.candidate_k < self.context_k:
            raise ValueError("Candidate count must be at least the final passage count")
        return self


class RunRequest(BaseModel):
    document_set_id: str
    test_set_id: str
    configuration_ids: list[str] = Field(min_length=2, max_length=4)

    @model_validator(mode="after")
    def unique_configs(self):
        if len(set(self.configuration_ids)) != len(self.configuration_ids):
            raise ValueError("Choose distinct configurations")
        return self
