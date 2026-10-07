"""Validate operational settings once at startup and snapshot measurement settings."""

import os
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

DEFAULT_MODEL = "qwen/qwen3.8-27b"


class RuntimeSettings(BaseModel):
    model: str = Field(default=DEFAULT_MODEL, min_length=1)
    inference_backend: Literal["sentence-transformers", "onnx"] = "sentence-transformers"
    request_interval: float = Field(default=4, ge=0.1, allow_inf_nan=False)
    judge_examples: int = Field(default=0, ge=0, le=3)

    @field_validator("model", mode="before")
    @classmethod
    def strip_model(cls, value):
        return value.strip() if isinstance(value, str) else value

    @property
    def reasoning_effort(self):
        return "none" if self.model.startswith("qwen/") else "low"

    def provenance(self):
        return {
            "generator": self.model,
            "judge": self.model,
            "ragas": "0.3.9",
            "relevancy_embeddings": "sentence-transformers/all-MiniLM-L6-v2",
            "inference_backend": self.inference_backend,
            "precision": "dynamic-int8" if self.inference_backend == "onnx" else "float32",
            "generator_temperature": 0,
            "judge_temperature": "Ragas default per metric",
            "judge_prompt_examples": self.judge_examples,
            "reasoning_effort": self.reasoning_effort,
            "relevancy_strictness": 3,
        }


def load_settings():
    names = {
        "model": "GROQ_MODEL",
        "inference_backend": "INFERENCE_BACKEND",
        "request_interval": "GROQ_REQUEST_INTERVAL",
        "judge_examples": "JUDGE_EXAMPLES",
    }
    try:
        return RuntimeSettings.model_validate(
            {field: os.environ[name] for field, name in names.items() if name in os.environ}
        )
    except ValidationError as exc:
        details = "; ".join(f"{names[str(e['loc'][0])]}: {e['msg']}" for e in exc.errors())
        raise ValueError("Invalid backend settings: " + details) from None


def validate_operations() -> None:
    """Credentials are required unless local development explicitly opts out."""
    environment = os.getenv("APP_ENV", "development")
    if environment not in {"development", "production"}:
        raise ValueError("APP_ENV must be development or production")
    if not os.getenv("API_TOKEN") and os.getenv("RAGBENCH_ALLOW_INSECURE_LOCAL") != "true":
        raise ValueError(
            "API_TOKEN is required; insecure local access requires an explicit opt-out"
        )
    if environment == "production":
        if len(os.getenv("API_TOKEN", "").strip()) < 32:
            raise ValueError("Production API_TOKEN must contain at least 32 characters")
        if os.getenv("DATABASE_URL"):
            from backend.postgres import validate_database_url

            validate_database_url(os.environ["DATABASE_URL"])
            from backend.postgres import connect

            with connect(os.environ["DATABASE_URL"]) as db:
                role = db.execute(
                    "SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user"
                ).fetchone()
                if role is None or any(role):
                    raise ValueError("Production database role must not bypass row-level security")
        elif os.getenv("DATA_STORAGE") != "persistent":
            raise ValueError("Production requires DATA_STORAGE=persistent and a mounted disk")
