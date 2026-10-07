"""Resolve operator-managed generation prompts once; persist in existing provenance.

Ragas instructions and judge embeddings remain fixed. No user dataset is uploaded.
Remote prompt reads require an explicit label and verified project configuration.
"""

import os
from contextvars import ContextVar
from functools import lru_cache

from backend.provenance import SYSTEM_PROMPT, digest

_generation = ContextVar("ragbench_generation_prompt", default=SYSTEM_PROMPT)


def resolve(name="ragbench-generation", fallback=SYSTEM_PROMPT, version="generation-1"):
    result = {"name": name, "version": version, "text": fallback, "source": "local"}
    label = os.getenv("RAGBENCH_PROMPT_LABEL")
    if label:
        from backend.observability import enabled, get_client

        if not enabled("benchmark"):
            raise ValueError("Remote prompt label requires verified Langfuse configuration")
        # Explicit prompt changes fail closed; an outage cannot silently select a different prompt.
        prompt = get_client().get_prompt(
            name, label=label, type="text", max_retries=0, fetch_timeout_seconds=2
        )
        if (
            not isinstance(prompt.prompt, str)
            or not 1 <= len(prompt.prompt) <= 8000
            or "{" in prompt.prompt
        ):
            raise ValueError(
                "Operator system prompt must be bounded plain text without placeholders"
            )
        result.update(version=prompt.version, text=prompt.prompt, source="langfuse", label=label)
    result["fingerprint"] = digest(result)
    return result


def validate(snapshot):
    if digest({k: v for k, v in snapshot.items() if k != "fingerprint"}) != snapshot["fingerprint"]:
        raise ValueError("Resolved prompt snapshot changed")


def generation():
    return _generation.get()


@lru_cache(maxsize=4)
def judge_templates(examples):
    """Exact local Ragas templates and schemas, with the existing example truncation."""
    from copy import deepcopy

    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    result = {}
    for metric in (
        Faithfulness(),
        ResponseRelevancy(strictness=3),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ):
        prompts = deepcopy(metric.get_prompts())
        for prompt in prompts.values():
            prompt.examples = prompt.examples[:examples]
        result[metric.name] = {name: prompt.to_string() for name, prompt in prompts.items()}
    return result
