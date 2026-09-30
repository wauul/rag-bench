import pytest
from pydantic import ValidationError
from backend.models import Configuration
from backend.settings import DEFAULT_MODEL, load_settings


def test_legacy_top_k_and_new_candidate_validation():
    old = Configuration(name="Legacy", top_k=2)
    assert old.context_k == old.candidate_k == 2
    assert "top_k" not in old.model_dump()
    new = Configuration(name="Reranked", context_k=3, candidate_k=20, rerank=True)
    assert new.candidate_k == 20 and new.context_k == 3
    for values in ({"candidate_k": 2, "context_k": 3}, {"candidate_k": 41},
                   {"top_k": 2, "context_k": 3}, {"name": "   "}):
        with pytest.raises(ValidationError):
            Configuration(**{"name": "Invalid", **values})


@pytest.mark.parametrize("name,value", [("INFERENCE_BACKEND", "typo"),
    ("GROQ_REQUEST_INTERVAL", "nan"), ("GROQ_REQUEST_INTERVAL", "0"),
    ("JUDGE_EXAMPLES", "4"), ("JUDGE_EXAMPLES", "text"), ("GROQ_MODEL", ""), ("GROQ_MODEL", "   ")])
def test_invalid_environment_settings_identify_the_variable(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        load_settings()


def test_example_model_matches_runtime_default():
    from pathlib import Path
    from dotenv import dotenv_values
    settings = dotenv_values(Path(__file__).resolve().parents[1] / ".env.example")
    assert settings["GROQ_MODEL"] == DEFAULT_MODEL
