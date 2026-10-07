"""Mutable remote labels resolve at creation; resume uses saved content only."""

from types import SimpleNamespace

import pytest

from backend import observability, prompts
from backend.provenance import validate_snapshot
from tests.graph_fakes import create_run


def test_local_fallback_is_versioned_and_tamper_evident(monkeypatch):
    monkeypatch.delenv("RAGBENCH_PROMPT_LABEL", raising=False)
    snapshot = prompts.resolve()
    assert snapshot["source"] == "local" and snapshot["version"] == "generation-1"
    prompts.validate(snapshot)
    snapshot["text"] = "changed"
    with pytest.raises(ValueError, match="changed"):
        prompts.validate(snapshot)


def test_remote_label_freezes_content_for_resume(monkeypatch, tmp_path):
    from backend.pipeline import execute_run
    from backend.storage import Store
    from tests.graph_fakes import install

    monkeypatch.setenv("RAGBENCH_PROMPT_LABEL", "approved")
    monkeypatch.setattr(observability, "enabled", lambda *_: True)
    remote = SimpleNamespace(prompt="Use the saved approved evidence.", version=7)
    calls = []
    client = SimpleNamespace(get_prompt=lambda *a, **kw: (calls.append(kw), remote)[1])
    monkeypatch.setattr(observability, "get_client", lambda: client)
    store = Store(tmp_path)
    run = create_run(store, ("langgraph",))
    assert run["provenance"]["generation_prompt"]["version"] == 7
    assert len(calls) == 1
    remote.prompt = "Do not use this changed label."
    model_calls, messages = install(monkeypatch)
    result = execute_run(store, run["id"])
    assert result["status"] == "completed", result.get("error")
    assert len(calls) == 1 and model_calls["generation"] == 1
    assert messages[0][0][1] == "Use the saved approved evidence."
    validate_snapshot(store, result)


def test_explicit_remote_label_missing_configuration_fails(monkeypatch):
    monkeypatch.setenv("RAGBENCH_PROMPT_LABEL", "approved")
    monkeypatch.setenv("RAGBENCH_LANGFUSE", "false")
    with pytest.raises(ValueError, match="verified"):
        prompts.resolve()
