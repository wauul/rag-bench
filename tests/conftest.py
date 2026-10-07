"""Legacy UI/API fixtures opt into isolated, explicitly insecure local development."""

import pytest


@pytest.fixture(autouse=True)
def isolated_local_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGBENCH_ALLOW_INSECURE_LOCAL", "true")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "default-store"))
    monkeypatch.delenv("DATABASE_URL", raising=False)
