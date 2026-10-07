# Contributing

Use Python 3.11 and the checked-in uv lock. Preserve CPU-only PyTorch, pinned local models and generated requirement exports. Read the workflow-specific documents before changing recovery, provenance or accounting. Use synthetic data; never put production credentials/documents in tests.

```powershell
uv sync --locked --extra backend --extra cpu --extra compact --extra dashboard --extra models
uv run --no-sync python -m scripts.export_requirements
uv run --no-sync python -m scripts.checks
```

Keep one API worker. Ordinary tests use controlled provider fixtures and require no paid calls. PostgreSQL tests need RAGBENCH_TEST_POSTGRES_URL pointing to a disposable database and PostgreSQL client tools, or RAGBENCH_PG_DOCKER=1. Real local-model tests explicitly require RAGBENCH_TEST_REAL_MODELS=1. Live evaluation requires explicit spend approval and a bounded optimizer plan. Never reinterpret fixtures as measured answer quality.

Explain behavior, recovery/privacy/accounting impact and verification in each PR. Do not lower coverage thresholds, silently substitute models, loosen snapshot validation or update security exceptions just to pass. Source/prompt changes can make interrupted snapshots incompatible; document a recoverable original environment. Follow [CI](docs/cicd.md), [release policy](docs/release-policy.md) and [security reporting](SECURITY.md).
