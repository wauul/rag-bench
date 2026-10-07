# CI, security and release gates

`uv.lock` is authoritative. Python 3.11 is the tested runtime; CPU PyTorch stays on its dedicated CPU index. Generated requirements are compatibility exports, checked by `scripts.checks`, not a second dependency policy.

## Flow

PR â†’ Engineering (Windows/Linux quality, security, CPU/compact containers, PostgreSQL) â†’ **Engineering gate** â†’ protected main. Real local models remain a separate, credential-free workflow. An immutable `vX.Y.Z` tag on main reruns Engineering and real models, builds and tests actual CPU/compact/dashboard images, scans them, produces SPDX SBOMs, then waits for the `release` environment reviewer. Publishing loads the tested archives instead of rebuilding, pushes version/full-commit tags to GHCR and produces OIDC build/SBOM attestations and `release-manifest`.

Authorized live evaluation uses the `evaluation` environment and a signed release digest. It runs the existing optimizer against isolated fictional sample data with a durable HTTP-attempt ceiling. This exploratory report cannot promote production. Production requires a separately reviewed repeated-baseline policy and complete independent evaluation bound to the exact revision, digest, dataset, prompt, model and evaluator. Missing/stale/incomplete evidence and quota failures fail closed.

Manual deployment downloads only a successful tag Release run from this repository, verifies attestations and measured quality, then waits for the `production` environment reviewer. It uses existing Render source hosting and Streamlit hosting; provider rebuilds are recorded as such. See [deployment](deployment.md) for the dashboard release-branch prerequisite and external setup.

## Checks and trust boundaries

`Engineering gate` keeps its stable name and explicitly rejects failed, cancelled and unexpectedly skipped dependencies. PostgreSQL tests use disposable PostgreSQL 17 and cover locks, replacement workers, saved checkpoints, all four workflows, backup/restore and explicit resume. Ordinary CI never receives Groq/Langfuse credentials. Controlled provider fixtures are deterministic regression evidence, not measured Ragas quality.

Actions are pinned to verified official commit SHAs. Workflow permissions default to `contents: read`; publishing alone has packages/attestations/OIDC writes. No `pull_request_target`, untrusted PR artifact download or automatic deployment of optimizer suggestions exists. Release archives expire after 3 days; tests/security/evaluation after 14 days; manifests/digest/deployment evidence after 90 days. Retain a reviewed release manifest and attestations externally for longer rollback needs.

`python -m scripts.checks` validates lock, requirements, installed dependencies, Ruff, scoped mypy, pytest and existing coverage thresholds. Mypy covers changed privacy, provenance, budget, backup and delivery modules without new broad ignores. Bandit, pip-audit, Gitleaks, actionlint and Trivy remain enforced. Existing security exceptions require owner, reason and expiry; they are not permission to use vulnerable HTTP handlers or unsafe caches. Dependabot tracks uv, Docker and Actions weekly.

## Repository settings

Verified owner: `wauul`. Main protection was configured with strict `Engineering gate`, mandatory PRs, conversation resolution, admin enforcement and no force pushes/deletion. Required review count is zero because a solo owner cannot approve their own PR. Release/evaluation/production environments require owner review and restrict deployment refs respectively to tags `v*`, branch `main`, branch `main`. `codex/dashboard-release` was also prepared at the existing deployed revision with strict Engineering gate/admin enforcement and no force pushes/deletion. Restricting individual push users is unavailable on a personal repository; owner access is not presented as organization-level separation. Streamlit still needs an approved coordinate change to use that branch. Environment reviews protect provider-spend/publishing/deployment independently of merge checks.

```powershell
uv sync --locked --extra backend --extra cpu --extra compact --extra dashboard --extra models
uv run --no-sync python -m scripts.checks
python -m scripts.install_tools
.tools/bin/actionlint
.tools/bin/gitleaks git --redact=100 --no-banner
python -m bandit -r backend scripts -ll
python -m scripts.security
```

Official contracts checked: [Actions attest](https://github.com/actions/attest), [Render deployment API](https://api-docs.render.com/reference/create-deploy), [Streamlit source deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [Langfuse Python SDK](https://python.reference.langfuse.com/langfuse).
