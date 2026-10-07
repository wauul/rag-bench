# Delivery and rollback

## Verified hosting

On 2026-10-07 Render `rag-bench-api` (`srv-daklmbgae00c73bsdptg`) was a **Free, source-backed Docker** service for `wauul/rag-bench`, branch main, Dockerfile `./Dockerfile.render`, region Oregon, health path `/health`. Its live revision was `d5ea194cf7a62d623e37dfb046f9b915a3920eb9`; Render visibly reported auto-deploy disabled after a specific-commit deployment. This is evidence for that revision, not the changed tree. Backend URL: https://rag-bench-api.onrender.com. Dashboard remains Streamlit Community Cloud at https://wauul-rag-bench-dashboardapp-peuvxw.streamlit.app. No hosting/storage migration or paid resource was provisioned.

The prior deployment script only supported prebuilt Render images. It now supports exact `commitId` source deployment with API confirmation of the resulting commit. `--backend-mode image` remains supported only for an already image-backed service. Source rebuilding cannot prove byte identity with a signed GHCR image. The manifest identifies the tested artifacts; evidence labels source rebuilds `provider-rebuild-digest-unavailable`. Neither Render nor Streamlit offers an assumed OIDC credential path here: Render uses a backend-scoped account API key; GHCR attestations use GitHub OIDC.

## Activation prerequisites

1. Keep Render auto-deploy **Off**, one backend worker, `DATABASE_URL` configured with a direct TLS Neon URL. Never clear it for rollback. Verify `/health` still reports postgres and authenticated `/ready` succeeds.
2. Configure Streamlit to a **dedicated protected dashboard release branch** pointing initially at the known deployed revision. Only advance that branch to a quality-approved release, after checks and human review. Tracking main automatically can otherwise bypass the quality gate. Branch selection in the hosted app and its resulting build revision need operator verification; a workflow alone cannot change this hosting configuration.
3. Configure protected production secrets `RENDER_API_KEY`, `API_TOKEN`; variables `RENDER_API_SERVICE_ID`, `API_URL`, `DASHBOARD_URL`, `DASHBOARD_APPROVED_REVISION`. The last value must be the exact verified dashboard build commit, never a speculative candidate. Streamlit secrets include matching `API_TOKEN`, `BACKEND_URL`, production password and environment settings. Groq, Neon and Langfuse credentials stay backend-side.
4. Publish a protected tag release. Review repeated independent evaluations and commit `evaluations/promotion-policy.json` plus `evaluations/promotions/<revision>.json`. No fabricated default thresholds or smoke-to-promotion conversion is provided.
5. Approve dashboard branch promotion, inspect Streamlit build logs and exercise dashboard login/history against the authenticated backend. Then dispatch Deploy approved release with `backend_mode=source` and the immutable version tag. Deployment concurrency is serialized and checks use no provider calls.

```powershell
python -m scripts.quality_gate evaluations/promotions/REVISION.json evaluations/promotion-policy.json --revision REVISION --image ghcr.io/wauul/rag-bench-compact@sha256:DIGEST
python -m scripts.deploy release.json --backend-mode source --dashboard-mode streamlit --evidence reports/deployment.json
python -m scripts.smoke --api https://rag-bench-api.onrender.com --dashboard https://wauul-rag-bench-dashboardapp-peuvxw.streamlit.app --revision REVISION
```

Smoke verifies API revision, PostgreSQL pre/post deployment, authentication, readiness, read-only history and Streamlit health. It does not establish hosted UI-to-API connectivity; that needs the dashboard login/history check. Container smoke separately exercises the actual dashboard and API connection using Streamlit AppTest. Free-tier cold starts may require a later operator retry; do not run keep-alive polls.

## Rollback

Retain the current and candidate release manifests. Before rollback compare `storage_schema`, `storage_backends` and `checkpoint_serializer`; the script rejects mismatches. Manifest equality is a necessary compatibility check, not proof that old code can interpret new immutable snapshots. Provenance includes exact backend source/dependencies/prompt fingerprints: interrupted work may need the original revision/environment to resume. Keep those releases and their snapshots; never rewrite old provenance to force compatibility.

Back up PostgreSQL during a stopped-write recovery window. Promote the previous compatible dashboard branch/revision, verify the build, then deploy the previous manifest:

```powershell
python -m scripts.deploy previous-release.json --previous-manifest current-release.json --backend-mode source --dashboard-mode streamlit --evidence reports/rollback.json
```

A SQLite-only manifest is rejected. Removing `DATABASE_URL` creates a separate empty store and is never data recovery. Database restoration is a separate reviewed operation into an empty destination; update production connection settings only after validating recovery. Do not overwrite Neon records to roll code back.
