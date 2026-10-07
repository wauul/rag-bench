# Delivery and rollback

## Verified hosting

On 2026-10-07 Render `rag-bench-api` (`srv-daklmbgae00c73bsdptg`) was a **Free, source-backed Docker** service for `wauul/rag-bench`, branch main, Dockerfile `./Dockerfile.render`, region Oregon, health path `/health`. Its live revision was `d5ea194cf7a62d623e37dfb046f9b915a3920eb9`; Render visibly reported auto-deploy disabled after a specific-commit deployment. This is evidence for that revision, not the changed tree. Backend URL: https://rag-bench-api.onrender.com. Dashboard remains Streamlit Community Cloud at https://wauul-rag-bench-dashboardapp-peuvxw.streamlit.app. No hosting/storage migration or paid resource was provisioned.

The prior deployment script only supported prebuilt Render images. It now supports exact `commitId` source deployment with API confirmation of the resulting commit. `--backend-mode image` remains supported only for an already image-backed service. Source rebuilding cannot prove byte identity with a signed GHCR image. The manifest identifies the tested artifacts; evidence labels source rebuilds `provider-rebuild-digest-unavailable`. Neither Render nor Streamlit offers an assumed OIDC credential path here: Render uses an account API key covering all owner workspaces (the UI offers no service-only scope); GHCR attestations use GitHub OIDC.

## Activation prerequisites

1. Keep Render auto-deploy **Off**, one backend worker, `DATABASE_URL` configured with a direct TLS Neon URL. Never clear it for rollback. Verify `/health` still reports postgres and authenticated `/ready` succeeds.
2. The protected `codex/dashboard-release` branch was prepared at the known deployed revision and adopted after explicit owner-approved deletion/redeployment on 2026-10-07. The original subdomain, Python 3.11 and privately recovered backend secrets were restored. The dashboard loaded two existing history records against the unchanged Render API. Streamlit viewer access is now private, with no invited viewers; owner/GitHub management access remains. Current Community Cloud has no in-place GitHub-coordinate edit: its [official branch-change procedure](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/rename-your-app) requires deleting and redeploying the app. Future coordinate changes also need owner approval, temporary downtime, private restoration of dashboard secrets and settings, and reuse of the existing subdomain. Never delete the app before those are recoverable. Only advance that branch to a quality-approved release, after checks and human review. Tracking main automatically can otherwise bypass the quality gate. Branch selection in the hosted app and its resulting build revision need operator verification; a workflow alone cannot change this hosting configuration.
3. Configure protected production secrets `RENDER_API_KEY`, `API_TOKEN`; variables `RENDER_API_SERVICE_ID`, `API_URL`, `DASHBOARD_URL`, `DASHBOARD_ACCESS`, `DASHBOARD_APPROVED_REVISION`. The last value must be the exact verified dashboard build commit, never a speculative candidate. The current protected environment has the Render/API secrets and all these variables; approved dashboard revision still names the prior deployed baseline, not this PR. Existing Streamlit secrets include matching `API_TOKEN` and `BACKEND_URL`; native Community Cloud private viewer authentication protects the hosted dashboard. Set `DASHBOARD_ACCESS=streamlit-private` for that gateway. Standalone/public dashboards require `APP_ENV=production` and their own `DASHBOARD_PASSWORD`; do not expose an authenticated backend through an unguarded dashboard. Groq, Neon and Langfuse credentials stay backend-side.
4. Publish a protected tag release. Review repeated independent evaluations and commit `evaluations/promotion-policy.json` plus `evaluations/promotions/<revision>.json`. No fabricated default thresholds or smoke-to-promotion conversion is provided.
5. Approve dashboard branch promotion, inspect Streamlit build logs and exercise dashboard login/history against the authenticated backend. Then dispatch Deploy approved release with `backend_mode=source` and the immutable version tag. Deployment concurrency is serialized and checks use no provider calls.

```powershell
python -m scripts.quality_gate evaluations/promotions/REVISION.json evaluations/promotion-policy.json --revision REVISION --image ghcr.io/wauul/rag-bench-compact@sha256:DIGEST
python -m scripts.deploy release.json --backend-mode source --dashboard-mode streamlit --evidence reports/deployment.json
python -m scripts.smoke --api https://rag-bench-api.onrender.com --dashboard https://wauul-rag-bench-dashboardapp-peuvxw.streamlit.app --revision REVISION
```

Smoke verifies API revision, PostgreSQL pre/post deployment, authentication, readiness and read-only history. Public dashboard mode checks Streamlit health. Private Community Cloud mode instead verifies both root/health return the exact HTTPS Streamlit authentication gateway and return URL; it rejects public responses and unexpected hosts. This gateway check does not prove private application health, which requires the recorded owner UI/build verification and exact `DASHBOARD_APPROVED_REVISION`. It does not establish hosted UI-to-API connectivity; that needs the dashboard login/history check. Container smoke separately exercises the actual dashboard and API connection using Streamlit AppTest. Free-tier cold starts may require a later operator retry; do not run keep-alive polls.

## Rollback

Retain the current and candidate release manifests. Before rollback compare `storage_schema`, `storage_backends` and `checkpoint_serializer`; the script rejects mismatches. Manifest equality is a necessary compatibility check, not proof that old code can interpret new immutable snapshots. Provenance includes exact backend source/dependencies/prompt fingerprints: interrupted work may need the original revision/environment to resume. Keep those releases and their snapshots; never rewrite old provenance to force compatibility.

Back up PostgreSQL during a stopped-write recovery window. Restore the previous compatible dashboard source and verify its build before deploying the previous manifest. Protected release branches cannot be force-rewound. A rollback may require a separate protected branch at the previous release and an approved Streamlit redeployment, or a newly tested forward-revert release with its own manifest:

```powershell
python -m scripts.deploy previous-release.json --previous-manifest current-release.json --backend-mode source --dashboard-mode streamlit --evidence reports/rollback.json
```

A SQLite-only manifest is rejected. Removing `DATABASE_URL` creates a separate empty store and is never data recovery. Database restoration is a separate reviewed operation into an empty destination; update production connection settings only after validating recovery. Do not overwrite Neon records to roll code back.
