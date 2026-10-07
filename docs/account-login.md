# Account sign-in

GitHub and email/password sign-in use managed Neon Auth on the existing database
branch. Ragbench never stores passwords, email addresses, provider JWTs or OAuth
refresh tokens. Neon manages those records in its private `neon_auth` schema;
Resend delivers verification and password-reset email. Review both providers'
retention policies separately from Ragbench's experiment retention.

The Streamlit server creates a ten-minute sign-in flow with a SHA-256 challenge.
Its random verifier remains in that browser's server-side Streamlit session. A
separate top-level sign-in tab handles Neon cookies and OAuth. Finishing requires
the pinned provider's signed JWT, exact issuer/audience, verified active database
account, exact Origin, and the matching HttpOnly browser cookie. No identity is
accepted from email, browser-submitted profile claims, or user-editable metadata.

Streamlit polls using its private verifier, atomically redeems the result once,
and receives a random one-hour session. Only its SHA-256 hash is stored. Every API
request rechecks expiry, account enablement, verification and provider ban state.
Logout revokes that dashboard session; revoked sessions cannot make further API
requests. Sign-in/provider errors fail closed. Public handoff routes use durable
socket-peer and global throttles and bounded request bodies. Unknown accounts get
separate stable ownership IDs; they never inherit the owner's existing records.

Optional API/CLI keys expire in 30 days, are stored only as hashes and can be
rotated/revoked in the sidebar. They remain bound to the managed account's active
state. Legacy operator-provisioned keys and owner recovery are still supported.
Never share the owner password or owner API token with regular users.

Backend settings: `NEON_AUTH_URL`, `AUTH_GATEWAY_URL`, and pinned
`NEON_AUTH_ISSUER` / `NEON_AUTH_AUDIENCE` where needed. Dashboard setting:
`ACCOUNT_LOGIN_ENABLED=true`. Enable verified email, custom SMTP and the GitHub
OAuth provider in Neon; use only the exact gateway origin in trusted domains and
disable localhost. Configure GitHub's callback as
`NEON_AUTH_URL/callback/github`. OAuth client secrets and SMTP credentials belong
only in provider settings, never Git or client code. Benchmark quota, ownership,
RLS, deletion and retention controls continue to apply to managed accounts.

Neon's auth endpoints have their own provider limits. Ragbench admission caps
govern benchmark work, not all direct requests to Neon's public signup service.
Keep Resend's sending limits enabled and monitor its transactional email usage.
