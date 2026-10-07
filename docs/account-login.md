# Account sign-in

GitHub and email/password sign-in use managed Neon Auth on the existing database
branch. Ragbench never stores passwords, email addresses, provider JWTs or OAuth
refresh tokens. Neon manages those records in its private `neon_auth` schema;
Resend delivers verification and password-reset email. Review both providers'
retention policies separately from Ragbench's experiment retention.

The Streamlit server creates a ten-minute sign-in flow with a SHA-256 challenge.
A random, ten-minute Secure/SameSite browser nonce binds the return to that same
browser. A separate top-level sign-in tab handles Neon cookies and OAuth.
The welcome screen offers GitHub and email directly, without a preliminary
"continue" step. GitHub creates an account on first use. Email signup and reset
remain available on the email screen. The OAuth return exchanges Neon's
`neon_auth_session_verifier` with `/get-session` before requesting identity proof,
then removes the consumed verifier from the URL. This matches the managed
client's [getSession contract](https://github.com/neondatabase/neon-js/blob/main/packages/auth/src/core/better-auth-methods.ts).
Finishing requires the pinned provider's signed JWT, exact issuer/audience,
verified active database account, exact Origin, and matching HttpOnly gateway
cookie. No identity is accepted from email or user-editable metadata.

Only the browser that presents identity proof receives a random completion
ticket. Its hash is stored, it expires after three minutes, and it is redeemed
once together with the original dashboard browser nonce. A forwarded sign-in
link, initiator-only polling, or a forwarded completion link cannot grant a
session in another browser. The callback query is cleared before exchange.
Actual API sessions remain server-side in Streamlit and only their SHA-256 hashes
are stored in PostgreSQL. Every API request rechecks expiry, account enablement,
verification and provider ban state. Logout revokes that dashboard session.

Sign-in/provider errors fail closed. Public handoff routes use durable socket-peer
and global throttles and bounded request bodies. Unknown accounts get separate
stable ownership IDs; they never inherit the owner's existing records.

Optional API/CLI keys expire in 30 days, are stored only as hashes and can be
rotated/revoked in the sidebar. They remain bound to the managed account's active
state. Legacy operator-provisioned keys and owner recovery are still supported.
Never share the owner password or owner API token with regular users.

Backend settings: `NEON_AUTH_URL`, `AUTH_GATEWAY_URL`, `DASHBOARD_URL` (the exact
HTTPS Streamlit app origin), and pinned
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
