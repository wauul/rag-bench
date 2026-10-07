"""Short-lived TLS connections allow Neon to sleep between requests and runs."""

from urllib.parse import parse_qs, urlsplit

import psycopg


def validate_database_url(url):
    parts = urlsplit(url)
    if (
        parts.scheme not in {"postgres", "postgresql"}
        or not parts.hostname
        or not parts.path.strip("/")
    ):
        raise ValueError("DATABASE_URL must be a PostgreSQL connection URL")
    if "-pooler" in parts.hostname:
        raise ValueError(
            "Use Neon's direct DATABASE_URL, not its pooled endpoint (worker session locks)"
        )
    if parts.hostname not in {"localhost", "127.0.0.1", "::1"}:
        if parse_qs(parts.query).get("sslmode", [""])[0] not in {
            "require",
            "verify-ca",
            "verify-full",
        }:
            raise ValueError("Remote DATABASE_URL requires sslmode=require or stricter")


def connect(url, **kwargs):
    return psycopg.connect(url, connect_timeout=15, prepare_threshold=None, **kwargs)
