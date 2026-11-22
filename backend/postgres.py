"""Short-lived TLS connections allow Neon to sleep between requests and runs."""

from urllib.parse import urlsplit

import psycopg


def validate_database_url(url):
    parts = urlsplit(url)
    if (
        parts.scheme not in {"postgres", "postgresql"}
        or not parts.hostname
        or not parts.path.strip("/")
    ):
        raise ValueError("DATABASE_URL must be a PostgreSQL connection URL")
    # Use libpq's effective settings so query options cannot override validation.
    try:
        options = psycopg.conninfo.conninfo_to_dict(url)
    except psycopg.ProgrammingError:
        raise ValueError("Invalid PostgreSQL connection URL") from None
    hosts = options.get("host", "").lower().split(",")
    addresses = options.get("hostaddr", "").split(",")
    if any("-pooler" in host for host in hosts):
        raise ValueError(
            "Use Neon's direct DATABASE_URL, not its pooled endpoint (worker session locks)"
        )
    local = {"localhost", "127.0.0.1", "::1"}
    if not set(hosts) <= local or not set(addresses) <= local | {""}:
        if options.get("sslmode") != "verify-full":
            raise ValueError("Remote DATABASE_URL requires sslmode=verify-full")


def connect(url, **kwargs):
    return psycopg.connect(url, connect_timeout=15, prepare_threshold=None, **kwargs)
