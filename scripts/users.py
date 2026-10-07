"""Operator-only personal access keys. Secrets are generated once, stored as SHA-256."""

import argparse
import hashlib
import re
import secrets
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

from backend.storage import Store


def provision(store, user, key, days=30):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", user) or user == "owner":
        raise ValueError("Choose a non-owner user identifier")
    if not 1 <= days <= 90 or len(key) < 32:
        raise ValueError("Keys require at least 32 characters and 1-90 days validity")
    expires = datetime.now(timezone.utc) + timedelta(days=days)
    with store.connect(operator=True) as db:
        db.execute(
            "INSERT INTO users(id,token_hash,expires_at) VALUES (%s,%s,%s) ON CONFLICT(id) DO UPDATE SET token_hash=EXCLUDED.token_hash,expires_at=EXCLUDED.expires_at,enabled=true"
            if store.postgres
            else "INSERT INTO users(id,token_hash,expires_at) VALUES (?,?,?) ON CONFLICT(id) DO UPDATE SET token_hash=excluded.token_hash,expires_at=excluded.expires_at,enabled=1",
            (user, hashlib.sha256(key.encode()).hexdigest(), expires.isoformat()),
        )


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("user")
    parser.add_argument("--revoke", action="store_true")
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    store = Store()
    if args.revoke:
        with store.connect(operator=True) as db:
            db.execute(
                "UPDATE users SET enabled=false WHERE id=%s"
                if store.postgres
                else "UPDATE users SET enabled=0 WHERE id=?",
                (args.user,),
            )
        print("Access revoked")
    else:
        key = secrets.token_urlsafe(32)
        provision(store, args.user, key, args.days)
        print(key)


if __name__ == "__main__":
    main()
