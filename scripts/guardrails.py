"""Operator-only policy management; no public API can increase workspace caps."""

import argparse
import json
from datetime import datetime, timezone

from dotenv import load_dotenv

from backend.guardrails import Policy, locked, persist
from backend.storage import Store


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", action="append", default=[], metavar="FIELD=JSON_VALUE")
    args = parser.parse_args()
    store = Store()
    with locked(store) as (db, state, policy):
        changes = {}
        for assignment in args.set:
            name, value = assignment.split("=", 1)
            changes[name] = json.loads(value)
        state["policy"] = Policy.model_validate({**policy.model_dump(), **changes}).model_dump()
        if changes:
            state["audit"] = [
                *state.get("audit", []),
                {"at": datetime.now(timezone.utc).isoformat(), "changes": changes},
            ][-50:]
        persist(store, db, state)
        print(json.dumps({"policy": state["policy"], "usage": state["usage"]}, indent=2))


if __name__ == "__main__":
    main()
