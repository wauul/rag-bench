"""Authenticated provider-free smoke checks against a deployed API and dashboard."""

import argparse
import os

import httpx


def smoke(api: str, dashboard: str, revision: str) -> None:
    token = os.environ["API_TOKEN"]
    with httpx.Client(timeout=30, follow_redirects=False) as client:
        health = client.get(api + "/health")
        health.raise_for_status()
        assert health.json()["revision"] == revision, "Unexpected source revision"
        assert health.json()["authentication_required"], "Authentication is disabled"
        assert client.get(api + "/api/runs").status_code == 401, "Unauthenticated access allowed"
        headers = {"Authorization": "Bearer " + token}
        ready = client.get(api + "/ready", headers=headers)
        ready.raise_for_status()
        assert ready.json()["status"] == "ready"
        response = client.get(api + "/api/runs?limit=1", headers=headers)
        response.raise_for_status()
        assert isinstance(response.json()["runs"], list)
        client.get(dashboard + "/_stcore/health").raise_for_status()
    print("Authenticated readiness, revision, API history and dashboard health passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", required=True)
    parser.add_argument("--dashboard", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    smoke(args.api.rstrip("/"), args.dashboard.rstrip("/"), args.revision)


if __name__ == "__main__":
    main()
