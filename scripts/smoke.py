"""Authenticated provider-free smoke checks against a deployed API and dashboard."""

import argparse
import os
from urllib.parse import parse_qs, urlparse

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
        access = os.getenv("DASHBOARD_ACCESS", "public")
        if access == "streamlit-private":
            for path in ("/", "/_stcore/health"):
                response = client.get(dashboard + path)
                destination = urlparse(response.headers.get("location", ""))
                assert response.status_code == 303, "Private dashboard is not access guarded"
                assert (
                    destination.scheme == "https"
                    and destination.hostname == "share.streamlit.io"
                    and destination.path == "/-/auth/app"
                    and parse_qs(destination.query).get("redirect_uri") == [dashboard + path]
                ), "Unexpected dashboard authentication destination"
        else:
            assert access == "public", "Unknown dashboard access mode"
            client.get(dashboard + "/_stcore/health").raise_for_status()
    print("Authenticated API readiness, revision and history passed")
    print(
        "Private dashboard gateway passed; build/health requires recorded owner UI verification"
        if access == "streamlit-private"
        else "Dashboard health passed"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", required=True)
    parser.add_argument("--dashboard", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    smoke(args.api.rstrip("/"), args.dashboard.rstrip("/"), args.revision)


if __name__ == "__main__":
    main()
