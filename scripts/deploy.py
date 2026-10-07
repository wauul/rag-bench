"""Deploy exact release images to existing Render services; never provision resources."""

import argparse
import json
import os
import re
import time
from pathlib import Path

import httpx

from scripts.smoke import smoke

API = "https://api.render.com/v1"


def deploy(client: httpx.Client, service: str, image: str) -> dict:
    if not re.fullmatch(r"srv-[a-z0-9]+", service):
        raise ValueError("Invalid Render service ID")
    if not re.fullmatch(r"ghcr.io/[a-z0-9_./-]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("Deployment requires an immutable GHCR digest")
    response = client.get(f"{API}/services/{service}")
    response.raise_for_status()
    configuration = response.json()
    # Source-backed services can bypass CI through auto-deploy or rebuild a different artifact.
    if not configuration.get("imagePath"):
        raise ValueError("Render service must use a prebuilt image")
    if configuration.get("autoDeploy") != "no":
        raise ValueError("Disable Render autoDeploy before deployment")
    response = client.post(f"{API}/services/{service}/deploys", json={"imageUrl": image})
    response.raise_for_status()
    identity = response.json()["id"]
    for _ in range(120):
        response = client.get(f"{API}/services/{service}/deploys/{identity}")
        response.raise_for_status()
        state = response.json()
        if state["status"] == "live":
            if state.get("image", {}).get("ref") != image:
                raise ValueError("Render deployment digest differs from release")
            if (
                state.get("image", {}).get("sha", "").removeprefix("sha256:")
                != image.split("sha256:")[1]
            ):
                raise ValueError("Render resolved image digest differs from release")
            return {"service": service, "deploy": identity, "image": image}
        if state["status"] in {
            "build_failed",
            "update_failed",
            "canceled",
            "deactivated",
            "pre_deploy_failed",
        }:
            raise RuntimeError("Render deployment failed: " + state["status"])
        time.sleep(10)
    raise TimeoutError("Render deployment did not become live within 20 minutes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--evidence", type=Path, default=Path("deployment.json"))
    args = parser.parse_args()
    release = json.loads(args.manifest.read_text(encoding="utf-8"))
    with httpx.Client(
        headers={"Authorization": "Bearer " + os.environ["RENDER_API_KEY"]}, timeout=60
    ) as client:
        evidence = [
            deploy(client, os.environ["RENDER_API_SERVICE_ID"], release["images"]["compact"]),
            deploy(
                client, os.environ["RENDER_DASHBOARD_SERVICE_ID"], release["images"]["dashboard"]
            ),
        ]
    smoke(os.environ["API_URL"], os.environ["DASHBOARD_URL"], release["revision"])
    args.evidence.write_text(
        json.dumps({"revision": release["revision"], "deployments": evidence}, indent=2),
        encoding="utf-8",
    )
    print("Deployment verified; image digests and source revision recorded")


if __name__ == "__main__":
    main()
