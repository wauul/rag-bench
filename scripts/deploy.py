"""Deploy exact release images to existing Render services; never provision resources."""

import argparse
import json
import os
import re
import time
from pathlib import Path

import httpx

from scripts.release_manifest import validate
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
    if (
        configuration.get(
            "autoDeployTrigger", "off" if configuration.get("autoDeploy") == "no" else "commit"
        )
        != "off"
    ):
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


def deploy_source(client: httpx.Client, service: str, revision: str) -> dict:
    if not re.fullmatch(r"srv-[a-z0-9]+", service) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Source deployment needs an exact revision and service identity")
    response = client.get(f"{API}/services/{service}")
    response.raise_for_status()
    configuration = response.json()
    if (
        configuration.get("imagePath")
        or configuration.get("repo", "").removesuffix(".git")
        != "https://github.com/wauul/rag-bench"
    ):
        raise ValueError("Expected the existing source-backed Ragbench service")
    if (
        configuration.get(
            "autoDeployTrigger", "off" if configuration.get("autoDeploy") == "no" else "commit"
        )
        != "off"
    ):
        raise ValueError("Disable source auto-deploy before CI-controlled deployment")
    response = client.post(f"{API}/services/{service}/deploys", json={"commitId": revision})
    response.raise_for_status()
    identity = response.json()["id"]
    for _ in range(120):
        response = client.get(f"{API}/services/{service}/deploys/{identity}")
        response.raise_for_status()
        state = response.json()
        if state["status"] == "live":
            if state.get("commit", {}).get("id") != revision:
                raise ValueError("Provider rebuilt a different revision")
            return {
                "service": service,
                "deploy": identity,
                "revision": revision,
                "artifact": "provider-rebuild-digest-unavailable",
            }
        if state["status"] in {
            "build_failed",
            "update_failed",
            "canceled",
            "deactivated",
            "pre_deploy_failed",
        }:
            raise RuntimeError("Source deployment failed")
        time.sleep(10)
    raise TimeoutError("Source deployment exceeded 20 minutes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--evidence", type=Path, default=Path("deployment.json"))
    parser.add_argument("--backend-mode", choices=["source", "image"], default="source")
    parser.add_argument("--dashboard-mode", choices=["streamlit", "render"], default="streamlit")
    parser.add_argument("--previous-manifest", type=Path)
    args = parser.parse_args()
    release = validate(json.loads(args.manifest.read_text(encoding="utf-8")))
    if args.previous_manifest:
        previous = validate(json.loads(args.previous_manifest.read_text(encoding="utf-8")))
        for key in ("storage_schema", "checkpoint_serializer", "storage_backends"):
            if release[key] != previous[key]:
                raise ValueError("Rollback needs an explicit storage/checkpoint migration review")
    if (
        args.dashboard_mode == "streamlit"
        and os.getenv("DASHBOARD_APPROVED_REVISION") != release["revision"]
    ):
        raise ValueError(
            "Promote the Streamlit release branch and verify its rebuild revision first"
        )
    with httpx.Client(timeout=30, follow_redirects=False) as preflight:
        health = preflight.get(os.environ["API_URL"] + "/health")
        health.raise_for_status()
        if health.json().get("storage") != "postgres":
            raise ValueError("Existing Neon deployment must still use PostgreSQL before promotion")
        ready = preflight.get(
            os.environ["API_URL"] + "/ready",
            headers={"Authorization": "Bearer " + os.environ["API_TOKEN"]},
        )
        ready.raise_for_status()
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(
        headers={"Authorization": "Bearer " + os.environ["RENDER_API_KEY"]}, timeout=60
    ) as client:
        evidence = [
            deploy(client, os.environ["RENDER_API_SERVICE_ID"], release["images"]["compact"])
            if args.backend_mode == "image"
            else deploy_source(client, os.environ["RENDER_API_SERVICE_ID"], release["revision"])
        ]
        if args.dashboard_mode == "render":
            evidence.append(
                deploy(
                    client,
                    os.environ["RENDER_DASHBOARD_SERVICE_ID"],
                    release["images"]["dashboard"],
                )
            )
        else:
            evidence.append(
                {
                    "provider": "streamlit-community-cloud",
                    "revision": release["revision"],
                    "artifact": "provider-rebuild-digest-unavailable",
                    "revision_evidence": "operator-confirmed-dashboard-build-log",
                }
            )
    smoke(os.environ["API_URL"], os.environ["DASHBOARD_URL"], release["revision"])
    with httpx.Client(timeout=30) as client:
        if client.get(os.environ["API_URL"] + "/health").json().get("storage") != "postgres":
            raise ValueError("Existing Neon deployment must still use PostgreSQL")
    args.evidence.write_text(
        json.dumps({"revision": release["revision"], "deployments": evidence}, indent=2),
        encoding="utf-8",
    )
    print("Deployment readiness and source verified; artifact/rebuild limitations recorded")


if __name__ == "__main__":
    main()
