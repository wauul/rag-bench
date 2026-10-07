"""Exercise the documented Render API contract without cloud credentials or paid resources."""

import httpx
import pytest

from scripts.deploy import deploy

IMAGE = "ghcr.io/wauul/ragbench-compact@sha256:" + "a" * 64


def test_render_digest_verified_and_source_autodeploy_rejected():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if request.method == "POST":
            assert b'"imageUrl"' in request.content
            return httpx.Response(201, json={"id": "dep-test"})
        if "/deploys/" in request.url.path:
            return httpx.Response(
                200, json={"status": "live", "image": {"ref": IMAGE, "sha": "sha256:" + "a" * 64}}
            )
        return httpx.Response(200, json={"imagePath": IMAGE, "autoDeploy": "no"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = deploy(client, "srv-test", IMAGE)
    assert result["image"] == IMAGE and len(paths) == 3
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"repo": "git"}))
    ) as client:
        with pytest.raises(ValueError, match="prebuilt"):
            deploy(client, "srv-test", IMAGE)
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"imagePath": IMAGE, "autoDeploy": "yes"})
        )
    ) as client:
        with pytest.raises(ValueError, match="autoDeploy"):
            deploy(client, "srv-test", IMAGE)


def test_render_rejects_mismatched_digest():
    def handler(request):
        if request.method == "POST":
            return httpx.Response(201, json={"id": "dep-test"})
        if "/deploys/" in request.url.path:
            return httpx.Response(
                200, json={"status": "live", "image": {"ref": IMAGE, "sha": "b" * 64}}
            )
        return httpx.Response(200, json={"imagePath": IMAGE, "autoDeploy": "no"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="resolved"):
            deploy(client, "srv-test", IMAGE)
