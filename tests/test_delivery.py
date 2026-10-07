"""Exercise the documented Render API contract without cloud credentials or paid resources."""

import httpx
import pytest

from scripts.deploy import deploy
from scripts.smoke import smoke

IMAGE = "ghcr.io/wauul/ragbench-compact@sha256:" + "a" * 64


@pytest.mark.parametrize("gateway", ["guarded", "public", "wrong-host", "wrong-return"])
def test_private_dashboard_requires_exact_authentication_gateway(monkeypatch, gateway):
    from urllib.parse import urlencode

    monkeypatch.setenv("API_TOKEN", "synthetic")
    monkeypatch.setenv("DASHBOARD_ACCESS", "streamlit-private")

    def handler(request):
        if request.url.host == "api.example":
            if request.url.path == "/health":
                return httpx.Response(
                    200, json={"revision": "tested", "authentication_required": True}
                )
            if request.url.path == "/ready":
                return httpx.Response(200, json={"status": "ready"})
            return httpx.Response(
                200 if request.headers.get("Authorization") else 401, json={"runs": []}
            )
        host = "evil.example" if gateway == "wrong-host" else "share.streamlit.io"
        target = "https://wrong.example/" if gateway == "wrong-return" else str(request.url)
        return httpx.Response(
            200 if gateway == "public" else 303,
            headers={
                "Location": f"https://{host}/-/auth/app?{urlencode({'redirect_uri': target})}"
            },
        )

    original = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: original(transport=httpx.MockTransport(handler), **kw)
    )
    if gateway == "guarded":
        smoke("https://api.example", "https://dashboard.example", "tested")
    else:
        with pytest.raises(AssertionError):
            smoke("https://api.example", "https://dashboard.example", "tested")


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
