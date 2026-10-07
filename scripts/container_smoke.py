"""Test actual artifacts: UID, HTTP auth, dashboard API connection and restart persistence."""

import argparse
import subprocess
import time
from uuid import uuid4

TOKEN = "container-test-only-token-32-characters"
PASSWORD = "container-test-only-password-32-characters"


def docker(*args, capture=True):
    return subprocess.run(
        ["docker", *args], check=True, capture_output=capture, text=True
    ).stdout.strip()


def wait_healthy(name):
    for _ in range(60):
        status = docker("inspect", "--format", "{{.State.Health.Status}}", name)
        if status == "healthy":
            return
        if status == "unhealthy":
            raise RuntimeError("Container failed its healthcheck: " + name)
        time.sleep(2)
    raise TimeoutError("Container did not become healthy: " + name)


def smoke(api_image, dashboard_image):
    identity = "ragbench-smoke-" + uuid4().hex[:12]
    api, dashboard = identity + "-api", identity + "-dashboard"
    volume, cache, network = identity + "-data", identity + "-cache", identity + "-network"
    created = []
    try:
        docker("network", "create", network)
        created.append(("network", network))
        for name in (volume, cache):
            docker("volume", "create", name)
            created.append(("volume", name))
        docker(
            "run",
            "-d",
            "--name",
            api,
            "--network",
            network,
            "--network-alias",
            "api",
            "--mount",
            f"type=volume,source={volume},target=/app/data",
            "--mount",
            f"type=volume,source={cache},target=/app/cache",
            "-e",
            "APP_ENV=production",
            "-e",
            "DATA_STORAGE=persistent",
            "-e",
            "API_TOKEN=" + TOKEN,
            "-e",
            "PORT=8099",
            api_image,
        )
        created.append(("container", api))
        wait_healthy(api)
        assert docker("exec", api, "id", "-u") == "10001"
        code = """import json,os,urllib.request,urllib.error
base='http://127.0.0.1:8099'
try: urllib.request.urlopen(base+'/api/runs'); raise RuntimeError('Auth bypass')
except urllib.error.HTTPError as error:
 if error.code!=401: raise
headers={'Authorization':'Bearer '+os.environ['API_TOKEN']}
def call(path,data=None):
 return json.load(urllib.request.urlopen(urllib.request.Request(base+path,data=data,headers=headers)))
assert call('/ready')['status']=='ready'
fixture=call('/api/demo',b'')
from backend.storage import Store
store=Store()
saved=store.save('run',{'status':'running','stage':'Smoke fixture','rows':[],'configurations':[], 'questions':[], 'created_at':'2026-10-07', 'total':0, 'completed':0})
open('/app/data/smoke-id','w').write(saved['id'])
assert os.access('/app/cache',os.W_OK)
print('auth/readiness/demo/non-root-volume-write passed')
"""
        print(docker("exec", api, "python", "-c", code))
        docker("restart", "--time", "180", api)
        wait_healthy(api)
        print(
            docker(
                "exec",
                api,
                "python",
                "-c",
                "from backend.storage import Store; from pathlib import Path; s=Store(); r=s.get('run',Path('/app/data/smoke-id').read_text()); assert r['status']=='failed'; assert s.list('documents'); print('restart persistence/interruption passed')",
            )
        )
        docker(
            "run",
            "-d",
            "--name",
            dashboard,
            "--network",
            network,
            "-e",
            "BACKEND_URL=http://api:8099",
            "-e",
            "API_TOKEN=" + TOKEN,
            "-e",
            "APP_ENV=production",
            "-e",
            "DASHBOARD_PASSWORD=" + PASSWORD,
            dashboard_image,
        )
        created.append(("container", dashboard))
        wait_healthy(dashboard)
        print(
            docker(
                "exec",
                dashboard,
                "python",
                "-c",
                """import os
from streamlit.testing.v1 import AppTest
app=AppTest.from_file('dashboard/app.py',default_timeout=30).run()
assert not app.exception
app.text_input[0].set_value(os.environ['DASHBOARD_PASSWORD'])
app.button[0].click().run()
assert not app.exception
app.radio(key='page').set_value('History').run()
assert not app.exception and not app.error
assert app.session_state['authenticated']
print('dashboard login/render/authenticated API history passed')
""",
            )
        )
        docker("stop", "--time", "180", api)
        docker(
            "run",
            "--rm",
            "--mount",
            f"type=volume,source={volume},target=/app/data",
            api_image,
            "python",
            "-m",
            "scripts.backup",
            "backup",
            "/app/data",
            "/home/app/smoke-backup",
        )
        print("Offline container backup passed")
    finally:
        for kind, name in reversed(created):
            command = ["docker", kind, "rm"]
            if kind == "container":
                command.append("-f")
            subprocess.run([*command, name], check=False, capture_output=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", required=True)
    parser.add_argument("--dashboard", required=True)
    args = parser.parse_args()
    smoke(args.api, args.dashboard)
