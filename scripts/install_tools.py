"""Install pinned official tooling into .tools/bin after verifying checked-in SHA256."""

import hashlib
import io
import json
import os
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse


def install():
    destination = Path(".tools/bin")
    destination.mkdir(parents=True, exist_ok=True)
    platform = "windows" if os.name == "nt" else "linux"
    for name, definition in json.loads(Path("scripts/tools.json").read_text()).items():
        asset = definition["platforms"][platform]
        binary_name = name + (".exe" if os.name == "nt" else "")
        parsed = urlparse(asset["url"])
        if parsed.scheme != "https" or parsed.hostname != "github.com":
            raise ValueError("Tool downloads must use official GitHub HTTPS release URLs")
        # HTTPS source validated above; archive digest checked below.
        with urllib.request.urlopen(asset["url"], timeout=120) as response:  # nosec B310
            data = response.read()
        if hashlib.sha256(data).hexdigest() != asset["sha256"]:
            raise ValueError("Tool archive checksum mismatch: " + name)
        # Extract only the named binary, never archive paths or symlinks.
        if asset["url"].endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                content = archive.read(binary_name)
        else:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
                member = next(m for m in archive.getmembers() if Path(m.name).name == binary_name)
                if not member.isfile():
                    raise ValueError("Tool binary must be a regular file")
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError("Tool binary missing")
                content = handle.read()
        target = destination / binary_name
        target.write_bytes(content)
        target.chmod(0o755)
        print(f"Installed {name} {definition['version']} from verified archive")


if __name__ == "__main__":
    install()
