"""Keep pip/Streamlit compatibility files generated from uv.lock; never edit them by hand."""

import argparse
import subprocess
from pathlib import Path

EXPORTS = {
    "backend/requirements-core.txt": ["backend"],
    "backend/requirements.txt": ["backend", "cpu"],
    "backend/requirements-compact.txt": ["backend", "compact"],
    "dashboard/requirements.txt": ["dashboard"],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, extras in EXPORTS.items():
        command = ["uv", "export", "--locked", "--no-dev", "--no-emit-project", "--no-header"]
        for extra in extras:
            command.extend(["--extra", extra])
        result = subprocess.run(command, check=True, capture_output=True, text=True).stdout
        if args.check:
            if Path(name).read_text(encoding="utf-8") != result:
                raise SystemExit(
                    f"{name} differs from uv.lock; run python -m scripts.export_requirements"
                )
        else:
            Path(name).write_text(result, encoding="utf-8")
