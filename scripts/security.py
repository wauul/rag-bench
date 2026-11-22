"""Fail on unexcepted dependency vulnerabilities and expired exceptions."""

import argparse
import json
import subprocess
import sys
from datetime import date
from importlib.metadata import distributions
from pathlib import Path
from tempfile import TemporaryDirectory


def allowed_findings():
    findings = {}
    for item in json.loads(Path("security/exceptions.json").read_text()):
        if date.fromisoformat(item["expires"]) < date.today():
            raise ValueError("Expired security exception: " + item["id"])
        if not item["owner"] or not item["justification"]:
            raise ValueError("Security exception needs an owner and justification")
        for identity in [item["id"], *item.get("aliases", [])]:
            findings[(item["package"], identity)] = item
    return findings


def check_dependencies(path: Path):
    allowed = allowed_findings()
    report = json.loads(path.read_text())
    if report.get("fixes"):
        raise ValueError("Audit must not modify dependencies")
    failures = set()
    for package in report["dependencies"]:
        if package["name"] == "torch" and package.get("skip_reason"):
            failures.add("torch: advisory check unavailable")
        for finding in package.get("vulns", []):
            if (package["name"], finding["id"]) not in allowed:
                failures.add(package["name"] + ": " + finding["id"])
    if failures:
        raise ValueError("Unexcepted vulnerabilities: " + ", ".join(sorted(failures)))
    print("Dependency audit passed the documented exception policy")


def check_container(path: Path):
    allowed = allowed_findings()
    failures = []
    for result in json.loads(path.read_text()).get("Results", []):
        for item in result.get("Vulnerabilities", []):
            # Actionable OS/library findings: fixed HIGH/CRITICAL; Python audit covers every severity.
            if item["Severity"] in {"HIGH", "CRITICAL"} and item.get("FixedVersion"):
                if (item["PkgName"], item["VulnerabilityID"]) not in allowed:
                    failures.append(item["PkgName"] + ": " + item["VulnerabilityID"])
    if failures:
        raise ValueError(
            "Actionable container vulnerabilities: " + ", ".join(sorted(set(failures)))
        )
    print("Container scan passed the actionable finding policy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", type=Path)
    args = parser.parse_args()
    Path("reports").mkdir(exist_ok=True)
    if args.container:
        check_container(args.container)
    else:
        report = Path("reports/dependencies.json")
        with TemporaryDirectory() as directory:
            requirements = Path(directory) / "installed.txt"
            scan = Path(directory) / "dependencies.json"
            pins = []
            for dist in distributions():
                name = dist.metadata["Name"]
                version = dist.version
                # PyPI's advisory feed uses the PyTorch base version for CPU wheels.
                if name.lower() == "torch":
                    version = version.removesuffix("+cpu")
                pins.append(f"{name}=={version}")
            requirements.write_text("\n".join(sorted(set(pins))) + "\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip_audit",
                    "--no-deps",
                    "--disable-pip",
                    "--requirement",
                    str(requirements),
                    "--format",
                    "json",
                    "--output",
                    str(scan),
                ]
            )
            if result.returncode not in (0, 1) or not scan.is_file():
                raise SystemExit("Dependency scanner failed")
            report.write_bytes(scan.read_bytes())
        check_dependencies(report)
