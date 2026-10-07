"""Fail on unexcepted dependency vulnerabilities and expired exceptions."""

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path


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
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip_audit",
                "--local",
                "--format",
                "json",
                "--output",
                str(report),
            ]
        )
        if result.returncode not in (0, 1) or not report.is_file():
            raise SystemExit("Dependency scanner failed")
        check_dependencies(report)
