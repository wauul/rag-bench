"""Identical local/CI quality checks. No credentials or provider calls needed."""

import subprocess
import sys


def run(*args):
    subprocess.run(args, check=True)


if __name__ == "__main__":
    run("uv", "lock", "--check")
    run("uv", "pip", "check", "--python", sys.executable)
    run(
        sys.executable,
        "-m",
        "ruff",
        "format",
        "--check",
        "backend",
        "dashboard",
        "scripts",
        "tests",
    )
    run(sys.executable, "-m", "ruff", "check", "backend", "dashboard", "scripts", "tests")
    run(sys.executable, "-m", "mypy")
    run(
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--cov",
        "--cov-report=xml:reports/coverage.xml",
        "--cov-report=json:reports/coverage.json",
        "--junitxml=reports/tests.xml",
    )
    run(sys.executable, "-m", "scripts.coverage_gate", "reports/coverage.json")
