"""Exec the server so Docker/Render shutdown signals reach the single worker."""

import os
import sys


def main():
    port = int(os.getenv("PORT", "8000"))
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be between 1 and 65535")
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--workers",
            "1",
            "--timeout-graceful-shutdown",
            "150",
            "--no-access-log",
        ],
    )


if __name__ == "__main__":
    main()
