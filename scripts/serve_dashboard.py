"""Signal-preserving Streamlit entrypoint with provider port support."""

import os
import sys

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8501"))
    if not 1 <= port <= 65535:
        raise ValueError("Invalid PORT")
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "dashboard/app.py",
            "--server.address=0.0.0.0",
            f"--server.port={port}",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
        ],
    )
