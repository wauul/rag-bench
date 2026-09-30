#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .tools

# Reusing the container does not spawn duplicate workers or dashboard processes.
if ! curl --fail --silent --max-time 3 http://127.0.0.1:8000/health >/dev/null; then
  nohup python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1 >.tools/backend.log 2>&1 &
fi
if ! curl --fail --silent --max-time 3 http://127.0.0.1:8501/_stcore/health >/dev/null; then
  nohup python -m streamlit run dashboard/app.py --server.address 0.0.0.0 >.tools/dashboard.log 2>&1 &
fi
