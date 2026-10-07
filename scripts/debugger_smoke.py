"""One small real retrieval + optional Groq answer, isolated from benchmark results."""

import json
import os
from pathlib import Path
from threading import Event

from dotenv import load_dotenv

from backend.debugger import create, execute
from backend.models import Configuration
from backend.storage import Store


def main():
    load_dotenv()
    store = Store(Path("data/debugger-smoke"), database_url="")
    docs = store.save(
        "documents",
        {
            "documents": [
                {
                    "source": "smoke.txt",
                    "page": 1,
                    "text": "Ragbench debugger runs never affect benchmark scores. Retrieval-only runs do not call the generation or judge model. Generate answer uses the saved final context.",
                }
            ]
        },
    )
    record = create(
        store,
        "Do debugger runs affect benchmark scores?",
        docs["id"],
        Configuration(name="Debugger smoke", candidate_k=3, context_k=2).model_dump(),
    )
    result = execute(store, record["id"], Event())
    if result["status"] != "paused":
        raise RuntimeError(result.get("error", "Retrieval failed"))
    if os.getenv("GROQ_API_KEY"):
        result = execute(store, record["id"], Event(), True, result["context_fingerprint"])
    summary = {
        "id": result["id"],
        "status": result["status"],
        "stage": result["stage"],
        "real_retrieval": True,
        "real_generation": bool(result["answer"]),
        "answer": result["answer"],
        "usage": result["usage"],
        "error": result.get("error"),
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/debugger-smoke.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
