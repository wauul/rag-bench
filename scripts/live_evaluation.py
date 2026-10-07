"""Explicit, bounded sample evaluation using the existing optimizer and HTTP budget.

This smoke runner does not approve production quality, change defaults or upload data.
Run inside the candidate image to bind evaluation to its exact code/models.
"""

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from backend.ingestion import extract_document, parse_test_set
from backend.models import Configuration
from backend.optimization import PlanRequest, SearchSpace, create_plan, execute_experiment
from backend.provenance import digest
from backend.storage import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approve-provider-spend", action="store_true")
    parser.add_argument("--max-requests", type=int, default=48)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/live-evaluation.json"))
    args = parser.parse_args()
    if not args.approve_provider_spend or not os.getenv("GROQ_API_KEY"):
        raise ValueError("Explicit provider-spend approval and GROQ_API_KEY are required")
    if not 1 <= args.max_requests <= 300 or not 2 <= args.samples <= 10:
        raise ValueError("Live evaluation requires 2-10 samples and 1-300 HTTP attempts")
    with tempfile.TemporaryDirectory(prefix="ragbench-evaluation-") as directory:
        store = Store(Path(directory), database_url="")
        docs = store.save(
            "documents",
            {
                "documents": extract_document(
                    "harbor-handbook.txt", Path("sample_data/harbor-handbook.txt").read_bytes()
                )
            },
        )
        questions = parse_test_set(
            "questions.json", Path("sample_data/questions.json").read_bytes()
        )
        tests = store.save(
            "test_set",
            {
                "name": "Approved synthetic smoke",
                "questions": [q.model_dump() for q in questions.questions[: args.samples]],
            },
        )
        baseline = store.save(
            "configuration",
            Configuration(
                name="Sample baseline",
                engine="langgraph",
                chunk_size=128,
                overlap=0,
                candidate_k=3,
                context_k=1,
            ).model_dump(),
        )
        request = PlanRequest(
            document_set_id=docs["id"],
            test_set_id=tests["id"],
            baseline_configuration_id=baseline["id"],
            max_trials=4,
            max_requests=args.max_requests,
            exploratory_acknowledged=True,
            space=SearchSpace(
                chunk_size=[128],
                overlap=[0],
                embedding_model=[baseline["embedding_model"]],
                candidate_k=[3],
                context_k=[1, 2],
                rerank=[False],
            ),
        )
        plan = create_plan(store, request)
        plan.update(
            approved_at=datetime.now(timezone.utc).isoformat(),
            status="queued",
            attempt=plan["attempt"] + 1,
        )
        store.save("optimization", plan, plan["id"])
        result = execute_experiment(store, plan["id"])
        report = {
            "revision": os.getenv("APP_REVISION", "local"),
            "image": args.image,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "experiment_id": result["id"],
            "dataset_fingerprint": result["plan"]["dataset_fingerprint"],
            "prompt_fingerprint": result["plan"]["provenance"]["generation_prompt"]["fingerprint"],
            "model": result["plan"]["provenance"]["generator"],
            "evaluator_fingerprint": digest(result["plan"]["provenance"]["judge_templates"]),
            "sample_count": args.samples,
            "exploratory": result["plan"]["split"]["exploratory"],
            "complete": result["status"] == "completed",
            "usage": result["usage"],
            "results": result.get("tuning_results", []),
            "selection": result.get("selection"),
            "error": result.get("error"),
            "promotion_eligible": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not report["complete"] or any(not r["eligible"] for r in report["results"]):
            raise SystemExit("Bounded evaluation incomplete; report cannot authorize promotion")
    print("Bounded synthetic evaluation complete; smoke evidence only")


if __name__ == "__main__":
    main()
