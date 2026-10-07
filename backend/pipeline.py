"""Real local retrieval + Groq generation + Ragas evaluation. No synthetic score fallback."""

import gc
import logging
import math
import time
from contextvars import copy_context
from copy import deepcopy
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from backend.models import METRICS, MODELS
from backend.profiling import ProfiledAsyncClient, ProfiledClient, RunProfiler, measure
from backend.retrieval import embed, load_embedder, retrieve_questions
from backend.run_state import (
    RunCancelled,
    has_answer,
    missing_metrics,
    needs_work,
    progress,
    valid_score,
)
from backend.settings import load_settings

log = logging.getLogger(__name__)


def now():
    return datetime.now(timezone.utc).isoformat()


def make_llm():
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_groq import ChatGroq

    settings = load_settings()
    return ChatGroq(
        model=settings.model,
        temperature=0,
        max_tokens=2048,
        cache=False,
        n=1,
        reasoning_effort=settings.reasoning_effort,
        max_retries=1,
        timeout=120,
        http_client=ProfiledClient(),
        http_async_client=ProfiledAsyncClient(),
        rate_limiter=InMemoryRateLimiter(
            requests_per_second=1 / settings.request_interval,
            check_every_n_seconds=0.1,
            max_bucket_size=1,
        ),
    )


def make_metrics(llm, evaluation_model):
    from langchain_core.embeddings import Embeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

    from backend.groq_judge import GroqRagasLLM

    class FixedEmbeddings(Embeddings):
        # Hold the judge embedding model fixed across configurations for a fair relevancy comparison.
        def embed_documents(self, texts):
            return embed(evaluation_model, texts)

        def embed_query(self, text):
            return self.embed_documents([text])[0]

    # Groq only supports n=1. Ragas relevancy needs three independent completions;
    # bypass_n issues separate requests instead of forwarding unsupported n=3 to Groq.
    judge = GroqRagasLLM(
        llm, bypass_n=True, run_config=RunConfig(timeout=240, max_retries=1, max_workers=1)
    )
    embeddings = LangchainEmbeddingsWrapper(FixedEmbeddings())
    # Faithfulness: fraction of generated claims the judge finds supported by retrieved context.
    # Relevancy: cosine similarity of original query to questions regenerated from the answer,
    # penalized for evasive answers. Three regenerated questions is Ragas's standard strictness.
    # Precision: average precision of retrieved chunks judged useful against the reference.
    # Recall: fraction of reference claims supported by the retrieved context.
    metrics = {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": ResponseRelevancy(llm=judge, embeddings=embeddings, strictness=3),
        "context_precision": LLMContextPrecisionWithReference(llm=judge),
        "context_recall": LLMContextRecall(llm=judge),
    }
    # Preserve Ragas instructions, schemas and score algorithms, while omitting lengthy
    # few-shot examples by default to fit free-tier token quotas. Keep this fixed within
    # a run and disclose it in provenance; users may restore examples with JUDGE_EXAMPLES.
    examples = load_settings().judge_examples
    for metric in metrics.values():
        prompts = deepcopy(metric.get_prompts())
        for prompt in prompts.values():
            prompt.examples = prompt.examples[:examples]
            # No hidden output-repair provider loop on top of the SDK retry budget.
            from functools import partial

            prompt.generate_multiple = partial(prompt.generate_multiple, retries_left=0)
        metric.set_prompts(**prompts)
    return metrics


async def score_row(metrics, row, on_score=None, checkpoint=None):
    from ragas import SingleTurnSample

    sample = SingleTurnSample(
        user_input=row["question"],
        response=row["answer"],
        retrieved_contexts=[c["text"] for c in row["contexts"]],
        reference=row["reference"],
    )
    scores, errors = {}, {}
    for name, metric in metrics.items():
        if checkpoint:
            checkpoint()
        try:
            with measure(
                "scoring",
                configuration_id=row.get("configuration_id"),
                question_index=row.get("question_index"),
                metric=name,
            ):
                raw = await metric.single_turn_ascore(sample, timeout=240)
                if isinstance(raw, bool):
                    raise ValueError("Ragas returned a boolean instead of a score")
                value = float(raw)
                if (
                    not math.isfinite(value)
                    or not (-1 if name == "answer_relevancy" else 0) <= value <= 1
                ):
                    raise ValueError("Ragas returned an invalid score")
            scores[name] = value
        except Exception as exc:
            scores[name] = None
            # Exceptions can contain provider request details: expose type only, never secrets.
            errors[name] = type(exc).__name__ + ": judge call failed or returned invalid output"
            log.warning("Metric %s failed (%s)", name, type(exc).__name__)
        if on_score:
            on_score(name, scores[name], errors.get(name))
    return scores, errors


def summarize(rows, configurations, expected_questions):
    summaries = []
    for config in configurations:
        subset = [r for r in rows if r["configuration_id"] == config["id"]]
        frame = pd.DataFrame(
            [
                {
                    m: r["scores"].get(m) if valid_score(m, r["scores"].get(m)) else None
                    for m in METRICS
                }
                for r in subset
            ],
            columns=METRICS,
        )
        counts = {m: int(frame[m].count()) for m in METRICS}
        means = {m: float(frame[m].mean()) if counts[m] else None for m in METRICS}
        complete = len(subset) == expected_questions and all(
            v == expected_questions for v in counts.values()
        )
        summaries.append(
            {
                "configuration_id": config["id"],
                "name": config["name"],
                **means,
                "valid_counts": counts,
                "expected_questions": expected_questions,
                "complete": complete,
                "overall": float(np.mean(list(means.values()))) if complete else None,
            }
        )
    return summaries


def generate_answer(llm, row):
    from backend.provenance import SYSTEM_PROMPT

    context = "\n\n".join(
        f"[{i + 1}] {c['source']} p.{c['page']}\n{c['text']}" for i, c in enumerate(row["contexts"])
    )
    response = llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", f"PASSAGES:\n{context}\n\nQUESTION: {row['question']}"),
        ]
    )
    if not isinstance(response.content, str) or not response.content.strip():
        raise ValueError("Generator returned no text answer")
    return response.content


def evaluate_row(llm, metrics, row, runner, save, checkpoint):
    """Commit generation and each metric independently so interruption loses no completed work."""
    started = time.monotonic()
    try:
        checkpoint()
        if has_answer(row):
            row["errors"].pop("generation", None)
            for name in METRICS:
                if name not in missing_metrics(row):
                    row["errors"].pop(name, None)
        if not has_answer(row):
            row["scores"] = dict.fromkeys(METRICS)
            row["errors"] = {}
            save()
            try:
                with measure(
                    "generation",
                    configuration_id=row["configuration_id"],
                    question_index=row["question_index"],
                ):
                    row["answer"] = generate_answer(llm, row)
            except Exception as exc:
                row["errors"]["generation"] = (
                    type(exc).__name__ + ": generation failed; check API quota and credentials"
                )
                row.update(processed=True, stage="evaluated")
                return
            row["errors"] = {}
            row["scores"] = {m: None for m in METRICS}
            row["stage"] = "generated"
            save()
        pending = missing_metrics(row)
        row["stage"] = "scoring"

        def commit_score(name, value, error):
            row["scores"][name] = value
            row["errors"].pop(name, None)
            if error:
                row["errors"][name] = error
            save()

        runner.run(
            score_row(
                {m: metrics[m] for m in pending}, row, on_score=commit_score, checkpoint=checkpoint
            ),
            context=copy_context(),
        )
        row.update(processed=True, stage="evaluated")
    finally:
        row["latency_seconds"] = round(
            row.get("latency_seconds", 0) + time.monotonic() - started, 2
        )
        save()


def execute_run(store, run_id, cancel_event=None, retry=False):
    import os

    from langsmith import tracing_context

    from backend.execution_lock import execution_lock

    with (
        execution_lock(store),
        tracing_context(enabled=os.getenv("RAGBENCH_LANGSMITH", "false").lower() == "true"),
    ):
        return _execute_run(store, run_id, cancel_event, retry)


def _execute_run(store, run_id, cancel_event=None, retry=False):
    import asyncio

    run = store.get("run", run_id)
    runner = asyncio.Runner()
    profiler = None
    llm = None

    def checkpoint():
        from backend.execution_lock import assert_execution_lock
        from backend.optimization_budget import current_budget

        assert_execution_lock(store)
        if current_budget():
            current_budget().check()
        if store.postgres:
            try:
                cancellation = store.get("cancellation", run_id + "-cancel")
            except KeyError:
                cancellation = {}
            if cancellation.get("attempt") == run.get("attempt", 1):
                raise RunCancelled()
        if cancel_event is not None and cancel_event.is_set():
            raise RunCancelled()

    def save(refresh_profile=True):
        if profiler is not None and refresh_profile:
            profiler.snapshot()
        run.update(progress(run))
        run["summary"] = summarize(run["rows"], run["configurations"], len(run["questions"]))
        store.save("run", run, run_id)

    def stage(text):
        checkpoint()
        run["stage"] = text
        save()

    profiler = RunProfiler(run, on_update=lambda: save(refresh_profile=False))
    with profiler:
        try:
            checkpoint()
            from backend.provenance import validate_snapshot

            validate_snapshot(store, run)
            run.update(
                status="running",
                started_at=run.get("started_at", now()),
                stage="Resuming missing work" if retry else "Loading local models",
            )
            run.pop("error", None)
            run.pop("finished_at", None)
            save()
            # A restart/cancellation can happen after the last score commit but before finalization.
            if all(s["complete"] for s in run["summary"]) and all(
                not needs_work(r) for r in run["rows"]
            ):
                run.update(status="completed", stage="Finished from saved scores")
                return run
            llm = make_llm()
            # The first required completion verifies the provider; no extra quota on retries.
            configs, questions = run["configurations"], run["questions"]
            for config in configs:
                checkpoint()
                if config.get("engine", "existing") == "langgraph":
                    from backend.graph_pipeline import execute_configuration

                    execute_configuration(
                        store, run, config, llm, runner, save, checkpoint, stage, retry
                    )
                    continue
                by_index = {
                    r["question_index"]: r
                    for r in run["rows"]
                    if r["configuration_id"] == config["id"]
                }
                missing = [i for i in range(len(questions)) if i not in by_index]
                if missing:
                    retrieved = retrieve_questions(
                        store, run_id, config, questions, missing, checkpoint, stage
                    )
                    for index, contexts in retrieved.items():
                        row = {
                            "configuration_id": config["id"],
                            "configuration_name": config["name"],
                            "engine": "existing",
                            "question_index": index,
                            **questions[index],
                            "contexts": contexts,
                            "answer": "",
                            "scores": {m: None for m in METRICS},
                            "errors": {},
                            "latency_seconds": 0,
                            "processed": False,
                            "stage": "retrieved",
                        }
                        run["rows"].append(row)
                        by_index[index] = row
                    # Save passages before any generation or judge request.
                    save()
                pending = [by_index[i] for i in range(len(questions)) if needs_work(by_index[i])]
                if not pending:
                    continue
                checkpoint()
                stage(f"Loading judge embeddings for {config['name']}")
                with measure("judge_model_load", configuration_id=config["id"], model=MODELS[0]):
                    evaluation_model = load_embedder(MODELS[0])
                try:
                    with measure("judge_setup", configuration_id=config["id"]):
                        metrics = make_metrics(llm, evaluation_model)
                    for row in pending:
                        checkpoint()
                        if retry:
                            run.setdefault("retry_history", []).append(
                                {
                                    "retried_at": now(),
                                    "attempt": run.get("attempt", 1),
                                    "previous_row": deepcopy(row),
                                }
                            )
                            save()
                        stage(
                            f"{config['name']} · question {row['question_index'] + 1}/{len(questions)} · generation and Ragas"
                        )
                        evaluate_row(llm, metrics, row, runner, save, checkpoint)
                    del metrics
                finally:
                    del evaluation_model
                    gc.collect()
            checkpoint()
            run["status"] = (
                "completed"
                if all(s["complete"] for s in run["summary"])
                and all(not needs_work(r) for r in run["rows"])
                else "partial"
            )
            run["stage"] = (
                "Finished"
                if run["status"] == "completed"
                else "Finished with errors; retry missing work when quota is available"
            )
        except RunCancelled:
            run.update(status="cancelled", stage="Cancelled; saved work can be resumed")
        except Exception as exc:
            log.exception("Run failed: %s", type(exc).__name__)
            run.update(
                status="failed",
                stage="Run failed; saved work can be resumed",
                error=type(exc).__name__ + ": pipeline failed; inspect server logs",
            )
        finally:
            try:
                if isinstance(getattr(llm, "http_client", None), ProfiledClient):
                    llm.http_client.close()
                if isinstance(getattr(llm, "http_async_client", None), ProfiledAsyncClient):
                    runner.run(llm.http_async_client.aclose(), context=copy_context())
            finally:
                runner.close()
                run["finished_at"] = now()
                profiler.status = run["status"]
                save()
    return run
