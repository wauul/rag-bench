"""Small, durable StateGraph per benchmark/configuration/question.

Every explicit attempt reconciles at the validation entry. Application commits are
authoritative, so a lost checkpoint never repeats a durably saved provider result.
"""

import gc
import json
import sqlite3
import time
from contextlib import contextmanager
from contextvars import copy_context
from copy import deepcopy
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from backend.chains import ChromaRetriever, answer_chain, passage_selector, to_chunks, to_documents
from backend.graph_index import SharedIndex
from backend.models import METRICS, MODELS, Configuration
from backend.profiling import measure
from backend.provenance import GRAPH_VERSION, digest, validate_snapshot
from backend.retrieval_trace import load_trace, save_trace
from backend.run_state import has_answer, missing_metrics, needs_work


class WorkState(TypedDict):
    benchmark_id: str
    configuration_id: str
    question_index: int
    fingerprint: str
    graph_version: str
    stage: str


class JsonStateSerializer:
    """Checkpoints contain only primitive references; never deserialize Python objects."""

    def dumps_typed(self, value):
        return "json", json.dumps(value, allow_nan=False).encode()

    def loads_typed(self, value):
        kind, data = value
        if kind != "json":
            raise ValueError("Unsupported checkpoint serialization")
        return json.loads(data)


@contextmanager
def checkpointer(store):
    if store.postgres:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg.rows import dict_row

        from backend.postgres import connect

        with connect(store.database_url, autocommit=True, row_factory=dict_row) as connection:
            saver = PostgresSaver(connection, serde=JsonStateSerializer())
            saver.setup()
            yield saver
        return
    connection = sqlite3.connect(
        store.root / "langgraph.sqlite3", timeout=30, check_same_thread=False
    )
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        saver = SqliteSaver(connection, serde=JsonStateSerializer())
        saver.setup()
        yield saver
    finally:
        connection.close()


def thread_id(run_id, configuration_id, question_index):
    return "work-" + digest([run_id, configuration_id, question_index])


def failure_kind(error):
    name = error if isinstance(error, str) else type(error).__name__
    if any(t in name for t in ("RateLimit", "Timeout", "Connection", "InternalServer")):
        return "retryable_provider"
    if any(
        t in name for t in ("Authentication", "Permission", "BadRequest", "NotFound", "ValueError")
    ):
        return "validation_or_configuration"
    return "provider_or_output_error"


def execute_configuration(
    store, run, config_data, llm, runner, save, checkpoint, stage, retry=False
):
    """The outer scheduler owns admission/cancellation/profiling, the graph owns row stages."""
    from backend import pipeline, retrieval

    config = Configuration(**config_data)
    with checkpointer(store) as saver:
        for question_index, question in enumerate(run["questions"]):
            checkpoint()
            row = next(
                (
                    r
                    for r in run["rows"]
                    if r["configuration_id"] == config_data["id"]
                    and r["question_index"] == question_index
                ),
                None,
            )
            identity = thread_id(run["id"], config_data["id"], question_index)
            graph_config = {"configurable": {"thread_id": identity}, "recursion_limit": 20}
            initial: WorkState = {
                "benchmark_id": run["id"],
                "configuration_id": config_data["id"],
                "question_index": question_index,
                "fingerprint": run["input_fingerprint"],
                "graph_version": GRAPH_VERSION,
                "stage": "validate",
            }
            # Check before any provider/model work, even if the application row is complete.
            previous = saver.get_tuple(graph_config)
            if previous:
                previous_state = previous.checkpoint["channel_values"]
                # A process can die after LangGraph commits input but before START applies it.
                previous_state = previous_state.get("__start__", previous_state)
                if any(previous_state.get(k) != initial[k] for k in initial if k != "stage"):
                    raise ValueError("Stale or incompatible LangGraph checkpoint")
            if row and not needs_work(row):
                continue
            if retry and row:
                run.setdefault("retry_history", []).append(
                    {
                        "retried_at": pipeline.now(),
                        "attempt": run.get("attempt", 1),
                        "previous_row": deepcopy(row),
                    }
                )
                save()
            labels = {"configuration_id": config_data["id"], "question_index": question_index}
            index = SharedIndex(store, run, config_data, checkpoint)
            metric_objects = None
            node_seconds = 0.0
            answer_seconds = 0.0
            artifact_id = identity + "-candidates"
            try:
                candidates = store.get("graph_artifact", artifact_id)
                if candidates["fingerprint"] != run["input_fingerprint"]:
                    raise ValueError("Stale retrieval artifact")
            except KeyError:
                candidates = None

            def wrap(name, action):
                def node(state):
                    nonlocal node_seconds, answer_seconds
                    started = time.monotonic()
                    try:
                        checkpoint()
                        label = {
                            "validate": "Checking saved inputs",
                            "prepare_index": "Preparing index",
                            "retrieve": "Retrieving passages",
                            "rerank": "Reranking passages",
                            "select": "Selecting passages",
                            "generate": "Generating answer",
                            "finalize": "Saving completion",
                        }.get(name, name.replace("evaluate_", "Scoring ").replace("_", " "))
                        stage(
                            f"{config.name} · question {question_index + 1}/{len(run['questions'])} · {label}"
                        )
                        result = action()
                        return {"stage": name, **(result or {})}
                    finally:
                        elapsed = time.monotonic() - started
                        node_seconds += elapsed
                        if name == "generate" or name.startswith("evaluate_"):
                            answer_seconds += elapsed

                return node

            def validate():
                validate_snapshot(store, run)
                if row:
                    # A durable value wins over stale error bookkeeping from an older attempt.
                    for key in (["generation"] if has_answer(row) else []) + [
                        m for m in METRICS if m not in missing_metrics(row)
                    ]:
                        row["errors"].pop(key, None)
                        row.get("failure_kinds", {}).pop(key, None)
                    save()

            def route(_):
                if row and row.get("contexts"):
                    return "generate" if not has_answer(row) else "evaluate_faithfulness"
                if candidates:
                    return "rerank" if config.rerank else "select"
                return "prepare_index"

            def retrieve():
                nonlocal candidates
                index.load_model()
                try:
                    with measure("retrieval", **labels):
                        documents = ChromaRetriever(index=index).invoke(question["question"])
                    candidates = {
                        "run_id": run["id"],
                        "fingerprint": run["input_fingerprint"],
                        "contexts": to_chunks(documents),
                    }
                    store.save("graph_artifact", candidates, artifact_id)
                finally:
                    index.close()

            def select():
                nonlocal row
                if row and row.get("contexts"):
                    return
                reranker = None
                try:
                    trace = load_trace(store, run, config_data, question_index)
                    if config.rerank and trace is None:
                        with measure(
                            "reranker_load", model="cross-encoder/ms-marco-MiniLM-L-6-v2", **labels
                        ):
                            reranker = retrieval.load_reranker()
                    with measure("reranking" if config.rerank else "passage_selection", **labels):
                        selected = (
                            to_documents(trace["selected"])
                            if trace
                            else passage_selector(
                                config,
                                reranker,
                                lambda value: save_trace(
                                    store, run, config_data, question_index, value
                                ),
                            ).invoke(
                                {
                                    "question": question["question"],
                                    "documents": to_documents(candidates["contexts"]),
                                }
                            )
                        )
                    row = {
                        **labels,
                        "configuration_name": config.name,
                        "engine": "langgraph",
                        **question,
                        "contexts": to_chunks(selected),
                        "answer": "",
                        "scores": dict.fromkeys(METRICS),
                        "errors": {},
                        "processed": False,
                        "stage": "retrieved",
                        "latency_seconds": 0,
                        "graph_thread_id": identity,
                        "input_fingerprint": run["input_fingerprint"],
                    }
                    run["rows"].append(row)
                    save()
                finally:
                    del reranker
                    gc.collect()

            def generate():
                if has_answer(row):
                    row["errors"].pop("generation", None)
                    return
                # Orphaned scores cannot describe a newly generated answer (e.g. partial restore).
                row["scores"] = dict.fromkeys(METRICS)
                row["errors"] = {}
                row["failure_kinds"] = {}
                save()
                try:
                    with measure("generation", **labels):
                        answer = answer_chain(llm).invoke(
                            {
                                "question": row["question"],
                                "documents": to_documents(row["contexts"]),
                            }
                        )
                    row.update(answer=answer, stage="generated")
                    row["errors"].pop("generation", None)
                    row.setdefault("failure_kinds", {}).pop("generation", None)
                except Exception as exc:
                    row["errors"]["generation"] = type(exc).__name__ + ": generation failed"
                    row.setdefault("failure_kinds", {})["generation"] = failure_kind(exc)
                # Save even if cancellation arrived while the provider was running.
                save()

            def evaluate(name):
                def action():
                    nonlocal metric_objects
                    if not has_answer(row) or name not in missing_metrics(row):
                        if name not in missing_metrics(row):
                            row["errors"].pop(name, None)
                        return
                    if metric_objects is None:
                        with measure("judge_model_load", model=MODELS[0], **labels):
                            evaluation_model = pipeline.load_embedder(MODELS[0])
                        with measure("judge_setup", **labels):
                            metric_objects = pipeline.make_metrics(llm, evaluation_model)
                    row["stage"] = "scoring"
                    save()

                    def commit(name, value, error):
                        row["scores"][name] = value
                        row["errors"].pop(name, None)
                        row.setdefault("failure_kinds", {}).pop(name, None)
                        if error:
                            row["errors"][name] = error
                            row["failure_kinds"][name] = failure_kind(error)
                        save()

                    runner.run(
                        pipeline.score_row(
                            {name: metric_objects[name]},
                            row,
                            on_score=commit,
                            checkpoint=checkpoint,
                        ),
                        context=copy_context(),
                    )

                return action

            def finalize():
                row.update(processed=True, stage="evaluated")
                save()

            def next_metric(after=-1):
                def route_metric(_):
                    if has_answer(row):
                        for metric in METRICS[after + 1 :]:
                            if metric in missing_metrics(row):
                                return "evaluate_" + metric
                    return "finalize"

                return route_metric

            builder = StateGraph(WorkState)
            for name, action in (
                ("validate", validate),
                ("prepare_index", index.prepare),
                ("retrieve", retrieve),
                ("rerank", select),
                ("select", select),
                ("generate", generate),
                ("finalize", finalize),
            ):
                builder.add_node(name, wrap(name, action))
            for name in METRICS:
                builder.add_node("evaluate_" + name, wrap("evaluate_" + name, evaluate(name)))
            builder.add_edge(START, "validate")
            builder.add_conditional_edges("validate", route)
            builder.add_edge("prepare_index", "retrieve")
            builder.add_conditional_edges(
                "retrieve", lambda _: "rerank" if config.rerank else "select"
            )
            builder.add_edge("rerank", "generate")
            builder.add_edge("select", "generate")
            builder.add_conditional_edges("generate", next_metric())
            for i, name in enumerate(METRICS):
                builder.add_conditional_edges("evaluate_" + name, next_metric(i))
            builder.add_edge("finalize", END)
            started = time.monotonic()
            try:
                with measure("orchestration_setup", **labels):
                    graph = builder.compile(checkpointer=saver)
                # Re-enter validation, preserving thread history. Never trust a graph cursor
                # over a missing application commit. Each expensive stage has a durable guard.
                graph.invoke(initial, graph_config, durability="sync")
            finally:
                elapsed = time.monotonic() - started
                if row:
                    row["latency_seconds"] += round(answer_seconds, 6)
                    row["graph_elapsed_seconds"] = row.get("graph_elapsed_seconds", 0) + elapsed
                    row["orchestration_seconds"] = row.get("orchestration_seconds", 0) + max(
                        0, elapsed - node_seconds
                    )
                    save()
                index.close()
                metric_objects = None
                gc.collect()
