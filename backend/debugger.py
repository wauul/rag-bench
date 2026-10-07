"""Isolated retrieval inspection; application commits reconcile graph checkpoints."""

import re
import time
from copy import deepcopy
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from backend import pipeline, retrieval
from backend.chains import (
    ChromaRetriever,
    answer_chain,
    passage_selector,
    prompt_messages,
    to_chunks,
    to_documents,
)
from backend.execution_lock import execution_lock
from backend.graph_index import SharedIndex
from backend.graph_pipeline import checkpointer
from backend.models import Configuration
from backend.observability import stage as observe_stage
from backend.observability import workflow
from backend.profiling import RunProfiler, measure, timestamp
from backend.provenance import digest, implementation, run_documents
from backend.retrieval_trace import load_trace
from backend.settings import load_settings

MAX_HISTORY = 100


class DebugState(TypedDict):
    debug_id: str
    fingerprint: str
    stage: str


class RunCancelled(Exception):
    pass


def create(store, question, document_set_id, configuration, *, origin=None, documents=None):
    if len(store.list("debug_run", limit=MAX_HISTORY)) >= MAX_HISTORY:
        raise ValueError("Delete an old debug run before creating another (limit 100)")
    from backend.prompts import resolve

    config = Configuration(**configuration).model_dump()
    documents = deepcopy(
        documents if documents is not None else store.get("documents", document_set_id)["documents"]
    )
    if not 3 <= len(question.strip()) <= 2000:
        raise ValueError("Question must contain 3–2000 characters")
    doc_hash = digest(documents)
    fingerprint = digest([question.strip(), config, doc_hash])
    return store.save(
        "debug_run",
        {
            "schema_version": 1,
            "question": question.strip(),
            "configuration": config,
            "document_set_id": document_set_id,
            "document_fingerprint": doc_hash,
            "fingerprint": fingerprint,
            "origin": origin,
            "evidence_kind": "replay" if origin else "new",
            "inputs": documents,
            "status": "queued",
            "stage": "validate",
            "created_at": timestamp(),
            "candidates": None,
            "ordered_candidates": None,
            "selected": None,
            "context_fingerprint": None,
            "prompt_messages": None,
            "answer": None,
            "usage": None,
            "timings": {},
            "rows": [],
            "attempt": 0,
            "provenance": {
                **load_settings().provenance(),
                **implementation(),
                "generation_prompt": resolve(),
            },
            "score_semantics": {
                "distance": "Chroma cosine distance; lower is better",
                "rerank_score": "CrossEncoder relevance score; higher is better; not calibrated",
            },
            "context_policy": "First context_k passages after optional reranking; no additional truncation",
        },
    )


def public(record):
    return {k: v for k, v in record.items() if k not in {"inputs", "rows"}}


def context_identity(record):
    return digest([record["fingerprint"], record["selected"]])


def validate(record):
    if (
        digest(record["inputs"]) != record["document_fingerprint"]
        or digest([record["question"], record["configuration"], record["document_fingerprint"]])
        != record["fingerprint"]
    ):
        raise ValueError("Saved inputs changed")
    if record["selected"] is not None and context_identity(record) != record["context_fingerprint"]:
        raise ValueError("Saved context changed")
    if {k: v for k, v in record["provenance"].items() if k != "generation_prompt"} != {
        **load_settings().provenance(),
        **implementation(),
    }:
        raise ValueError("Pipeline or model settings changed; create a replay")


@workflow("debugger", "debug_run")
def execute(store, debug_id, event, generate=False, expected_context=None):
    from backend.guardrails import policy_context

    with execution_lock(store), policy_context(store):
        record = store.get("debug_run", debug_id)
        record.update(status="running", attempt=record["attempt"] + 1)

        def save():
            store.save("debug_run", record, debug_id)

        def checkpoint():
            from backend.guardrails import check

            check(store)
            if event.is_set():
                raise RunCancelled()
            try:
                store.get("debug_cancel", debug_id + "-cancel")
            except KeyError:
                return
            raise RunCancelled()

        config = Configuration(**record["configuration"])
        index_hash = digest(
            [
                record["document_fingerprint"],
                record["configuration"],
                {
                    k: record["provenance"][k]
                    for k in ("model_revisions", "inference_backend", "precision")
                },
            ]
        )
        index = SharedIndex(
            store,
            {"id": "debug-" + index_hash[:24], "input_fingerprint": index_hash},
            {**record["configuration"], "id": "debug"},
            checkpoint,
            documents=record["inputs"],
        )

        def node(name, action):
            def call(state):
                checkpoint()
                record["stage"] = name
                save()
                started = time.monotonic()
                with measure(name) if name == "generate" else observe_stage(name):
                    action()
                record["timings"][name] = (
                    record["timings"].get(name, 0) + time.monotonic() - started
                )
                save()
                return {"stage": name}

            return call

        def prepare():
            if record["candidates"] is None:
                index.prepare()
                record["index"] = {
                    "fingerprint": index_hash,
                    "collection": index.collection.name,
                    "reuse": bool(getattr(index, "reused", False)),
                }

        def retrieve():
            if record["candidates"] is None:
                if index.collection is None:
                    index.prepare()
                record["candidates"] = to_chunks(
                    ChromaRetriever(index=index).invoke(record["question"])
                )
            index.close()

        def rerank():
            if record["ordered_candidates"] is None:
                reranker = retrieval.load_reranker() if config.rerank else None

                def capture(trace):
                    record["ordered_candidates"] = trace["ordered_candidates"]

                passage_selector(config, reranker, capture).invoke(
                    {
                        "question": record["question"],
                        "documents": to_documents(record["candidates"]),
                    }
                )

        def select():
            if record["selected"] is None:
                record["selected"] = deepcopy(record["ordered_candidates"][: config.context_k])
                record["context_fingerprint"] = context_identity(record)

        def review(state):
            # No side effects before interrupt: this node is replayed on resume.
            decision = interrupt({"context_fingerprint": record["context_fingerprint"]})
            validate(record)
            if decision != record["context_fingerprint"]:
                raise ValueError("Resume context fingerprint does not match")
            checkpoint()
            return {"stage": "reviewed"}

        def prompt():
            record["prompt_messages"] = exact_prompt(record["question"], record["selected"])

        def generation():
            if record["answer"] is not None:
                return
            if not generate:
                raise ValueError("Generation requires an explicit generation action")
            validate(record)
            llm = pipeline.make_llm()

            class Capture:
                def invoke(self, value):
                    actual = [{"role": m.type, "content": m.content} for m in value.to_messages()]
                    if actual != record["prompt_messages"]:
                        raise ValueError("Invocation differs from saved prompt")
                    response = llm.invoke(value)
                    record["usage"] = getattr(response, "usage_metadata", None)
                    return response

            record["answer"] = answer_chain(Capture()).invoke(
                {"question": record["question"], "documents": to_documents(record["selected"])}
            )
            # Save provider output before checking cancellation or checkpointing.
            save()

        builder = StateGraph(DebugState)
        actions = {
            "validate": lambda: validate(record),
            "prepare_index": prepare,
            "retrieve": retrieve,
            "rerank": rerank,
            "select": select,
            "persist_retrieval": save,
            "prompt": prompt,
            "generate": generation,
            "persist_final": lambda: record.update(status="completed"),
        }
        for name, action in actions.items():
            builder.add_node(name, node(name, action))
        builder.add_node("review", review)
        stages = [
            "validate",
            "prepare_index",
            "retrieve",
            "rerank",
            "select",
            "persist_retrieval",
            "review",
            "prompt",
            "generate",
            "persist_final",
        ]
        builder.add_edge(START, stages[0])
        for first, second in zip(stages, stages[1:]):
            builder.add_edge(first, second)
        builder.add_edge(stages[-1], END)
        graph_config = {"configurable": {"thread_id": "debug-" + debug_id}}
        try:
            validate(record)
            if (
                generate
                and expected_context is not None
                and expected_context != record["context_fingerprint"]
            ):
                raise ValueError("Displayed context is stale")
            from langsmith import tracing_context

            with (
                tracing_context(enabled=False),
                RunProfiler(record, save) as profiler,
                checkpointer(store) as saver,
            ):
                graph = builder.compile(checkpointer=saver)
                previous = graph.get_state(graph_config)
                if previous.values and previous.values.get("fingerprint") != record["fingerprint"]:
                    raise ValueError("Checkpoint fingerprint differs")
                initial = {
                    "debug_id": debug_id,
                    "fingerprint": record["fingerprint"],
                    "stage": "validate",
                }
                paused = bool(previous.tasks and any(t.interrupts for t in previous.tasks))
                generation_pending = bool(
                    set(previous.next) & {"prompt", "generate", "persist_final"}
                )
                if generation_pending and not generate:
                    record.update(status="paused" if record["answer"] is None else "completed")
                elif paused:
                    if generate:
                        graph.invoke(Command(resume=record["context_fingerprint"]), graph_config)
                else:
                    graph.invoke(None if previous.next else initial, graph_config)
                    pending = graph.get_state(graph_config)
                    if generate and any(t.interrupts for t in pending.tasks):
                        checkpoint()
                        graph.invoke(Command(resume=record["context_fingerprint"]), graph_config)
                if record["answer"] is None:
                    record.update(status="paused", stage="review")
                else:
                    record.update(status="completed", stage="persist_final")
                profiler.status = record["status"]
        except RunCancelled:
            record.update(
                status="cancelled", error="Cancelled between stages; in-flight calls may finish"
            )
        except Exception as exc:
            record.update(
                status="failed", error=type(exc).__name__ + ": stage failed; retry or replay"
            )
        finally:
            index.close()
            save()
        return record


def exact_prompt(question, contexts):
    return prompt_messages(question, contexts)


def historical(store, run, configuration_id, question_index):
    config = next(c for c in run["configurations"] if c["id"] == configuration_id)
    row = next(
        r
        for r in run["rows"]
        if r["configuration_id"] == configuration_id and r["question_index"] == question_index
    )
    trace = load_trace(store, run, config, question_index, row.get("contexts"))
    return {
        "schema_version": 1,
        "evidence_kind": "historical",
        "origin": {
            "run_id": run["id"],
            "configuration_id": configuration_id,
            "question_index": question_index,
        },
        "question": row["question"],
        "configuration": config,
        "document_set_id": run["document_set_id"],
        "status": "recorded",
        "stage": row.get("stage"),
        "selected": row.get("contexts"),
        "answer": row.get("answer"),
        "candidates": trace["candidates"] if trace else None,
        "ordered_candidates": trace["ordered_candidates"] if trace else None,
        "prompt_messages": row.get("prompt_messages"),
        "provenance": run.get("provenance"),
        "timings": {"generation_and_scoring": row.get("latency_seconds")},
        "usage": None,
    }


def replay(store, evidence):
    run = store.get("run", evidence["origin"]["run_id"])
    # Historical snapshots are authoritative; legacy corpora are explicitly newly read.
    return create(
        store,
        evidence["question"],
        evidence["document_set_id"],
        evidence["configuration"],
        origin=evidence["origin"],
        documents=run_documents(store, run),
    )


def markers(answer, contexts):
    result = []
    for marker in dict.fromkeys(re.findall(r"\[(\d+)\]", answer or "")):
        value = int(marker) if len(marker) <= 6 else marker
        valid = isinstance(value, int) and 1 <= value <= len(contexts or [])
        result.append({"marker": value, "valid": valid, "passage": value if valid else None})
    return result


def is_stale(record, question, document_set_id, configuration):
    return (
        record["question"] != question.strip()
        or record["document_set_id"] != document_set_id
        or Configuration(**record["configuration"]).model_dump()
        != Configuration(**configuration).model_dump()
    )


def compare(left, right):
    if (
        left["question"] != right["question"]
        or left["document_fingerprint"] != right["document_fingerprint"]
    ):
        raise ValueError("Compare runs for the same question and document snapshot")
    changes = {
        k: [left["configuration"][k], right["configuration"][k]]
        for k in left["configuration"]
        if left["configuration"][k] != right["configuration"][k]
    }
    compatible = all(k not in changes for k in ("chunk_size", "overlap"))

    def identity(c):
        return digest(
            [c.get("source"), c.get("page"), c.get("token_start"), c.get("token_end"), c["text"]]
        )

    a = {identity(c): i for i, c in enumerate(left["candidates"] or [], 1)}
    b = {identity(c): i for i, c in enumerate(right["candidates"] or [], 1)}
    passages = {identity(c): c for c in (left["candidates"] or []) + (right["candidates"] or [])}
    return {
        "changed_settings": changes,
        "chunking_compatible": compatible,
        "correspondence": "Exact source/page/token/text identity"
        if compatible
        else "Unavailable: chunking differs",
        "ranks": [
            {
                "identity": key,
                "left": a.get(key),
                "right": b.get(key),
                "presence": "shared"
                if key in a and key in b
                else "left only"
                if key in a
                else "right only",
                "source": passages[key].get("source"),
                "page": passages[key].get("page"),
                "text": passages[key]["text"],
            }
            for key in sorted(a.keys() | b.keys())
        ]
        if compatible
        else None,
        "final_context_equal": left["selected"] == right["selected"],
        "final_contexts": [left["selected"], right["selected"]],
        "answers": [left["answer"], right["answer"]],
        "timings": [left["timings"], right["timings"]],
        "usage": [left["usage"], right["usage"]],
        "cache": [left.get("index"), right.get("index")],
    }
