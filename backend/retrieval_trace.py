"""Original retrieval/selection evidence, committed alongside newly retrieved results."""

from copy import deepcopy

from backend.provenance import digest

TRACE_VERSION = "1"


def trace_id(run_id, configuration_id, question_index):
    return "retrieval-" + digest([run_id, configuration_id, question_index])


def save_trace(store, run, config, question_index, selection, question=None):
    return store.save(
        "retrieval_trace",
        {
            "schema_version": TRACE_VERSION,
            "run_id": run["id"],
            "configuration_id": config["id"],
            "question_index": question_index,
            "fingerprint": run.get("input_fingerprint"),
            "selection_fingerprint": digest(
                [config, question or run["questions"][question_index]["question"]]
            ),
            "engine": config.get("engine", "existing"),
            **deepcopy(selection),
        },
        trace_id(run["id"], config["id"], question_index),
    )


def load_trace(store, run, config, question_index, contexts=None):
    try:
        trace = store.get("retrieval_trace", trace_id(run["id"], config["id"], question_index))
    except KeyError:
        return None
    if (
        trace["fingerprint"] != run.get("input_fingerprint")
        or trace["selection_fingerprint"]
        != digest([config, run["questions"][question_index]["question"]])
        or trace["schema_version"] != TRACE_VERSION
    ):
        raise ValueError("Stale original retrieval trace")
    if contexts is not None and trace["selected"] != contexts:
        raise ValueError("Retrieval trace does not describe this result's context")
    return trace


def trace_summary(trace):
    if not trace:
        return None
    order = {c["chunk_id"]: i for i, c in enumerate(trace["ordered_candidates"], 1)}
    scores = {c["chunk_id"]: c.get("rerank_score") for c in trace["ordered_candidates"]}
    selected = {c["chunk_id"] for c in trace["selected"]}
    return {
        "rerank_applied": trace["rerank_applied"],
        "candidate_k": trace["candidate_k"],
        "context_k": trace["context_k"],
        "passages": [
            {
                "chunk_id": c["chunk_id"],
                "retrieval_rank": i,
                "selection_rank": order[c["chunk_id"]],
                "rerank_score": scores[c["chunk_id"]],
                "supplied_to_answer": c["chunk_id"] in selected,
            }
            for i, c in enumerate(trace["candidates"], 1)
        ],
    }
