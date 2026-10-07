"""Frozen, bounded retrieval experiments. Benchmark rows remain authoritative."""

import itertools
import random
from copy import deepcopy
from typing import Literal, TypedDict
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.models import MODELS, Configuration
from backend.observability import stage as observe_stage
from backend.observability import workflow
from backend.provenance import digest, implementation, input_identity
from backend.settings import load_settings

VERSION = "optimization-1"
POLICY = "Equal mean of four complete Ragas metrics; baseline wins ties; stable candidate order."


class SearchSpace(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_size: list[int] = Field(default=[128, 192, 240], min_length=1, max_length=8)
    overlap: list[int] = Field(default=[0, 32], min_length=1, max_length=8)
    embedding_model: list[str] = Field(default=MODELS, min_length=1, max_length=2)
    candidate_k: list[int] = Field(default=[3, 8], min_length=1, max_length=8)
    context_k: list[int] = Field(default=[1, 3], min_length=1, max_length=8)
    rerank: list[bool] = Field(default=[False, True], min_length=1, max_length=2)

    @model_validator(mode="after")
    def valid_values(self):
        limits = {
            "chunk_size": (32, 240),
            "overlap": (0, 239),
            "candidate_k": (1, 40),
            "context_k": (1, 8),
        }
        for key, (low, high) in limits.items():
            values = getattr(self, key)
            if any(not low <= value <= high for value in values):
                raise ValueError(f"{key} outside supported range")
        if any(value not in MODELS for value in self.embedding_model):
            raise ValueError("Unsupported embedding model")
        for key in type(self).model_fields:
            if len(set(getattr(self, key))) != len(getattr(self, key)):
                raise ValueError(f"Duplicate values in {key}")
        return self


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_set_id: str
    test_set_id: str
    baseline_configuration_id: str
    space: SearchSpace = Field(default_factory=SearchSpace)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    objective: Literal["quality", "latency"] = "quality"
    quality_threshold: float = Field(default=0.7, ge=-0.25, le=1)
    max_trials: int = Field(default=6, ge=4, le=20)
    max_requests: int = Field(default=300, ge=1, le=10000)
    exploratory_acknowledged: bool = False


def retrieval_identity(config):
    return {key: config[key] for key in SearchSpace.model_fields}


def candidates(baseline, request):
    space = request.space.model_dump()
    keys = list(space)
    valid, rejected = [], 0
    for values in itertools.product(*(space[key] for key in keys)):
        raw = {**baseline, **dict(zip(keys, values)), "name": "Candidate"}
        try:
            config = Configuration(**raw).model_dump()
        except ValueError:
            rejected += 1
            continue
        if retrieval_identity(config) != retrieval_identity(baseline):
            valid.append(config)
    # Canonical order ensures identical sets and seeds produce identical plans.
    valid.sort(key=lambda c: digest(retrieval_identity(c)))
    # Reproducible experiment sampling, not a security decision.
    random.Random(request.seed).shuffle(valid)  # nosec B311
    result = valid[: request.max_trials - 3]  # baseline tuning + two held-out trials
    if not result:
        raise ValueError("Search space contains no valid alternative to the baseline")
    return (
        [
            {**c, "name": f"Candidate {i + 1}", "id": f"candidate-{i + 1}"}
            for i, c in enumerate(result)
        ],
        rejected,
        len(valid),
    )


def split_questions(questions, seed, exploratory_acknowledged):
    identities = [digest(q) for q in questions]
    if len(set(identities)) != len(identities) or len(
        {q["question"].strip().casefold() for q in questions}
    ) != len(questions):
        raise ValueError("Duplicate questions must be removed before splitting")
    exploratory = len(questions) < 6
    if exploratory and not exploratory_acknowledged:
        raise ValueError("Fewer than six questions: acknowledge exploratory tuning")
    indexes = list(range(len(questions)))
    # Reproducible evaluation split, not a security decision.
    random.Random(seed).shuffle(indexes)  # nosec B311
    # Tiny data has no useful independent holdout; never imply generalization.
    heldout_count = 0 if exploratory else max(2, len(questions) // 3)
    heldout = sorted(indexes[:heldout_count])
    tuning = sorted(indexes[heldout_count:])
    return {
        "tuning": tuning,
        "heldout": heldout,
        "question_ids": identities,
        "seed": seed,
        "exploratory": exploratory,
    }


def create_plan(store, request):
    documents = store.get("documents", request.document_set_id)["documents"]
    questions = store.get("test_set", request.test_set_id)["questions"]
    baseline = Configuration(
        **store.get("configuration", request.baseline_configuration_id)
    ).model_dump()
    if baseline["engine"] != "langgraph":
        raise ValueError("Choose a LangChain + LangGraph baseline; engine stays fixed")
    baseline.update(id="baseline", name="Baseline · " + baseline["name"][:48])
    choices, rejected, valid = candidates(baseline, request)
    split = split_questions(questions, request.seed, request.exploratory_acknowledged)
    from backend.prompts import resolve

    frozen = {
        "version": VERSION,
        "request": request.model_dump(),
        "baseline": baseline,
        "candidates": choices,
        "split": split,
        "documents": documents,
        "questions": questions,
        "provenance": {
            **load_settings().provenance(),
            **implementation(),
            "generation_prompt": resolve(),
        },
        "dataset_fingerprint": digest([documents, questions]),
        "selection_policy": POLICY,
    }
    plan = {
        "id": uuid4().hex,
        "status": "planned",
        "stage": "Review plan before starting",
        "plan_fingerprint": digest(frozen),
        "plan": frozen,
        "trials": [],
        "usage": {"requests_reserved": 0, "total_tokens": 0, "token_reports": 0},
        "max_requests": request.max_requests,
        "amendments": [],
        "attempt": 0,
        "rejected_combinations": rejected,
        "valid_alternatives": valid,
        "estimated_requests": (
            (1 + len(choices)) * len(split["tuning"]) + 2 * len(split["heldout"])
        )
        * 18,
        "estimate_note": "18 HTTP attempts per question/configuration is a planning estimate, not a guarantee. Ragas calls and retries vary.",
        "limitations": [
            "Observed dataset only; no global optimum or statistical significance claim.",
            "No monetary estimate; provider-reported tokens and reserved HTTP attempts are shown.",
            "Indexes are reused only inside an identical saved trial; new trials index fresh.",
        ],
    }
    if split["exploratory"]:
        plan["limitations"].append(
            "Exploratory tuning: no independent held-out evaluation or generalization claim."
        )
    return store.save("optimization", plan, plan["id"])


def validate_plan(record):
    frozen = record["plan"]
    if digest(frozen) != record["plan_fingerprint"] or frozen["version"] != VERSION:
        raise ValueError("Frozen experiment plan changed")
    if {k: v for k, v in frozen["provenance"].items() if k != "generation_prompt"} != {
        **load_settings().provenance(),
        **implementation(),
    }:
        raise ValueError(
            "Restore the original models, prompts, dependencies and settings before resuming"
        )


def trial_id(record, split, config_id):
    return "opt-" + digest([record["id"], record["plan_fingerprint"], split, config_id])[:40]


def prepare_trial(store, record, split, config):
    identity = trial_id(record, split, config["id"])
    plan = record["plan"]
    questions = [plan["questions"][i] for i in plan["split"][split]]
    expected = {
        "configurations": [config],
        "questions": questions,
        "document_set_id": plan["request"]["document_set_id"],
        "provenance": plan["provenance"],
    }
    fingerprint = digest({**expected, "documents": plan["documents"]})
    try:
        run = store.get("run", identity)
        if run.get("input_fingerprint") != fingerprint or input_identity(run) != expected:
            raise ValueError("Trial fingerprint mismatch")
        return run
    except KeyError:
        pass
    if identity not in [t["run_id"] for t in record["trials"]]:
        if len(record["trials"]) >= plan["request"]["max_trials"]:
            raise ValueError("Trial limit reached")
        record["trials"].append(
            {"run_id": identity, "split": split, "configuration_id": config["id"]}
        )
        store.save("optimization", record, record["id"])
    run = {
        "id": identity,
        **deepcopy(expected),
        "optimization_id": record["id"],
        "input_fingerprint": fingerprint,
        "status": "queued",
        "stage": "Queued trial",
        "created_at": record["approved_at"],
        "rows": [],
        "summary": [],
        "total": len(questions),
        "attempt": 1,
    }
    store.save(
        "snapshot",
        {
            "run_id": identity,
            "inputs": {**expected, "documents": plan["documents"]},
            "fingerprint": fingerprint,
        },
        identity + "-snapshot",
    )
    return store.save("run", run, identity)


def trial_summary(store, trial):
    from backend.profiling_report import summarize_profile
    from backend.run_state import needs_work

    run = store.get("run", trial["run_id"])
    score = run.get("summary", [{}])[0] if run.get("summary") else {}
    profile = summarize_profile(run.get("profiling", {}))
    timings = profile.get("timings", {})
    eligible = bool(score.get("complete")) and all(not needs_work(r) for r in run["rows"])
    # Include retrieval and generation, exclude judge/index time from serving latency.
    latency = sum(
        timings.get(k, 0)
        for k in ("query_embedding", "retrieval", "reranking", "passage_selection", "generation")
    )
    return {
        **trial,
        "status": run["status"],
        "eligible": eligible,
        "quality": score.get("overall"),
        "latency_seconds": latency / len(run["questions"]) if eligible else None,
        "timings": timings,
        "usage": profile.get("groq", {}),
        "metrics": score,
        "failures": sum(bool(row["errors"]) for row in run["rows"]),
        "index_cache": "fresh indexing + warm reuse"
        if "indexing" in timings and "index_reuse" in timings
        else "warm reuse"
        if "index_reuse" in timings
        else "fresh indexing",
    }


def select(tuning, objective, threshold):
    eligible = [t for t in tuning if t["eligible"]]
    baseline = next((t for t in eligible if t["configuration_id"] == "baseline"), None)
    if baseline is None:
        return {"selected": None, "outcome": "Baseline incomplete; comparison unavailable"}
    if objective == "latency":
        eligible = [t for t in eligible if t["quality"] >= threshold]
    if not eligible:
        return {"selected": None, "outcome": "No configuration meets the quality threshold"}
    key = (
        (lambda t: (-t["quality"], t["configuration_id"] != "baseline"))
        if objective == "quality"
        else (lambda t: (t["latency_seconds"], t["configuration_id"] != "baseline"))
    )
    winner = min(eligible, key=key)
    return {
        "selected": winner["configuration_id"],
        "outcome": "No improvement found"
        if winner["configuration_id"] == "baseline"
        else "Improvement observed on tuning questions",
    }


class ExperimentState(TypedDict):
    experiment_id: str
    plan_fingerprint: str
    stage: str


@workflow("optimization", "optimization")
def execute_experiment(store, experiment_id):
    import os

    from langgraph.graph import END, START, StateGraph
    from langsmith import tracing_context

    from backend.execution_lock import execution_lock
    from backend.graph_pipeline import checkpointer
    from backend.optimization_budget import ExperimentBudget, budget_context
    from backend.pipeline import _execute_run
    from backend.run_state import RunCancelled

    with (
        execution_lock(store),
        tracing_context(enabled=os.getenv("RAGBENCH_LANGSMITH", "false").lower() == "true"),
    ):
        record = store.get("optimization", experiment_id)
        budget = ExperimentBudget(store, record)

        def save():
            store.save("optimization", record, experiment_id)

        def node(name, action):
            def invoke(_):
                budget.check()
                record["stage"] = name
                save()
                with observe_stage(name):
                    action()
                save()
                return {"stage": name}

            return invoke

        def validate():
            validate_plan(record)
            if not record.get("approved_at"):
                raise ValueError("Plan has not been approved")
            record["status"] = "running"

        def run_trial(split, config):
            budget.check()
            run = prepare_trial(store, record, split, config)
            if run["status"] == "completed":
                from backend.provenance import validate_snapshot

                validate_snapshot(store, run)
                return
            retry = bool(run["rows"])
            if retry:
                run["attempt"] += 1
                store.save("run", run, run["id"])
            _execute_run(store, run["id"], retry=retry)
            budget.check()

        def tuning():
            # Selection is immutable once committed; recovery never retunes after holdout.
            if "selection" in record:
                return
            for config in [record["plan"]["baseline"], *record["plan"]["candidates"]]:
                run_trial("tuning", config)

        def choose():
            record["tuning_results"] = [
                trial_summary(store, t) for t in record["trials"] if t["split"] == "tuning"
            ]
            if "selection" not in record:
                request = record["plan"]["request"]
                record["selection"] = select(
                    record["tuning_results"], request["objective"], request["quality_threshold"]
                )

        def heldout():
            selected = record["selection"]["selected"]
            if not selected or not record["plan"]["split"]["heldout"]:
                return
            configs = [record["plan"]["baseline"], *record["plan"]["candidates"]]
            for config in configs:
                if config["id"] in {"baseline", selected}:
                    run_trial("heldout", config)

        def report():
            record["heldout_results"] = [
                trial_summary(store, t) for t in record["trials"] if t["split"] == "heldout"
            ]
            record["status"] = "completed"

        builder = StateGraph(ExperimentState)
        names = [
            "validate",
            "execute_tuning",
            "select_candidate",
            "evaluate_heldout",
            "persist_report",
        ]
        for name, action in zip(names, [validate, tuning, choose, heldout, report]):
            builder.add_node(name, node(name, action))
        for left, right in zip([START, *names], [*names, END]):
            builder.add_edge(left, right)
        try:
            with budget_context(budget), checkpointer(store) as saver:
                builder.compile(checkpointer=saver).invoke(
                    {
                        "experiment_id": experiment_id,
                        "plan_fingerprint": record["plan_fingerprint"],
                        "stage": "validate",
                    },
                    {"configurable": {"thread_id": "optimization-" + experiment_id}},
                )
        except RunCancelled:
            record.update(
                status="budget_exhausted" if budget.exhausted else "cancelled",
                stage="Stopped; completed work preserved",
            )
        except Exception as exc:
            record.update(
                status="failed",
                stage="Interrupted; explicit resume required",
                error=type(exc).__name__,
            )
        finally:
            # Report completed and failed work even when interrupted before selection.
            record["tuning_results"] = [
                trial_summary(store, t)
                for t in record["trials"]
                if t["split"] == "tuning" and _exists(store, t["run_id"])
            ]
            record["heldout_results"] = [
                trial_summary(store, t)
                for t in record["trials"]
                if t["split"] == "heldout" and _exists(store, t["run_id"])
            ]
            save()
        return record


def _exists(store, identity):
    try:
        store.get("run", identity)
        return True
    except KeyError:
        return False


def public(record):
    result = deepcopy(record)
    result["plan"].pop("documents", None)
    result["plan"].pop("questions", None)
    return result
