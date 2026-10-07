"""Explicit metadata-only export of a saved optimizer experiment; never execute trials.

Dataset items contain fingerprints and numeric indexes only, never user inputs or
expected answers. Local trial records and selection remain authoritative.
"""

import argparse
import os

from backend.observability import enabled, get_client, identity, metadata
from backend.provenance import digest
from backend.storage import Store


def export(store, experiment_id, max_items=128):
    if os.getenv("RAGBENCH_LANGFUSE_EXPERIMENTS") != "metadata" or not enabled("optimization"):
        raise ValueError(
            "Explicit metadata experiment export and verified Langfuse configuration required"
        )
    record = store.get("optimization", experiment_id)
    runs = [(trial, store.get("run", trial["run_id"])) for trial in record["trials"]]
    if sum(len(run["questions"]) for _, run in runs) > max_items:
        raise ValueError("Export exceeds the explicit item ceiling")
    api = get_client().api
    options = {"timeout_in_seconds": 2, "max_retries": 0}
    dataset = "ragbench-metadata-" + record["plan"]["dataset_fingerprint"]
    api.datasets.create(
        name=dataset, metadata={"kind": "fingerprint-slots-no-content"}, request_options=options
    )
    for trial, run in runs:
        for index in range(len(run["questions"])):
            item_id = digest([record["id"], trial["split"], trial["configuration_id"], index])
            api.dataset_items.create(
                dataset_name=dataset,
                id=item_id,
                input={"fingerprint": run["input_fingerprint"]},
                metadata=metadata(
                    {
                        "split": trial["split"],
                        "question_index": index,
                        "configuration_id": trial["configuration_id"],
                    }
                ),
                request_options=options,
            )
            api.dataset_run_items.create(
                run_name="optimization-" + record["id"],
                dataset_item_id=item_id,
                trace_id=identity("benchmark", run["id"]),
                metadata=metadata(
                    {
                        "optimization_id": record["id"],
                        "split": trial["split"],
                        "status": run["status"],
                    }
                ),
                request_options=options,
            )
    return {"experiment_id": record["id"], "items": sum(len(run["questions"]) for _, run in runs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id")
    parser.add_argument("--max-items", type=int, default=128)
    args = parser.parse_args()
    if not 1 <= args.max_items <= 128:
        raise ValueError("Export limit must be between 1 and 128")
    print(export(Store(), args.experiment_id, args.max_items))


if __name__ == "__main__":
    main()
