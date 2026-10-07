"""Assemble validated digest records; schema is shared by publishing and deployment."""

import argparse
import json
import re
from pathlib import Path


def validate(release: dict) -> dict:
    if release.get("format") != 1 or not re.fullmatch(r"[0-9a-f]{40}", release.get("revision", "")):
        raise ValueError("Release needs format=1 and an exact Git revision")
    if release.get("storage_schema") != 1 or release.get("checkpoint_serializer") != "json-v1":
        raise ValueError("Unsupported storage/checkpoint compatibility")
    if set(release.get("storage_backends", [])) != {"sqlite", "postgres"}:
        raise ValueError("Release must support both authoritative stores")
    if set(release.get("images", {})) != {"cpu", "compact", "dashboard"}:
        raise ValueError("Release requires all three tested images")
    for variant, image in release["images"].items():
        if not re.fullmatch(rf"ghcr.io/wauul/rag-bench-{variant}@sha256:[0-9a-f]{{64}}", image):
            raise ValueError("Unexpected registry or mutable image")
    if release.get("validation", {}).get("revision") != release["revision"]:
        raise ValueError("Validation does not describe the candidate revision")
    if any(
        release["validation"].get(name) != "success"
        for name in ("engineering", "models", "artifacts")
    ):
        raise ValueError("Missing, skipped or failed required validation")
    return release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, default=Path("release.json"))
    args = parser.parse_args()
    images = {}
    for name in ("cpu", "compact", "dashboard"):
        item = json.loads((args.directory / (name + ".json")).read_text())
        if item["revision"] != args.revision:
            raise ValueError("Image digest record belongs to another revision")
        images[name] = item["image"]
    release = validate(
        {
            "format": 1,
            "version": args.version,
            "revision": args.revision,
            "storage_schema": 1,
            "checkpoint_serializer": "json-v1",
            "storage_backends": ["sqlite", "postgres"],
            "images": images,
            "validation": {
                "revision": args.revision,
                "engineering": "success",
                "models": "success",
                "artifacts": "success",
            },
        }
    )
    args.output.write_text(json.dumps(release, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
