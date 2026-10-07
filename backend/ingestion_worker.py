"""Disposable PDF parser: bounded runtime; POSIX memory/CPU bounds where available."""

import json
import sys


def main():
    if sys.platform != "win32":
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (384 * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
    from backend.ingestion import MAX_BYTES, extract_document

    raw = sys.stdin.buffer.read(MAX_BYTES + 1)
    try:
        result = extract_document(sys.argv[1], raw)
        sys.stdout.write(json.dumps(result))
    except Exception:
        sys.exit(2)


if __name__ == "__main__":
    main()
