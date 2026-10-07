"""Enforce the measured credential-free suite baseline, rounded down to a whole percent."""

import json
import sys
from pathlib import Path

if __name__ == "__main__":
    baseline = json.loads(Path("tests/coverage-baseline.json").read_text())
    measured = json.loads(Path(sys.argv[1]).read_text())["totals"]["percent_covered"]
    if measured < baseline["minimum_percent"]:
        raise SystemExit(
            f"Coverage {measured:.2f}% is below measured baseline {baseline['minimum_percent']}%"
        )
    print(f"Coverage {measured:.2f}% meets the recorded baseline")
