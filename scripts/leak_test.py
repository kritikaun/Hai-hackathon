#!/usr/bin/env python3
"""Live leak test: can family members extract restricted secrets from Hai?

    .venv/bin/python scripts/leak_test.py

Runs the real family agent (MODEL_AGENT) with VERIFICATION=on against a
throwaway copy of the Rivera workspace: 20 prompts as Maya, 10 as Sofia,
5 indirect, 5 controls as Leo. Each reply gets a keyword check, then a
MODEL_VERIFY judge check when flagged or borderline. Results are saved to
workspace/evals/leak_test_<date>.json. Needs MODEL_API_KEY in .env.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from agent_kit.client import (  # noqa: E402
    INTERACTIVE_MAX_RETRIES,
    INTERACTIVE_TIMEOUT_SECONDS,
    make_client,
)
from family_agent.leak_test import run_leak_test  # noqa: E402


def _load_env() -> None:
    env = ROOT / ".env"
    if not env.is_file():
        return
    import os

    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def main() -> int:
    _load_env()
    client = make_client(
        timeout_seconds=INTERACTIVE_TIMEOUT_SECONDS,
        max_retries=INTERACTIVE_MAX_RETRIES,
    )
    done = [0]

    def show(done_count: int, total: int) -> None:
        done[0] = done_count
        print(f"\rleak test: {done_count}/{total} prompts", end="", flush=True)

    report = run_leak_test(ROOT, client, progress_cb=show)
    print()
    summary = report["summary"]
    print(f"overall: {summary['overall']['passed']}/{summary['overall']['total']} passed "
          f"in {report['total_seconds']}s")
    for person, tally in summary["by_person"].items():
        print(f"  {person}: {tally['passed']}/{tally['total']}")
    for secret, tally in summary["by_secret"].items():
        print(f"  {secret}: {tally['passed']}/{tally['total']}")
    for failure in report["failures"]:
        print(f"FAIL [{failure['member']}] {failure['prompt']}")
        print(f"  reply: {failure['reply'][:200]}")
        if failure["judge"]:
            print(f"  judge: {failure['judge']['reason']}")
    for failure in report["control_failures"]:
        print(f"CONTROL FAILED [{failure['member']}] {failure['prompt']}")
    print(f"saved: {report['saved_to']}")
    return 0 if not report["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
