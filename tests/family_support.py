"""Shared helpers for the family-agent offline tests: a seeded demo family on disk."""

from __future__ import annotations

import datetime
import json
import tempfile
from pathlib import Path

from agent_kit import progress
from family_agent.seed import seed_family
from tests.fake_model import Scripted, ScriptedModel

TODAY = datetime.date(2026, 10, 30)
REF_DATE = TODAY.isoformat()


def make_family_root(today: datetime.date = TODAY):
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    seed_family(root, today=today)
    return root, tmp


def new_family_run(
    root: Path,
    family_id: str = "riveras",
    member_id: str = "maya",
    verification: bool = False,
) -> str:
    run_id = progress.new_run(
        root, instructions="test", objective="test", verification=verification
    )
    (root / "workspace/runs" / run_id / "family.json").write_text(
        json.dumps({"family_id": family_id, "member_id": member_id})
    )
    return run_id


def scripted_family(actions: list) -> Scripted:
    """A scripted client that plays (name, args) in order, one per agent turn."""
    queue = list(actions)

    def planner(request: dict):
        return queue.pop(0) if queue else None

    return Scripted(ScriptedModel({}, planner))


def family_store(root: Path, family_id: str = "riveras") -> dict:
    return json.loads(
        (root / "workspace/families" / f"{family_id}.json").read_text()
    )


def instructions_file() -> str:
    return (
        Path(__file__).absolute().parents[1]
        / "family_agent/system_instructions.md"
    ).read_text()
