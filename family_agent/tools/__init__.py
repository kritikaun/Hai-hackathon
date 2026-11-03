"""The family agent's tools, one file per tool: its DEFINITION (what the model sees)
and run(ctx, args) (what happens when the model calls it).

    tools.definitions(root, run_id)        # the tools to offer the model in this run
    result = tools.run(client, root, run_id, call)   # run the tool the model chose
    result.get("status", "running")        # "running", or why the run must stop

A refused call comes back as {"error": ...} so the model can correct it. A result
carries "status" only when the run must stop: "needs_input" (a person must
review a proposed action).

Visibility and authorization are enforced by the family store on every call.
With verification on, request_action_approval additionally re-checks that every
recipient is still authorized for every cited memory before recording the
approval.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from ..store import FamilyStore
from agent_kit.progress import load_progress
from . import (
    draft_family_message,
    get_due_reminders,
    get_family_events,
    request_action_approval,
    retrieve_family_memories,
    save_family_memory,
    share_agent_family,
)

FAMILY_TOOLS = [
    "share_agent_family",
    "get_due_reminders",
    "get_family_events",
    "retrieve_family_memories",
    "save_family_memory",
    "draft_family_message",
    "request_action_approval",
]

MODULES: dict[str, Any] = {
    str(m.DEFINITION["name"]): m
    for m in (
        share_agent_family,
        get_due_reminders,
        get_family_events,
        retrieve_family_memories,
        save_family_memory,
        draft_family_message,
        request_action_approval,
    )
}
DEFINITIONS: dict[str, dict] = {
    name: module.DEFINITION for name, module in MODULES.items()
}


@dataclass
class Context:
    """What a family tool gets besides its arguments."""

    client: OpenAI
    root: Path
    run_id: str
    family_id: str
    member_id: str
    store: FamilyStore
    verification: bool


def names(verification: bool) -> list:
    return list(FAMILY_TOOLS)


def definitions(root: Path, run_id: str) -> list:
    """The tool definitions to offer the model in this run."""
    verification = bool(load_progress(root, run_id).get("verification"))
    return [DEFINITIONS[name] for name in names(verification)]


def run(client: OpenAI, root: Path, run_id: str, call: Any) -> dict:
    """Run the tool the model chose (an item of response.output, or the same as a dict)."""
    name, raw = _field(call, "name"), _field(call, "arguments") or "{}"
    try:
        args = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (TypeError, ValueError) as error:
        return {"error": f"tool arguments are not valid JSON: {error}"}
    if not isinstance(args, dict):
        return {"error": "tool arguments must be a JSON object"}
    root = Path(root)
    try:
        identity = json.loads(
            (root / "workspace/runs" / run_id / "family.json").read_text()
        )
        family_id, member_id = identity["family_id"], identity["member_id"]
    except (OSError, ValueError, KeyError):
        return {"error": f"{run_id} is not a family run"}
    verification = bool(load_progress(root, run_id).get("verification"))
    if name not in names(verification):
        return {"error": f"unknown tool: {name!r}"}
    try:
        store = FamilyStore(root, family_id)
        store.require_member(member_id)
    except Exception as error:
        return {"error": str(error) or type(error).__name__}
    ctx = Context(client, root, run_id, family_id, member_id, store, verification)
    try:
        result = MODULES[name].run(ctx, args)
    except Exception as error:  # the model reads the error and tries again
        return {"error": str(error) or type(error).__name__}
    return result if isinstance(result, dict) else {"result": result}


def _field(call: Any, key: str) -> Any:
    return call.get(key) if isinstance(call, dict) else getattr(call, key, None)
