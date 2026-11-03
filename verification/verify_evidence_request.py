"""Pre-built: is an evidence request allowed? Called by the request_evidence tool.

    run_verification(ctx, request) -> {"passed": bool, "feedback": str}

request = {"item", "need"}. The item must be on the checklist and not already proven, and
every visible document that is not flagged confidential must have been read. The
feedback says why a refused request is refused (the agent sees it).
"""

from __future__ import annotations

from agent_kit import document_text


def run_verification(ctx, request: dict) -> dict:
    case, item = ctx.case, str(request.get("item", ""))
    entry = case.data["checklist"].get(item)
    if entry is None:
        return _refuse(f"unknown checklist item {item!r}")
    if entry["status"] == "proven":
        return _refuse(f"{item} is already proven; do not escalate it")
    read = case.data.get("read_ids", [])
    visible = [d["id"] for d in document_text.list_documents(ctx.root)]
    unread = [d for d in visible if d not in read and not case.flagged(d)]
    if unread:
        return _refuse(
            f"read every visible document before escalating; unread: {', '.join(unread)}"
        )
    return {"passed": True, "feedback": ""}


def _refuse(feedback: str) -> dict:
    return {"passed": False, "feedback": feedback}
