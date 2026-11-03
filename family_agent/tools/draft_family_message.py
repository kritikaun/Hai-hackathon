"""draft_family_message: draft a reminder or message for named recipients.

Recipients must be family members, and every recipient must be authorized for
every cited memory. A draft that would disclose a restricted or another
member's private memory to someone unauthorized is refused, without leaking
the protected content.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..store import PermissionDeniedError

if TYPE_CHECKING:
    from . import Context

DEFINITION = {
    "type": "function",
    "name": "draft_family_message",
    "description": "Draft a reminder or message for named family members. "
    "Cite the memory ids the draft is based on; every recipient must be "
    "authorized for each of them.",
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "recipients": {
                "type": "array",
                "items": {"type": "string"},
                "description": "member ids who should receive this",
            },
            "subject": {"type": "string", "description": "short subject line"},
            "body": {"type": "string", "description": "the message text"},
            "memory_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "ids of the memories this draft is based on",
            },
        },
        "required": ["recipients", "body"],
    },
}


def check_recipients(ctx: Context, recipients: list, memory_ids: list) -> list:
    """Validated recipients, after authorizing each for each cited memory."""
    recipients = [str(r).strip() for r in recipients if str(r).strip()]
    if not recipients:
        raise ValueError("a draft needs at least one recipient")
    for recipient in recipients:
        ctx.store.require_member(recipient)
    by_id = {m["id"]: m for m in ctx.store.data["memories"]}
    for mid in [str(m).strip() for m in memory_ids if str(m).strip()]:
        if mid not in by_id:
            raise ValueError(f"unknown memory {mid!r}")
        for recipient in recipients:
            if not ctx.store.authorized_for(recipient, by_id[mid]):
                raise PermissionDeniedError(
                    f"{recipient} is not authorized for memory {mid}"
                )
    return recipients


def run(ctx: Context, args: dict) -> dict:
    body = str(args.get("body", "")).strip()
    if not body:
        return {"error": "the draft body is empty"}
    recipients = check_recipients(
        ctx, args.get("recipients") or [], args.get("memory_ids") or []
    )
    return {
        "draft": {
            "recipients": recipients,
            "subject": str(args.get("subject", "") or "").strip(),
            "body": body,
            "memory_ids": [str(m) for m in (args.get("memory_ids") or [])],
        },
        "authorized": True,
    }
