"""save_family_memory: store a confirmed memory (README workflow step 4).

Private memories store immediately for this member. Family-shared memories are
refused until the member confirms the text and its visibility in this
conversation (confirmed=true). Restricted memories stay visible to this member
only. A memory with an event date also creates a linked event.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import Context

DEFINITION = {
    "type": "function",
    "name": "save_family_memory",
    "description": "Save a memory for the member in this conversation: private "
    "(just them), shared (the whole family, only after they confirm it in this "
    "conversation), or restricted (kept from other members, e.g. a surprise plan).",
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "text": {"type": "string", "description": "the fact to remember"},
            "category": {
                "type": "string",
                "description": "private, shared, or restricted",
            },
            "people": {
                "type": "array",
                "items": {"type": "string"},
                "description": "member ids this memory is about",
            },
            "event_date": {
                "type": "string",
                "description": "YYYY-MM-DD when this memory is also an event",
            },
            "event_title": {
                "type": "string",
                "description": "title for the linked event (default: from the text)",
            },
            "event_kind": {
                "type": "string",
                "description": "e.g. birthday, anniversary, celebration",
            },
            "confirmed": {
                "type": "boolean",
                "description": "true when the member confirmed a shared memory in this conversation",
            },
            "shared_with": {
                "type": "array",
                "items": {"type": "string"},
                "description": "member ids the owner explicitly authorizes (private or restricted only)",
            },
        },
        "required": ["text", "category"],
    },
}


def run(ctx: Context, args: dict) -> dict:
    memory = ctx.store.save_memory(
        owner=ctx.member_id,
        category=str(args.get("category", "")),
        text=str(args.get("text", "")),
        people=args.get("people") or [],
        event_date=str(args.get("event_date", "") or ""),
        event_title=str(args.get("event_title", "") or ""),
        event_kind=str(args.get("event_kind", "") or ""),
        confirmed=bool(args.get("confirmed", False)),
        shared_with=args.get("shared_with") or [],
    )
    return {
        "saved": {
            "id": memory["id"],
            "category": memory["category"],
            "event_date": memory["event_date"],
        }
    }
