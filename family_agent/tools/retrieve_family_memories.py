"""retrieve_family_memories: search family memories for this conversation.

Only memories the requesting member is authorized for come back: every shared
memory, plus their own private and restricted ones. Another member's private
or restricted memories are never listed, quoted, or hinted at.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import Context

DEFINITION = {
    "type": "function",
    "name": "retrieve_family_memories",
    "description": "Search family memories relevant to the current request. "
    "Only memories the member in this conversation is authorized to see come back.",
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "query": {
                "type": "string",
                "description": "words to match against memory text and people (default: all visible memories)",
            },
        },
        "required": [],
    },
}


def run(ctx: Context, args: dict) -> dict:
    memories = ctx.store.visible_memories(
        ctx.member_id, query=str(args.get("query", "") or "")
    )
    return {
        "memories": [
            {
                "id": m["id"],
                "category": m["category"],
                "owner": m["owner"],
                "text": m["text"],
                "people": m["people"],
                "event_date": m["event_date"],
            }
            for m in memories
        ]
    }
