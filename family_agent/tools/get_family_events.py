"""get_family_events: upcoming birthdays, anniversaries, and family events.

Only events the requesting member is authorized for come back: shared events
for every member, private or restricted events for the owner plus the linked
memory's shared_with. Surprise and birthday honorees never receive their own
event.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import Context

DEFINITION = {
    "type": "function",
    "name": "get_family_events",
    "description": "Retrieve upcoming family events (birthdays, anniversaries, "
    "celebrations) visible to the member in this conversation, soonest first.",
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "from_date": {
                "type": "string",
                "description": "only events on or after this YYYY-MM-DD date (default: today)",
            },
            "limit": {
                "type": "integer",
                "description": "at most this many events (default 10)",
            },
        },
        "required": [],
    },
}


def run(ctx: Context, args: dict) -> dict:
    events = ctx.store.upcoming_events(
        ctx.member_id,
        ref_date=str(args.get("from_date", "") or ""),
        limit=args.get("limit", 10),
    )
    return {
        "events": [
            {
                "id": e["id"],
                "title": e["title"],
                "date": e["date"],
                "kind": e["kind"],
            }
            for e in events
        ]
    }
