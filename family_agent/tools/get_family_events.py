"""get_family_events: upcoming birthdays, anniversaries, and family events.

Only events addressed to the requesting member come back: birthdays skip the
birthday person, anniversaries skip the couple, secrets stay among those in
the know, shared celebrations go to everyone. Anything hidden from the member
is never sent.
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
