"""get_due_reminders: what this member should be proactively reminded about.

Family events in the window (birthdays skip the birthday person, secrets stay
among those in the know, shared celebrations go to everyone) plus upcoming US
public holidays for the whole family. Only items listing this member as a
recipient come back, each with a key to propose it for approval.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..reminders import due_reminders

if TYPE_CHECKING:
    from . import Context

DEFINITION = {
    "type": "function",
    "name": "get_due_reminders",
    "description": "List proactive reminders for the member in this conversation: "
    "upcoming family events and public holidays, each with its recipients. "
    "Propose one with request_action_approval (schedule_reminder).",
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "days_ahead": {
                "type": "integer",
                "description": "look this many days ahead (default 30)",
            },
        },
        "required": [],
    },
}


def run(ctx: Context, args: dict) -> dict:
    result = due_reminders(
        ctx.store, ctx.member_id, days_ahead=args.get("days_ahead", 30)
    )
    return {"as_of": result["as_of"], "reminders": result["items"]}
