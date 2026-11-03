"""share_agent_family: the family workspace roster.

Adds members (by id, with email/phone identifiers) to the run's family, and
always returns the current roster. Invitations are simulated in the prototype:
adding a member is the invitation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import Context

MEMBER = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "id": {"type": "string", "description": "snake_case id, e.g. maya"},
        "name": {"type": "string", "description": "display name, e.g. Maya Rivera"},
        "email": {"type": "string", "description": "email identifier, if any"},
        "phone": {"type": "string", "description": "phone identifier, if any"},
        "role": {"type": "string", "description": "e.g. parent, child, member"},
        "partner": {"type": "string", "description": "partner's member id, if any"},
        "parents": {
            "type": "array",
            "items": {"type": "string"},
            "description": "parents' member ids, if known",
        },
    },
    "required": ["id", "name"],
}

DEFINITION = {
    "type": "function",
    "name": "share_agent_family",
    "description": "Associate the shared agent with the family workspace: add members "
    "by id with their email or phone identifiers, and see the current roster. "
    "Call with no members to list the roster.",
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "members": {"type": "array", "items": MEMBER},
        },
        "required": [],
    },
}


def run(ctx: Context, args: dict) -> dict:
    added = []
    for raw in args.get("members") or []:
        if not isinstance(raw, dict):
            return {"error": "each member must be an object"}
        member_id = str(raw.get("id", ""))
        added.append(
            ctx.store.add_member(
                member_id,
                str(raw.get("name", "")),
                email=str(raw.get("email", "") or ""),
                phone=str(raw.get("phone", "") or ""),
                role=str(raw.get("role", "") or "member"),
            )
        )
        if raw.get("partner") or raw.get("parents"):
            ctx.store.set_relations(
                member_id,
                partner=str(raw.get("partner", "") or ""),
                parents=raw.get("parents") or [],
            )
    return {"family_id": ctx.family_id, "members": ctx.store.members()}
