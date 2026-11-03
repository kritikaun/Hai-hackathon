"""request_action_approval: propose an action for a person to review.

Records the action, recipients, and message details as a pending approval and
stops the run (needs_input) until a family member approves or rejects it. With
verification on, every recipient is re-checked against every cited memory at
this point, so an approval cannot go out on stale authorization.

An approved share_memory action grants the recipients access to the cited
memories (only the owner can propose sharing their memory). Other actions have
no delivery mechanism in the prototype: approval means "demonstrate it", never
"it was sent".

A schedule_reminder with reminder_key/title/date records a proactive reminder
that stays proposed until decided; each key is proposed at most once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .draft_family_message import check_recipients

if TYPE_CHECKING:
    from . import Context

DEFINITION = {
    "type": "function",
    "name": "request_action_approval",
    "description": "Propose an action (send a message, share a memory, schedule "
    "a reminder) for a family member to review. The run pauses until someone "
    "approves or rejects it.",
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {
                "type": "string",
                "description": "send_message, share_memory, or schedule_reminder",
            },
            "recipients": {
                "type": "array",
                "items": {"type": "string"},
                "description": "member ids affected by this action",
            },
            "message": {
                "type": "string",
                "description": "the message or action details to review",
            },
            "memory_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "ids of the memories this action is based on",
            },
            "note": {
                "type": "string",
                "description": "why this action, for the reviewer",
            },
            "reminder_key": {
                "type": "string",
                "description": "for schedule_reminder: the key from get_due_reminders",
            },
            "reminder_title": {
                "type": "string",
                "description": "for schedule_reminder: the reminder title",
            },
            "reminder_date": {
                "type": "string",
                "description": "for schedule_reminder: the event date (YYYY-MM-DD)",
            },
        },
        "required": ["action"],
    },
}


def run(ctx: Context, args: dict) -> dict:
    recipients = list(args.get("recipients") or [])
    memory_ids = list(args.get("memory_ids") or [])
    if ctx.verification:
        recipients = check_recipients(ctx, recipients, memory_ids)
    reminder_key = str(args.get("reminder_key", "") or "").strip()
    if str(args.get("action", "")) == "schedule_reminder" and reminder_key:
        proposed = ctx.store.propose_reminder(
            run_id=ctx.run_id,
            requested_by=ctx.member_id,
            key=reminder_key,
            title=str(args.get("reminder_title", "") or ""),
            date=str(args.get("reminder_date", "") or ""),
            recipients=recipients,
            message=str(args.get("message", "") or ""),
            memory_ids=memory_ids,
        )
        try:
            from ..invitations import attach_generated_image

            attach_generated_image(ctx.store, ctx.client, proposed["reminder"], run_id=ctx.run_id)
        except Exception:
            pass
        approval = proposed["approval"]
        return {
            "approval": {
                "id": approval["id"],
                "action": approval["action"],
                "recipients": approval["recipients"],
                "status": approval["status"],
            },
            "reminder": proposed["reminder"]["id"],
            "status": "needs_input",
            "note": "A family member must approve or reject this reminder before invitations are sent.",
        }
    approval = ctx.store.request_approval(
        run_id=ctx.run_id,
        requested_by=ctx.member_id,
        action=str(args.get("action", "")),
        recipients=recipients,
        message=str(args.get("message", "") or ""),
        memory_ids=memory_ids,
        note=str(args.get("note", "") or ""),
    )
    return {
        "approval": {
            "id": approval["id"],
            "action": approval["action"],
            "recipients": approval["recipients"],
            "status": approval["status"],
        },
        "status": "needs_input",
        "note": "A family member must approve or reject this action before anything happens.",
    }
