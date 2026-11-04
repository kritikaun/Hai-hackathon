"""Proactive reminders: due family events and public-holiday gathering proposals.

due_reminders(store, member_id, ref_date="", days_ahead=30) says what the
member should hear about: family events in the window, with recipients chosen
by kind (birthdays skip the birthday person, secrets stay among those in the
know, shared celebrations go to everyone), plus upcoming US public holidays
for the whole family. Only items listing the member as a recipient come back,
and anything already proposed, approved, or rejected is never repeated.
"""

from __future__ import annotations

import datetime

from .holidays import upcoming_holidays

REMIND_DAYS_BEFORE = {"birthday": 7, "anniversary": 7, "surprise_party": 3}
HOLIDAY_REMIND_DAYS_BEFORE = 7


def _parse(ref_date: str) -> datetime.date:
    ref_date = (ref_date or "").strip()
    if not ref_date:
        return datetime.date.today()
    try:
        return datetime.date.fromisoformat(ref_date)
    except ValueError:
        raise ValueError(f"date {ref_date!r} must be YYYY-MM-DD") from None


from .store import _recipients_for_event


def _suggest_event(title: str, date: str, days: int, reason: str, kind: str = "") -> str:
    """Follow-up body text only: no title, date, timing, or recipients.

    The card already shows those once, computed from the same date.
    """
    if kind == "birthday":
        return "Time to plan the celebration — pick a place and confirm who's coming."
    if kind == "anniversary":
        return "Plan something special and confirm who's coming."
    if kind == "surprise_party":
        return "Coordinate quietly with the helpers — no hints to the honoree."
    return "Coordinate where it will happen and who is coming."


def due_reminders(
    store, member_id: str, ref_date: str = "", days_ahead: int = 30
) -> dict:
    store.require_member(member_id)
    ref = _parse(ref_date)
    try:
        days_ahead = max(1, min(int(days_ahead), 366))
    except (TypeError, ValueError):
        days_ahead = 30
    end = ref + datetime.timedelta(days=days_ahead)
    reminded = {r["key"] for r in store.data.get("reminders", [])}
    items: list = []
    for event in store.data.get("events", []):
        try:
            date = datetime.date.fromisoformat(event["date"])
        except (ValueError, KeyError, TypeError):
            continue
        if not (ref <= date <= end):
            continue
        key = f"event:{event['id']}"
        if key in reminded:
            continue
        recipients, reason = _recipients_for_event(store, event)
        if member_id not in recipients:
            continue
        days = (date - ref).days
        before = REMIND_DAYS_BEFORE.get(event.get("kind") or "", 7)
        items.append(
            {
                "key": key,
                "type": "event",
                "title": event["title"],
                "date": event["date"],
                "kind": event.get("kind") or "",
                "recipients": recipients,
                "reason": reason,
                "status": "due" if days <= before else "upcoming",
                "remind_on": (date - datetime.timedelta(days=before)).isoformat(),
                "suggested_message": _suggest_event(
                    event["title"], event["date"], days, reason,
                    event.get("kind") or "",
                ),
            }
        )
    for holiday in upcoming_holidays(ref.isoformat(), days_ahead):
        date = datetime.date.fromisoformat(holiday["date"])
        days = (date - ref).days
        members = [m["id"] for m in store.members()]
        if days <= HOLIDAY_REMIND_DAYS_BEFORE:
            key = f"holiday:{holiday['date']}:remind"
            if key in reminded:
                continue
            items.append(
                {
                    "key": key,
                    "type": "public_holiday",
                    "title": holiday["title"],
                    "date": holiday["date"],
                    "kind": "public_holiday",
                    "recipients": members,
                    "reason": "a public holiday, the whole family",
                    "status": "due",
                    "remind_on": holiday["date"],
                    "suggested_message": (
                        "Shall the family get together? "
                        "Finalize where it will happen and who is coming."
                    ),
                }
            )
        else:
            key = f"holiday:{holiday['date']}:propose"
            if key in reminded:
                continue
            items.append(
                {
                    "key": key,
                    "type": "public_holiday",
                    "title": holiday["title"],
                    "date": holiday["date"],
                    "kind": "public_holiday",
                    "recipients": members,
                    "reason": "a public holiday, the whole family",
                    "status": "upcoming",
                    "remind_on": (
                        date
                        - datetime.timedelta(days=HOLIDAY_REMIND_DAYS_BEFORE)
                    ).isoformat(),
                    "suggested_message": (
                        "Shall we plan a family dinner? I can ask everyone "
                        "where it will happen and who is coming."
                    ),
                }
            )
    items.sort(key=lambda i: (i["date"], i["title"]))
    return {"as_of": ref.isoformat(), "member_id": member_id, "items": items}
