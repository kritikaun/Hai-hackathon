"""Fictional demo family for the standalone prototype. Never real user data.

Dates are relative to the seed day so the demo always has upcoming events:
Sofia's birthday in 9 days, the parents' anniversary in 30.
"""

from __future__ import annotations

import datetime

FAMILY_ID = "riveras"
DEMO_IDS = ("riveras",)


def demo_family(today: datetime.date | None = None) -> dict:
    today = today or datetime.date.today()
    birthday = (today + datetime.timedelta(days=9)).isoformat()
    party = (today + datetime.timedelta(days=8)).isoformat()
    anniversary = (today + datetime.timedelta(days=30)).isoformat()
    return {
        "family_id": FAMILY_ID,
        "members": [
            {
                "id": "maya",
                "name": "Maya Rivera",
                "email": "maya@example.com",
                "phone": "+15550001111",
                "role": "mother",
            },
            {
                "id": "leo",
                "name": "Leo Rivera",
                "email": "leo@example.com",
                "phone": "+15550002222",
                "role": "father",
            },
            {
                "id": "sofia",
                "name": "Sofia Rivera",
                "email": "sofia@example.com",
                "phone": "+15550003333",
                "role": "sister",
            },
            {
                "id": "eli",
                "name": "Eli Rivera",
                "email": "eli@example.com",
                "phone": "+15550004444",
                "role": "brother",
            },
            {
                "id": "elena",
                "name": "Elena Rivera",
                "email": "elena@example.com",
                "phone": "+15550003030",
                "role": "grandmother",
                "preferred_language": "Spanish",
            },
            {
                "id": "mateo",
                "name": "Mateo Rivera",
                "email": "mateo@example.com",
                "phone": "+15550004040",
                "role": "grandfather",
                "preferred_language": "Spanish",
            },
            {
                "id": "carlos",
                "name": "Carlos Rivera",
                "email": "carlos@example.com",
                "phone": "+15550005555",
                "role": "uncle",
            },
            {
                "id": "lucia",
                "name": "Lucia Rivera",
                "email": "lucia@example.com",
                "phone": "+15550006666",
                "role": "aunt",
            },
        ],
        "memories": [
            {
                "owner": "maya",
                "category": "shared",
                "text": "Sofia turns 12 soon. The family celebrates birthdays at home.",
                "people": ["sofia"],
                "event_date": birthday,
                "event_title": "Sofia's 12th birthday",
                "event_kind": "birthday",
                "confirmed": True,
            },
            {
                "owner": "leo",
                "category": "shared",
                "text": "Leo and Maya's wedding anniversary is coming up.",
                "people": ["leo", "maya"],
                "event_date": anniversary,
                "event_title": "Leo and Maya's anniversary",
                "event_kind": "anniversary",
                "confirmed": True,
            },
            {
                "owner": "maya",
                "category": "shared",
                "text": "The family has a Sunday video call with Abuela Elena.",
                "people": [
                    "maya", "leo", "sofia", "eli", "elena", "mateo", "carlos", "lucia"
                ],
                "confirmed": True,
            },
            {
                "owner": "maya",
                "category": "private",
                "text": "Maya's book club meets on Thursdays.",
                "people": ["maya"],
            },
            {
                "owner": "sofia",
                "category": "private",
                "text": "Sofia is learning guitar and practices after school.",
                "people": ["sofia"],
            },
            {
                "owner": "leo",
                "category": "restricted",
                "text": (
                    "Leo is planning a surprise birthday party for Sofia at home. "
                    "Maya and Eli are invited to help. Sofia must not find out."
                ),
                "people": ["leo", "maya", "eli"],
                "event_date": party,
                "event_title": "Surprise party for Sofia (secret)",
                "event_kind": "surprise_party",
            },
        ],
        "relations": [
            {"id": "maya", "partner": "leo", "parents": ["elena", "mateo"]},
            {"id": "leo", "partner": "maya", "parents": []},
            {"id": "sofia", "parents": ["maya", "leo"]},
            {"id": "eli", "parents": ["maya", "leo"]},
            {"id": "elena", "partner": "mateo"},
            {"id": "mateo", "partner": "elena"},
            {"id": "carlos", "partner": "lucia", "parents": ["elena", "mateo"]},
            {"id": "lucia", "partner": "carlos", "parents": []},
        ],
    }


def demo_family_for(family_id: str, today: datetime.date | None = None) -> dict:
    if family_id == FAMILY_ID:
        return demo_family(today)
    return {"family_id": family_id, "members": [], "memories": []}
