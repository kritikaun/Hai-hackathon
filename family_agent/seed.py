"""Seed family workspaces with the fictional demo families (or an empty one)."""

from __future__ import annotations

import datetime
from pathlib import Path

from . import demo_data
from .store import FamilyError, FamilyStore


def _write(root: Path, data: dict) -> dict:
    store = FamilyStore(root, data["family_id"])
    for member in data["members"]:
        store.add_member(
            member["id"],
            member["name"],
            email=member.get("email", ""),
            phone=member.get("phone", ""),
            role=member.get("role", "member"),
            preferred_language=member.get("preferred_language", "English"),
        )
    for memory in data["memories"]:
        store.save_memory(
            owner=memory["owner"],
            category=memory["category"],
            text=memory["text"],
            people=memory.get("people", []),
            event_date=memory.get("event_date", ""),
            event_title=memory.get("event_title", ""),
            event_kind=memory.get("event_kind", ""),
            confirmed=memory.get("confirmed", False),
            shared_with=memory.get("shared_with", []),
        )
    for relation in data.get("relations", []):
        store.set_relations(
            relation["id"],
            partner=relation.get("partner", ""),
            parents=relation.get("parents", []),
        )
    return {
        "family_id": data["family_id"],
        "members": len(store.data["members"]),
        "memories": len(store.data["memories"]),
        "events": len(store.data["events"]),
    }


def seed_family(
    root: Path, family_id: str = demo_data.FAMILY_ID, today: datetime.date | None = None
) -> dict:
    if (Path(root) / "workspace/families" / f"{family_id}.json").is_file():
        raise FamilyError(f"family {family_id} is already seeded")
    return _write(Path(root), demo_data.demo_family_for(family_id, today))


def seed_all(root: Path, today: datetime.date | None = None) -> dict:
    """Seed every demo family that is not seeded yet; skip the rest."""
    root = Path(root)
    seeded, skipped = [], []
    for family_id in demo_data.DEMO_IDS:
        if (root / "workspace/families" / f"{family_id}.json").is_file():
            skipped.append(family_id)
        else:
            seeded.append(_write(root, demo_data.demo_family_for(family_id, today)))
    return {"seeded": seeded, "skipped": skipped}
