"""The family-memory store: one JSON record per family at workspace/families/<family-id>.json.

Members, memories in three categories (private, shared, restricted), events, and
action approvals, written only through this module and append-audited. Every
method enforces visibility and authorization in code: the agent can only see or
use what the requesting member is authorized for, whatever the model asks for.
"""

from __future__ import annotations

import datetime
import json
import re
import uuid
from pathlib import Path
from typing import Any

from agent_kit.progress import now

CATEGORIES = ("private", "shared", "restricted")
ACTIONS = ("send_message", "share_memory", "schedule_reminder")
MEMBER_ID = re.compile(r"^[a-z][a-z0-9_]{0,47}$")


class FamilyError(ValueError):
    """A family-store refusal. The message is safe to show the model."""


class UnknownMemberError(FamilyError):
    """The member id is not part of this family."""


class PermissionDeniedError(FamilyError):
    """The member is not authorized for this memory, event, or action."""


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:6]}"


def _check_date(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    try:
        datetime.date.fromisoformat(value)
    except ValueError:
        raise FamilyError(f"date {value!r} must be YYYY-MM-DD") from None
    return value


def _recipients_for_event(store, event: dict) -> tuple[list, str]:
    """Who an event is for, and why: birthdays skip the birthday person,
    anniversaries skip the couple, secrets stay among those in the know,
    shared celebrations go to everyone."""
    members = [m["id"] for m in store.members()]
    linked = next(
        (m for m in store.data["memories"] if m["id"] == (event.get("memory_id") or "")),
        None,
    )
    people = linked["people"] if linked else []
    owner = event.get("owner", "")
    visibility = event.get("visibility", "shared")
    if visibility == "private":
        return [owner], f"a private event, only for {owner}"
    if visibility == "restricted" or event.get("kind") == "surprise_party":
        know = [owner] + [
            m for m in (linked.get("shared_with", []) if linked else []) if m != owner
        ]
        return know, "a secret, only for those in the know"
    kind = event.get("kind") or ""
    if kind in ("birthday", "anniversary"):
        excluded = [p for p in people if p in members]
        detail = " and ".join(excluded) if excluded else "the honoree"
        return (
            [m for m in members if m not in excluded],
            f"{kind}, everyone except {detail}",
        )
    return list(members), "a family celebration, the whole family"


class FamilyStore:
    def __init__(self, root: Path, family_id: str) -> None:
        family_id = (family_id or "").strip()
        if not MEMBER_ID.match(family_id):
            raise FamilyError(
                f"family id {family_id!r} must be snake_case, e.g. riveras"
            )
        self.root = Path(root)
        self.family_id = family_id
        self.path = self.root / "workspace/families" / f"{family_id}.json"
        if self.path.is_file():
            self.data: dict[str, Any] = json.loads(self.path.read_text())
            for _key, _default in (
                ("members", {}),
                ("memories", []),
                ("events", []),
                ("approvals", []),
                ("direct", []),
                ("reminders", []),
                ("audit", []),
            ):
                self.data.setdefault(_key, _default)
            for member in self.data.get("members", {}).values():
                member.setdefault("partner", "")
                member.setdefault("parents", [])
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.data = {
                "family_id": family_id,
                "created": now(),
                "members": {},
                "memories": [],
                "events": [],
                "approvals": [],
                "direct": [],
                "reminders": [],
                "audit": [],
            }
            self._save("created", "system", {})

    # ---- persistence ----
    def _save(self, action: str, actor: str, detail: dict) -> None:
        self.data["audit"].append(
            {"at": now(), "actor": actor, "action": action, "detail": detail}
        )
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(self.path)

    # ---- members ----
    def add_member(
        self,
        member_id: str,
        name: str,
        email: str = "",
        phone: str = "",
        role: str = "member",
        preferred_language: str = "English",
    ) -> dict:
        member_id = (member_id or "").strip()
        if not MEMBER_ID.match(member_id):
            raise FamilyError(
                f"member id {member_id!r} must be snake_case, e.g. maya"
            )
        if member_id in self.data["members"]:
            raise FamilyError(f"{member_id} is already a member of this family")
        name = (name or "").strip()
        if not name:
            raise FamilyError(f"{member_id}: name is empty")
        member = {
            "id": member_id,
            "name": name,
            "email": (email or "").strip(),
            "phone": (phone or "").strip(),
            "role": (role or "member").strip() or "member",
            "preferred_language": (preferred_language or "English").strip() or "English",
            "partner": "",
            "parents": [],
        }
        self.data["members"][member_id] = member
        self._save("add_member", member_id, {"member_id": member_id})
        return dict(member)

    def members(self) -> list:
        return [dict(m) for m in self.data["members"].values()]

    def require_member(self, member_id: str) -> dict:
        try:
            return self.data["members"][member_id]
        except KeyError:
            raise UnknownMemberError(
                f"{member_id!r} is not a member of this family"
            ) from None

    def set_relations(
        self, member_id: str, partner: str = "", parents: list | tuple = ()
    ) -> dict:
        """Set a member's partner and parents (ids must already be members)."""
        member = self.require_member(member_id)
        partner = (partner or "").strip()
        if partner:
            if partner == member_id:
                raise FamilyError("a member cannot partner themselves")
            self.require_member(partner)
        clean = []
        for parent in [str(p).strip() for p in parents if str(p).strip()]:
            if parent == member_id:
                raise FamilyError("a member cannot parent themselves")
            self.require_member(parent)
            if parent not in clean:
                clean.append(parent)
        member["partner"] = partner
        member["parents"] = clean
        self._save(
            "set_relations", member_id, {"partner": partner, "parents": clean}
        )
        return dict(member)

    # ---- memories ----
    def save_memory(
        self,
        *,
        owner: str,
        category: str,
        text: str,
        people: list | tuple = (),
        event_date: str = "",
        event_title: str = "",
        event_kind: str = "",
        confirmed: bool = False,
        shared_with: list | tuple = (),
    ) -> dict:
        self.require_member(owner)
        if category not in CATEGORIES:
            raise FamilyError(
                f"category {category!r} must be one of: {', '.join(CATEGORIES)}"
            )
        text = (text or "").strip()
        if not text:
            raise FamilyError("memory text is empty")
        if category == "shared" and not confirmed:
            raise FamilyError(
                "a family-shared memory needs the member's confirmation first: "
                "ask them to confirm the text and that it may be shared, "
                "then save again with confirmed=true"
            )
        date = _check_date(event_date)
        granted = []
        for mid in [str(m).strip() for m in shared_with if str(m).strip()]:
            self.require_member(mid)
            if mid != owner and mid not in granted:
                granted.append(mid)
        memory = {
            "id": _new_id("mem"),
            "category": category,
            "owner": owner,
            "text": text,
            "people": [str(p).strip() for p in people if str(p).strip()],
            "event_date": date,
            "shared_with": granted,
            "created": now(),
        }
        self.data["memories"].append(memory)
        self._save("save_memory", owner, {"id": memory["id"], "category": category})
        if date:
            title = (event_title or "").strip() or f"Event: {text[:60]}"
            self.add_event(
                owner=owner,
                title=title,
                date=date,
                kind=(event_kind or "").strip(),
                visibility=category,
                memory_id=memory["id"],
            )
        return dict(memory)

    def authorized_for(self, member_id: str, memory: dict) -> bool:
        if memory.get("category") == "shared":
            return True
        if memory.get("owner") == member_id:
            return True
        return member_id in (memory.get("shared_with") or [])

    def get_memory(self, member_id: str, memory_id: str) -> dict:
        self.require_member(member_id)
        memory = next(
            (m for m in self.data["memories"] if m["id"] == memory_id), None
        )
        if memory is None:
            raise FamilyError(f"unknown memory {memory_id!r}")
        if not self.authorized_for(member_id, memory):
            raise PermissionDeniedError(
                f"{member_id} is not authorized for memory {memory_id}"
            )
        return dict(memory)

    def visible_memories(self, member_id: str, query: str = "") -> list:
        self.require_member(member_id)
        words = [w for w in (query or "").lower().split() if w]
        found = []
        for memory in self.data["memories"]:
            if not self.authorized_for(member_id, memory):
                continue
            if words:
                haystack = f"{memory['text']} {' '.join(memory['people'])}".lower()
                if not all(w in haystack for w in words):
                    continue
            found.append(dict(memory))
        return found

    # ---- events ----
    def add_event(
        self,
        *,
        owner: str,
        title: str,
        date: str,
        kind: str = "",
        visibility: str = "shared",
        memory_id: str = "",
    ) -> dict:
        self.require_member(owner)
        if visibility not in CATEGORIES:
            raise FamilyError(
                f"visibility {visibility!r} must be one of: {', '.join(CATEGORIES)}"
            )
        title = (title or "").strip()
        if not title:
            raise FamilyError("event title is empty")
        event = {
            "id": _new_id("ev"),
            "title": title,
            "date": _check_date(date),
            "kind": (kind or "").strip(),
            "visibility": visibility,
            "owner": owner,
            "memory_id": memory_id,
        }
        if not event["date"]:
            raise FamilyError("event date is empty")
        self.data["events"].append(event)
        self._save("add_event", owner, {"id": event["id"], "date": event["date"]})
        return dict(event)

    def _linked_event_memory(self, event: dict) -> dict | None:
        """The memory an event was created from, if it is still stored."""
        memory_id = event.get("memory_id") or ""
        if not memory_id:
            return None
        return next(
            (m for m in self.data.get("memories", []) if m.get("id") == memory_id),
            None,
        )

    def _event_honorees(self, event: dict) -> set:
        """Members a birthday or surprise event is for, who never see it."""
        members = {m["id"]: m for m in self.members()}
        honorees = set()
        words = re.split(r"[^a-z]+", str(event.get("title") or "").lower())
        for mid, m in members.items():
            first = str(m.get("name") or "").split()
            first = first[0].lower() if first else ""
            if len(first) >= 3 and first in words:
                honorees.add(mid)
        if event.get("kind") == "birthday":
            linked = self._linked_event_memory(event)
            people = linked.get("people", []) if linked else []
            honorees.update(p for p in people if p in members)
        return honorees

    def authorized_for_event(self, member_id: str, event: dict) -> bool:
        """The same rule as memories: shared goes to everyone, otherwise the
        owner plus the linked memory's shared_with. Surprise and birthday
        honorees are always excluded."""
        if event.get("visibility", "shared") != "shared":
            if member_id != event.get("owner", ""):
                linked = self._linked_event_memory(event)
                shared_with = linked.get("shared_with", []) if linked else []
                if member_id not in shared_with:
                    return False
        if event.get("kind") in ("birthday", "surprise_party") and (
            event.get("kind") == "surprise_party"
            or event.get("visibility", "shared") != "shared"
        ):
            if member_id in self._event_honorees(event):
                return False
        return True

    def upcoming_events(
        self, member_id: str, ref_date: str = "", limit: int = 10
    ) -> list:
        self.require_member(member_id)
        ref = _check_date(ref_date) or datetime.date.today().isoformat()
        try:
            limit = max(1, min(int(limit), 50))
        except (TypeError, ValueError):
            limit = 10
        upcoming = [
            dict(e)
            for e in self.data["events"]
            if e["date"] >= ref
            and self.authorized_for_event(member_id, e)
            and member_id in _recipients_for_event(self, e)[0]
        ]
        upcoming.sort(key=lambda e: (e["date"], e["title"]))
        return upcoming[:limit]

    # ---- approvals ----
    def request_approval(
        self,
        *,
        run_id: str,
        requested_by: str,
        action: str,
        recipients: list | tuple = (),
        message: str = "",
        memory_ids: list | tuple = (),
        note: str = "",
    ) -> dict:
        self.require_member(requested_by)
        if action not in ACTIONS:
            raise FamilyError(
                f"action {action!r} must be one of: {', '.join(ACTIONS)}"
            )
        recipients = [str(r).strip() for r in recipients if str(r).strip()]
        if action in ("send_message", "schedule_reminder") and not recipients:
            raise FamilyError(f"{action} needs at least one recipient")
        for recipient in recipients:
            self.require_member(recipient)
        memory_ids = [str(m).strip() for m in memory_ids if str(m).strip()]
        if action == "share_memory":
            if not memory_ids:
                raise FamilyError("share_memory needs at least one memory")
            for mid in memory_ids:
                memory = next(
                    (m for m in self.data["memories"] if m["id"] == mid), None
                )
                if memory is None:
                    raise FamilyError(f"unknown memory {mid!r}")
                if memory["owner"] != requested_by:
                    raise PermissionDeniedError(
                        f"only {memory['owner']} can share memory {mid}"
                    )
        approval = {
            "id": _new_id("ap"),
            "run_id": run_id,
            "requested_by": requested_by,
            "action": action,
            "recipients": recipients,
            "message": (message or "").strip(),
            "memory_ids": memory_ids,
            "note": (note or "").strip(),
            "status": "pending",
            "decided_by": "",
            "decision_note": "",
            "granted": [],
            "reminder_id": "",
            "delivered": [],
            "created": now(),
        }
        self.data["approvals"].append(approval)
        self._save("request_approval", requested_by, {"id": approval["id"]})
        return dict(approval)

    def pending_approvals(self, run_id: str = "") -> list:
        return [
            dict(a)
            for a in self.data["approvals"]
            if a["status"] == "pending" and (not run_id or a["run_id"] == run_id)
        ]

    def approvals_for_run(self, run_id: str) -> list:
        return [dict(a) for a in self.data["approvals"] if a["run_id"] == run_id]

    def _run_member(self, run_id: str) -> str:
        try:
            identity = json.loads(
                (self.root / "workspace/runs" / run_id / "family.json").read_text()
            )
        except (OSError, ValueError):
            return ""
        if not isinstance(identity, dict):
            return ""
        if identity.get("family_id") != self.family_id:
            return ""
        return str(identity.get("member_id") or "")

    def decide_approval(
        self, approval_id: str, *, decided_by: str, decision: str, note: str = ""
    ) -> dict:
        self.require_member(decided_by)
        if decision not in ("approved", "rejected"):
            raise FamilyError("decision must be approved or rejected")
        approval = next(
            (a for a in self.data["approvals"] if a["id"] == approval_id), None
        )
        if approval is None:
            raise FamilyError(f"unknown approval {approval_id!r}")
        owner = self._run_member(approval.get("run_id", "")) or approval.get(
            "requested_by", ""
        )
        if owner != decided_by:
            raise FamilyError(
                f"{decided_by} cannot decide approval {approval_id!r} owned by {owner!r}"
            )
        if approval["status"] != "pending":
            raise FamilyError(f"approval {approval_id} is already decided")
        approval.update(
            status=decision, decided_by=decided_by, decision_note=(note or "").strip()
        )
        granted: list = []
        if decision == "approved" and approval["action"] == "share_memory":
            for mid in approval["memory_ids"]:
                memory = next(
                    (m for m in self.data["memories"] if m["id"] == mid), None
                )
                if memory is None:
                    continue
                current = list(memory.get("shared_with") or [])
                for recipient in approval["recipients"]:
                    if recipient != memory["owner"] and recipient not in current:
                        current.append(recipient)
                memory["shared_with"] = current
                granted.append({"memory": mid, "members": list(current)})
            approval["granted"] = granted
        if approval.get("reminder_id"):
            reminder = next(
                (
                    r
                    for r in self.data.get("reminders", [])
                    if r["id"] == approval["reminder_id"]
                ),
                None,
            )
            if reminder is not None:
                reminder["status"] = decision
                if decision == "approved" and approval["action"] == "schedule_reminder":
                    from .invitations import deliver_approved_reminder

                    approval["delivered"] = [
                        m["id"] for m in deliver_approved_reminder(self, reminder)
                    ]
        self._save(decision, decided_by, {"id": approval_id, "granted": granted})
        return dict(approval)

    # ---- reminders ----
    def propose_reminder(
        self,
        *,
        run_id: str,
        requested_by: str,
        key: str,
        title: str,
        date: str,
        recipients: list | tuple = (),
        message: str = "",
        memory_ids: list | tuple = (),
    ) -> dict:
        """Propose a proactive reminder: a reminder record plus its approval.

        The reminder stays proposed until a person approves or rejects the
        linked approval. Each key is proposed at most once, so reminders
        never nag.
        """
        self.require_member(requested_by)
        key = (key or "").strip()
        if not key:
            raise FamilyError("a reminder needs a key")
        if any(r["key"] == key for r in self.data.get("reminders", [])):
            raise FamilyError("this reminder was already proposed")
        title = (title or "").strip()
        if not title:
            raise FamilyError("a reminder needs a title")
        date = _check_date(date)
        if not date:
            raise FamilyError("a reminder needs a date")
        recipients = [str(r).strip() for r in recipients if str(r).strip()]
        if not recipients:
            raise FamilyError("a reminder needs at least one recipient")
        for recipient in recipients:
            self.require_member(recipient)
        reminder = {
            "id": _new_id("rm"),
            "key": key,
            "title": title,
            "date": date,
            "recipients": recipients,
            "message": (message or "").strip(),
            "status": "proposed",
            "requested_by": requested_by,
            "run_id": run_id,
            "approval_id": "",
            "created": now(),
        }
        self.data.setdefault("reminders", []).append(reminder)
        approval = self.request_approval(
            run_id=run_id,
            requested_by=requested_by,
            action="schedule_reminder",
            recipients=recipients,
            message=reminder["message"],
            memory_ids=list(memory_ids),
            note=f"reminder {key}: {title} on {date}",
        )
        reminder["approval_id"] = approval["id"]
        stored = next(a for a in self.data["approvals"] if a["id"] == approval["id"])
        stored["reminder_id"] = reminder["id"]
        approval["reminder_id"] = reminder["id"]
        self._save("propose_reminder", requested_by, {"id": reminder["id"], "key": key})
        return {"reminder": dict(reminder), "approval": approval}

    def reminders(self) -> list:
        return [dict(r) for r in self.data.get("reminders", [])]

    def save_reminder_image(self, reminder_id: str, filename: str) -> None:
        reminder = next(
            (r for r in self.data.get("reminders", []) if r["id"] == reminder_id),
            None,
        )
        if reminder is None:
            raise FamilyError(f"unknown reminder {reminder_id!r}")
        reminder["image"] = filename
        self._save("attach_image", "agent", {"id": reminder_id})

    # ---- direct messages ----
    def send_direct(
        self,
        *,
        sender: str,
        recipient: str,
        text: str,
        kind: str = "text",
        card: dict | None = None,
    ) -> dict:
        """A member's own words to another member (WhatsApp-style chat).

        kind="card" sends an invitation card; the card names its event.
        """
        self.require_member(sender)
        self.require_member(recipient)
        text = (text or "").strip()
        if not text:
            raise FamilyError("message text is empty")
        if kind not in ("text", "card"):
            raise FamilyError("kind must be text or card")
        if kind == "card" and not (card or {}).get("event_title"):
            raise FamilyError("a card invitation needs an event_title")
        message = {
            "id": _new_id("dm"),
            "sender": sender,
            "recipient": recipient,
            "text": text,
            "kind": kind,
            "card": dict(card or {}),
            "created": now(),
        }
        self.data.setdefault("direct", []).append(message)
        self._save("send_direct", sender, {"id": message["id"], "to": recipient})
        return dict(message)

    def thread(self, member_id: str, other_id: str) -> list:
        """Every direct message exchanged between two members, oldest first."""
        self.require_member(member_id)
        self.require_member(other_id)
        return [
            dict(m)
            for m in self.data.get("direct", [])
            if {m["sender"], m["recipient"]} == {member_id, other_id}
        ]
