"""Invitation cards for approved reminders, with Muse Image art when available.

Approving a schedule_reminder approval delivers one invitation card per
recipient into their People thread with the proposer. Card art is generated
through the Meta Model API when reachable; otherwise cards render from CSS.
"""

from __future__ import annotations

import base64
import time
import uuid
from pathlib import Path

from agent_kit import progress
from agent_kit.settings import image_model


def generate_invitation_image(
    client, title: str, kind: str, *, root: Path | None = None, run_id: str = ""
) -> bytes | None:
    """Festive invitation art, or None when the image model is unavailable."""
    step_entry = (
        progress.start_step(root, run_id, task="generate_image", model=image_model(),
                            summary="Making an invitation card…")
        if root is not None else None
    )
    step_no = step_entry["step"] if step_entry else 0
    t0 = time.perf_counter()
    try:
        response = client.images.generate(
            model=image_model(),
            prompt=(
                f"Warm family {kind} invitation illustration, "
                f"festive, no words, no text: {title}"
            ),
            size="1024x1024",
        )
        if root is not None:
            usage = getattr(response, "usage", None)
            progress.finish_step(
                root, run_id, step_no,
                duration_ms=(time.perf_counter() - t0) * 1000,
                summary=f"invitation art for {title}",
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
            )
        data = (response.data or [None])[0]
        if data is None:
            return None
        encoded = getattr(data, "b64_json", None)
        if encoded:
            return base64.b64decode(encoded)
        url = getattr(data, "url", None)
        if url:
            import httpx

            fetched = httpx.get(url, timeout=30)
            if fetched.status_code == 200:
                return fetched.content
    except Exception as error:
        if root is not None:
            progress.finish_step(
                root, run_id, step_no,
                duration_ms=(time.perf_counter() - t0) * 1000,
                status="error",
                summary=f"invitation art unavailable: {error}",
            )
        return None
    return None


def _extension(raw: bytes) -> str:
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "webp"
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if raw[:3] == b"\xff\xd8\xff":
        return "jpg"
    return "png"


def attach_generated_image(store, client, reminder: dict, run_id: str = "") -> str:
    """Generate art for a proposed reminder and attach it; "" when unavailable."""
    event = None
    key = reminder.get("key", "")
    if key.startswith("event:"):
        event = next(
            (e for e in store.data["events"] if e["id"] == key[len("event:") :]),
            None,
        )
    kind = (event.get("kind") if event else None) or (
        "public_holiday" if key.startswith("holiday:") else "celebration"
    )
    image = generate_invitation_image(
        client, reminder.get("title", ""), kind, root=store.root, run_id=run_id
    )
    if not image:
        return ""
    folder = store.path.parent / store.family_id / "images"
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex}.{_extension(image)}"
    (folder / name).write_bytes(image)
    store.save_reminder_image(reminder["id"], name)
    return name


def build_card(store, reminder: dict, recipient: str) -> dict:
    """A personalized invitation card for one recipient."""
    names = {m["id"]: m["name"] for m in store.members()}
    key = reminder.get("key", "")
    event = None
    if key.startswith("event:"):
        event = next(
            (e for e in store.data["events"] if e["id"] == key[len("event:") :]),
            None,
        )
    if event is not None:
        linked = next(
            (
                m
                for m in store.data["memories"]
                if m["id"] == (event.get("memory_id") or "")
            ),
            None,
        )
        people = linked["people"] if linked else []
        kind = event.get("kind") or "celebration"
    else:
        people = []
        kind = "public_holiday" if key.startswith("holiday:") else "celebration"
    title = reminder.get("title", "")
    kind_messages = {
        "birthday": f"Join us to celebrate {title}!",
        "anniversary": f"Come celebrate {title} with the family!",
        "surprise_party": "Psst — a secret! Help us make it special.",
        "public_holiday": f"Let's get together for {title} on {reminder.get('date', '')}!",
    }
    image = reminder.get("image") or ""
    return {
        "type": "invitation",
        "event_title": title,
        "event_date": reminder.get("date", ""),
        "event_kind": kind,
        "for": ", ".join(names.get(p, p) for p in people),
        "from": names.get(reminder.get("requested_by", ""), reminder.get("requested_by", "")),
        "to": names.get(recipient, recipient),
        "message": kind_messages.get(kind, f"You're invited: {title}!"),
        "image": f"/images/{store.family_id}/{image}" if image else "",
    }


def deliver_approved_reminder(store, reminder: dict) -> list:
    """Send one invitation card per recipient; returns the sent messages."""
    sent = []
    for recipient in reminder.get("recipients", []):
        card = build_card(store, reminder, recipient)
        sent.append(
            store.send_direct(
                sender=reminder["requested_by"],
                recipient=recipient,
                text=card["message"],
                kind="card",
                card=card,
            )
        )
    return sent
