"""Local demo console for the Hai Family Agent (standard library only).

    ./helper fconsole [--port 8766]   # then open http://127.0.0.1:8766

Chat with the shared agent as any member, then review approvals as another.
Live model calls need MODEL_API_KEY in .env; without it the API reports an
error and no run starts.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import ssl
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from agent_kit.client import (
    INTERACTIVE_MAX_RETRIES,
    INTERACTIVE_TIMEOUT_SECONDS,
    MissingKeyError,
    make_client,
)
from agent_kit.progress import load_progress, record_step
from agent_kit.settings import transcribe_model

from .loop import LAST_TURN_MESSAGE, MORE_TURNS_MESSAGE, resume_run, run_agent
from .reminders import _recipients_for_event, due_reminders
from .seed import seed_all, seed_family
from .store import CATEGORIES, FamilyError, FamilyStore

ROOT = Path(__file__).resolve().parents[1]


def api_seed(root: Path, payload: dict) -> dict:
    family = str(payload.get("family") or "").strip()
    if family:
        return seed_family(root, family)
    return seed_all(root)


def api_start(root: Path, client, payload: dict) -> dict:
    return run_agent(
        root=root,
        client=client,
        family_id=str(payload.get("family") or ""),
        member_id=str(payload.get("member") or ""),
        message=str(payload.get("message") or ""),
        turns=int(payload.get("turns") or 10),
    )


def api_message(root: Path, client, payload: dict) -> dict:
    run_id = str(payload.get("run_id") or "")
    member_id = _caller_member(root, str(payload.get("family") or ""), payload)
    _require_run_owner(
        root, run_id, str(payload.get("family") or ""), member_id
    )
    return resume_run(
        root=root,
        run_id=run_id,
        client=client,
        message=str(payload.get("message") or ""),
        turns=int(payload.get("turns") or 10),
    )


def _check_wav(audio: bytes) -> None:
    """Validate a complete WAV file (16 or 24 kHz mono 16-bit), header included.

    The transcribe endpoint takes a whole WAV file, not raw PCM.
    """
    try:
        with wave.open(io.BytesIO(audio), "rb") as w:
            channels, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
            frames = w.readframes(w.getnframes())
    except (wave.Error, EOFError, ValueError) as error:
        raise FamilyError(f"audio is not a readable WAV file: {error}")
    if channels != 1 or width != 2 or rate not in (16000, 24000):
        raise FamilyError(
            f"audio must be mono 16-bit WAV at 16 or 24 kHz "
            f"(got {channels}ch, {width * 8}-bit, {rate} Hz)"
        )
    if not frames:
        raise FamilyError("audio is empty")


def _transcribe_url() -> str:
    """Exactly https://api.meta.ai/v1/asr/transcribe by default.

    A base URL override never produces a doubled /v1.
    """
    base = (os.environ.get("MODEL_API_BASE_URL") or "").strip().rstrip("/")
    if not base:
        return "https://api.meta.ai/v1/asr/transcribe"
    if base.endswith("/v1"):
        return base + "/asr/transcribe"
    return base + "/v1/asr/transcribe"


def _ssl_context() -> "ssl.SSLContext":
    """TLS context using the same CA bundle as the Meta chat calls.

    The OpenAI SDK reaches api.meta.ai through httpx, which verifies
    against certifi's bundle (present in the project venv). Stdlib
    urllib does not pick that up on its own, so point it there too.
    Verification is never disabled.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _transcribe_wav(wav: bytes, model: str, token: str) -> dict:
    """One-shot file transcription, mirroring the cookbook's transcribe_recording."""
    crlf = bytes((13, 10))
    boundary = "----haivoice" + os.urandom(8).hex()
    bcode = boundary.encode("ascii")
    request_part = json.dumps({
        "model": model,
        "mode": "PUSH_TO_TALK",
        "audioEncoding": "WAV",
    })
    body = b"".join([
        b"--" + bcode + crlf,
        b'Content-Disposition: form-data; name="request"' + crlf,
        b"Content-Type: application/json" + crlf + crlf,
        request_part.encode("utf-8") + crlf,
        b"--" + bcode + crlf,
        b'Content-Disposition: form-data; name="audio"; filename="voice_note.wav"' + crlf,
        b"Content-Type: audio/wav" + crlf + crlf,
        wav + crlf,
        b"--" + bcode + b"--" + crlf,
    ])
    req = Request(
        _transcribe_url(),
        data=body,
        headers={
            "Authorization": "Bearer " + token,
            "Accept": "application/json",
            "Content-Type": "multipart/form-data; boundary=" + boundary,
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=90, context=_ssl_context()) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except HTTPError as error:
        try:
            detail = error.read().decode("utf-8", "replace")[:300]
        except Exception:
            detail = ""
        raise FamilyError(f"transcription failed [{error.code}]: {detail or 'no detail'}")
    except (URLError, TimeoutError, OSError, ValueError) as error:
        raise FamilyError(f"transcription failed: {error}")
    if not isinstance(result, dict):
        raise FamilyError("transcription failed: unexpected response")
    return result


def api_voice_note(root: Path, client, payload: dict) -> dict:
    """Transcribe a voice note, then run it as the member's message.

    Body: {family, member, audio_b64, mime, run_id?}. Transcribes with Muse
    Voice Transcribe, records a step for it, then starts (or resumes) the
    member's run with the transcript plus its language noted.
    """
    family_id = str(payload.get("family") or "")
    member_id = _caller_member(root, family_id, payload)
    run_id = str(payload.get("run_id") or "")
    if run_id:
        _require_run_owner(root, run_id, family_id, member_id)
    try:
        audio = base64.b64decode(str(payload.get("audio_b64") or ""), validate=True)
    except (ValueError, TypeError) as error:
        raise FamilyError(f"audio is not valid base64: {error}")
    if not audio:
        raise FamilyError("audio is empty")
    if len(audio) > 25 * 1024 * 1024:
        raise FamilyError("audio is too large (25 MB max)")
    store = FamilyStore(root, family_id)
    preferred = (store.require_member(member_id).get("preferred_language") or "English").strip() or "English"
    token = os.environ.get("MODEL_API_KEY", "")
    if not token:
        raise FamilyError("MODEL_API_KEY is not set: add it to agent/.env")
    if audio[:4] != b"RIFF":
        raise FamilyError(
            "audio must be a WAV file (16 kHz mono 16-bit) — "
            "re-record in the browser, which sends WAV directly"
        )
    _check_wav(audio)
    model = transcribe_model()
    started = time.perf_counter()
    result = _transcribe_wav(audio, model, token)
    duration_ms = (time.perf_counter() - started) * 1000
    transcript = (result.get("transcript") or "").strip()
    if not transcript:
        turns = result.get("turns") or []
        transcript = " ".join(
            str(t.get("transcript") or "") for t in turns if isinstance(t, dict)
        ).strip()
    if not transcript:
        raise FamilyError("transcription is empty")
    nl = chr(10)
    message = (
        f"{transcript}{nl}{nl}"
        f"(This arrived as a voice note transcribed from {preferred}.)"
    )
    if run_id:
        out = resume_run(
            root=root, run_id=run_id, client=client, message=message,
            turns=int(payload.get("turns") or 10),
        )
    else:
        out = run_agent(
            root=root, client=client, family_id=family_id,
            member_id=member_id, message=message,
            turns=int(payload.get("turns") or 10),
        )
    record_step(
        root, out["run_id"], task="voice_note", model=model,
        duration_ms=duration_ms, status="ok", summary="Transcribed voice note",
    )
    out = dict(out)
    out.update({"transcript": transcript, "language": preferred})
    return out


def _run_identity(root: Path, run_id: str) -> dict:
    try:
        identity = json.loads(
            (root / "workspace/runs" / run_id / "family.json").read_text()
        )
    except (OSError, ValueError):
        return {}
    return identity if isinstance(identity, dict) else {}


def _require_run_owner(root: Path, run_id: str, family_id: str, member_id: str) -> None:
    """Reject unless the run's family.json names this family and member.

    Same ownership match as api_conversations; run_id alone proves nothing.
    """
    if not member_id:
        raise FamilyError("member is required")
    identity = _run_identity(root, run_id)
    if (
        identity.get("family_id") != family_id
        or identity.get("member_id") != member_id
    ):
        raise FamilyError(f"run {run_id!r} is not owned by {member_id!r}")


def _caller_member(root: Path, family_id: str, payload: dict) -> str:
    """Return the requesting member, validated against the family."""
    member_id = str(payload.get("member") or "")
    if not member_id:
        raise FamilyError("member is required")
    FamilyStore(root, family_id).require_member(member_id)
    return member_id


def api_approvals(root: Path, payload: dict) -> dict:
    family_id = str(payload.get("family") or "")
    store = FamilyStore(root, family_id)
    member_id = _caller_member(root, family_id, payload)
    owned_runs = set()
    runs_dir = root / "workspace/runs"
    if runs_dir.is_dir():
        for folder in runs_dir.iterdir():
            identity = _run_identity(root, folder.name)
            if (
                identity.get("family_id") == family_id
                and identity.get("member_id") == member_id
            ):
                owned_runs.add(folder.name)
    approvals: list = []
    for run_id in sorted(owned_runs):
        for approval in store.pending_approvals(run_id):
            reminder_id = approval.get("reminder_id") or ""
            if reminder_id:
                reminder = next(
                    (
                        r
                        for r in store.data.get("reminders", [])
                        if r.get("id") == reminder_id
                    ),
                    None,
                )
                image = (reminder or {}).get("image") or ""
                if image:
                    approval["image"] = f"/images/{store.family_id}/{image}"
            approvals.append(approval)
    return {"family_id": store.family_id, "approvals": approvals}


def api_decide(root: Path, payload: dict) -> dict:
    store = FamilyStore(root, str(payload.get("family") or ""))
    approval = store.decide_approval(
        str(payload.get("approval_id") or ""),
        decided_by=str(payload.get("by") or ""),
        decision=str(payload.get("decision") or ""),
        note=str(payload.get("note") or ""),
    )
    return {"approval_id": approval["id"], "decision": approval["status"]}


def api_run(root: Path, payload: dict) -> dict:
    run_id = str(payload.get("run_id") or "")
    family_id = str(payload.get("family") or "")
    member_id = _caller_member(root, family_id, payload)
    _require_run_owner(root, run_id, family_id, member_id)
    progress = load_progress(root, run_id)
    identity = json.loads(
        (root / "workspace/runs" / run_id / "family.json").read_text()
    )
    outputs = {
        m.get("call_id"): m.get("output")
        for m in progress["messages"]
        if m.get("type") == "function_call_output"
    }
    calls = [
        {"name": m.get("name"), "result": outputs.get(m.get("call_id"))}
        for m in progress["messages"]
        if m.get("type") == "function_call"
    ]
    return {
        "run_id": run_id,
        "family_id": identity["family_id"],
        "member_id": identity["member_id"],
        "status": progress["status"],
        "turn": progress["turn"],
        "model": progress.get("model", ""),
        "steps": progress.get("steps", []),
        "calls": calls,
    }


def api_run_steps(root: Path, payload: dict) -> dict:
    """Live steps for one run, polled by the chat UI while a turn runs."""
    run_id = str(payload.get("run_id") or "")
    family_id = str(payload.get("family") or "")
    member_id = _caller_member(root, family_id, payload)
    _require_run_owner(root, run_id, family_id, member_id)
    progress = load_progress(root, run_id)
    return {
        "run_id": run_id,
        "status": progress.get("status"),
        "steps": progress.get("steps", []),
    }


_LEAK_STATE: dict = {"state": "idle", "done": 0, "total": 0, "report": None, "error": ""}


def api_leak_test_start(root: Path, payload: dict) -> dict:
    """Start the live leak test in a background thread (it takes minutes)."""
    if _LEAK_STATE["state"] == "running":
        return {"state": "running", "done": _LEAK_STATE["done"], "total": _LEAK_STATE["total"]}
    from .leak_test import run_leak_test

    _LEAK_STATE.update(state="running", done=0, total=40, report=None, error="")

    def work() -> None:
        try:
            client = make_client(
                timeout_seconds=INTERACTIVE_TIMEOUT_SECONDS,
                max_retries=INTERACTIVE_MAX_RETRIES,
            )
            report = run_leak_test(
                root, client,
                progress_cb=lambda d, t: _LEAK_STATE.update(done=d, total=t),
            )
            _LEAK_STATE.update(state="done", report=report)
        except Exception as error:
            _LEAK_STATE.update(state="error", error=f"{type(error).__name__}: {error}")

    threading.Thread(target=work, daemon=True).start()
    return {"state": "running", "done": 0, "total": _LEAK_STATE["total"]}


def api_leak_test_status(root: Path, payload: dict) -> dict:
    return {
        "state": _LEAK_STATE["state"],
        "done": _LEAK_STATE["done"],
        "total": _LEAK_STATE["total"],
        "error": _LEAK_STATE["error"],
        "report": _LEAK_STATE["report"],
    }


def api_families(root: Path, payload: dict) -> dict:
    folder = root / "workspace/families"
    ids = sorted(p.stem for p in folder.glob("*.json")) if folder.is_dir() else []
    return {"families": ids}


def api_members(root: Path, payload: dict) -> dict:
    store = FamilyStore(root, str(payload.get("family") or ""))
    return {"family_id": store.family_id, "members": store.members()}


def api_memories(root: Path, payload: dict) -> dict:
    store = FamilyStore(root, str(payload.get("family") or ""))
    member_id = str(payload.get("member") or "")
    category = str(payload.get("category") or "")
    if category and category not in CATEGORIES:
        raise FamilyError(f"category {category!r} must be private, shared, or restricted")
    memories = [
        m for m in store.visible_memories(member_id) if not category or m["category"] == category
    ]
    return {
        "family_id": store.family_id,
        "member_id": member_id,
        "memories": [
            {
                "id": m["id"],
                "category": m["category"],
                "owner": m["owner"],
                "text": m["text"],
                "people": m["people"],
                "shared_with": m.get("shared_with") or [],
                "event_date": m["event_date"],
            }
            for m in memories
        ],
    }


def api_dm_send(root: Path, payload: dict) -> dict:
    store = FamilyStore(root, str(payload.get("family") or ""))
    return {
        "sent": store.send_direct(
            sender=str(payload.get("sender") or ""),
            recipient=str(payload.get("recipient") or ""),
            text=str(payload.get("text") or ""),
        )
    }


def api_dm_thread(root: Path, payload: dict) -> dict:
    store = FamilyStore(root, str(payload.get("family") or ""))
    return {
        "messages": store.thread(
            str(payload.get("member") or ""), str(payload.get("other") or "")
        )
    }


def api_due_reminders(root: Path, payload: dict) -> dict:
    store = FamilyStore(root, str(payload.get("family") or ""))
    return due_reminders(
        store,
        str(payload.get("member") or ""),
        ref_date=str(payload.get("ref_date") or ""),
        days_ahead=payload.get("days_ahead", 30),
    )


def api_family_events(root: Path, payload: dict) -> dict:
    """Upcoming events the requesting member may see, soonest first."""
    family_id = str(payload.get("family") or "")
    store = FamilyStore(root, family_id)
    member_id = _caller_member(root, family_id, payload)
    try:
        limit = max(1, min(int(payload.get("limit") or 30), 50))
    except (TypeError, ValueError):
        limit = 30
    items = []
    for event in store.upcoming_events(
        member_id, ref_date=str(payload.get("from_date") or ""), limit=limit
    ):
        recipients, reason = _recipients_for_event(store, event)
        items.append(
            {
                "id": event["id"],
                "title": event["title"],
                "date": event["date"],
                "kind": event.get("kind") or "",
                "recipients": recipients,
                "reason": reason,
            }
        )
    return {"family_id": store.family_id, "events": items}


def api_propose_reminder(root: Path, payload: dict) -> dict:
    store = FamilyStore(root, str(payload.get("family") or ""))
    proposed = store.propose_reminder(
        run_id="console",
        requested_by=str(payload.get("by") or ""),
        key=str(payload.get("key") or ""),
        title=str(payload.get("title") or ""),
        date=str(payload.get("date") or ""),
        recipients=list(payload.get("recipients") or []),
        message=str(payload.get("message") or ""),
    )
    return {
        "reminder_id": proposed["reminder"]["id"],
        "approval_id": proposed["approval"]["id"],
    }


def api_conversations(root: Path, payload: dict) -> dict:
    family_id = str(payload.get("family") or "")
    member_id = str(payload.get("member") or "")
    runs_dir = root / "workspace/runs"
    conversations = []
    if runs_dir.is_dir():
        for folder in runs_dir.iterdir():
            ident = folder / "family.json"
            if not ident.is_file():
                continue
            try:
                identity = json.loads(ident.read_text())
                if (
                    identity.get("family_id") != family_id
                    or identity.get("member_id") != member_id
                ):
                    continue
                progress = load_progress(root, folder.name)
            except (OSError, ValueError):
                continue
            conversations.append(
                {
                    "run_id": folder.name,
                    "status": progress.get("status"),
                    "turn": progress.get("turn", 0),
                    "started": progress.get("started", ""),
                }
            )
    conversations.sort(key=lambda c: c["started"], reverse=True)
    return {
        "family_id": family_id,
        "member_id": member_id,
        "conversations": conversations,
    }


def _history_turns(messages: list) -> list:
    turns = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        if item.get("role") == "user" and isinstance(item.get("content"), str):
            text = re.sub(r"^\[Talking to [^\]]+\]\n\n", "", item["content"]).strip()
            if not text or text in (LAST_TURN_MESSAGE, MORE_TURNS_MESSAGE):
                continue
            if text.startswith(
                ("A family member approved", "A family member rejected")
            ):
                turns.append({"from": "system", "text": text})
            else:
                turns.append({"from": "member", "text": text})
        elif item.get("type") == "message":
            parts = [
                b.get("text", "")
                for b in (item.get("content") or [])
                if isinstance(b, dict)
                and b.get("type") == "output_text"
                and b.get("text")
            ]
            text = "\n".join(parts).strip()
            if text:
                turns.append({"from": "agent", "text": text})
        elif item.get("type") == "function_call":
            turns.append({"from": "system", "text": f"called {item.get('name')}"})
    return turns


def api_chat_history(root: Path, payload: dict) -> dict:
    run_id = str(payload.get("run_id") or "")
    family_id = str(payload.get("family") or "")
    member_id = _caller_member(root, family_id, payload)
    _require_run_owner(root, run_id, family_id, member_id)
    progress = load_progress(root, run_id)
    return {
        "run_id": run_id,
        "status": progress.get("status"),
        "turn": progress.get("turn", 0),
        "model": progress.get("model", ""),
        "steps": progress.get("steps", []),
        "turns": _history_turns(progress.get("messages", [])),
    }


def api_add_member(root: Path, payload: dict) -> dict:
    store = FamilyStore(root, str(payload.get("family") or ""))
    name = str(payload.get("name") or "").strip()
    if not name:
        raise FamilyError("name is empty")
    member_id = str(payload.get("id") or "").strip() or re.sub(
        r"[^a-z0-9]+", "_", name.lower()
    ).strip("_")
    member = store.add_member(
        member_id,
        name,
        email=str(payload.get("email", "") or ""),
        phone=str(payload.get("phone", "") or ""),
        role=str(payload.get("role", "") or "member"),
        preferred_language=str(payload.get("preferred_language", "") or "English"),
    )
    partner = str(payload.get("partner", "") or "")
    parents = [p for p in (payload.get("parents") or []) if str(p).strip()]
    if partner or parents:
        member = store.set_relations(member_id, partner=partner, parents=parents)
    return {"family_id": store.family_id, "member": dict(member)}


NEEDS_CLIENT = {
    "start": api_start,
    "message": api_message,
    "voice_note": api_voice_note,
}
NO_CLIENT = {
    "seed": api_seed,
    "approvals": api_approvals,
    "decide": api_decide,
    "run": api_run,
    "families": api_families,
    "members": api_members,
    "memories": api_memories,
    "dm_send": api_dm_send,
    "dm_thread": api_dm_thread,
    "due_reminders": api_due_reminders,
    "family_events": api_family_events,
    "propose_reminder": api_propose_reminder,
    "conversations": api_conversations,
    "chat_history": api_chat_history,
    "run_steps": api_run_steps,
    "leak_test_start": api_leak_test_start,
    "leak_test_status": api_leak_test_status,
    "add_member": api_add_member,
}


def _member_may_view_image(
    store: FamilyStore, family_id: str, member_id: str, filename: str
) -> bool:
    """True when member sent or receives a message/invitation using this image.

    The image is authorized only through stored records: a direct card
    message whose card image points at the file (sender or recipient may
    view it), or a reminder whose attached image is the file (the member
    who requested it or one of its recipients may view it). Filenames that
    no record references are denied.
    """
    expected = f"/images/{family_id}/{filename}"
    for message in store.data.get("direct", []):
        card = message.get("card") or {}
        image = str(card.get("image") or "").split("?", 1)[0]
        if image and (image == expected or image == filename):
            if member_id in (message.get("sender"), message.get("recipient")):
                return True
    for reminder in store.data.get("reminders", []):
        if reminder.get("image") == filename:
            if member_id == reminder.get("requested_by") or member_id in (
                reminder.get("recipients") or []
            ):
                return True
    return False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # keep the log to errors
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path.startswith("/images/"):
            parts = parsed.path[len("/images/") :].split("/")
            if len(parts) == 2 and all(
                re.fullmatch(r"[A-Za-z0-9_.-]+", p) for p in parts
            ):
                query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
                member_id = (query.get("member") or "").strip()
                query_family = (query.get("family") or "").strip()
                authorized = False
                if member_id and query_family and query_family == parts[0]:
                    try:
                        image_store = FamilyStore(ROOT, parts[0])
                        image_store.require_member(member_id)
                        authorized = _member_may_view_image(
                            image_store, parts[0], member_id, parts[1]
                        )
                    except FamilyError:
                        authorized = False
                if not authorized:
                    self._send(403, {"error": "forbidden"})
                    return
                path = ROOT / "workspace/families" / parts[0] / "images" / parts[1]
                media = {
                    ".png": "image/png",
                    ".webp": "image/webp",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                }
                if path.suffix.lower() in media and path.is_file():
                    body = path.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", media[path.suffix.lower()])
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
            self._send(404, {"error": "not found"})
            return
        if parsed.path.startswith("/api/"):
            name = parsed.path[len("/api/") :]
            if name in NO_CLIENT and name in (
                "approvals",
                "run",
                "families",
                "members",
                "memories",
                "dm_thread",
                "due_reminders",
                "family_events",
                "conversations",
                "chat_history",
                "run_steps",
                "leak_test_status",
            ):
                try:
                    query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
                    self._send(200, NO_CLIENT[name](ROOT, query))
                except (FamilyError, ValueError, OSError) as error:
                    self._send(400, {"error": str(error)})
                return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        name = parsed.path[len("/api/") :] if parsed.path.startswith("/api/") else ""
        if name not in NEEDS_CLIENT and name not in NO_CLIENT:
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("body must be a JSON object")
            if name in NEEDS_CLIENT:
                result = NEEDS_CLIENT[name](
                    ROOT,
                    make_client(
                        timeout_seconds=INTERACTIVE_TIMEOUT_SECONDS,
                        max_retries=INTERACTIVE_MAX_RETRIES,
                    ),
                    payload,
                )
            else:
                result = NO_CLIENT[name](ROOT, payload)
            self._send(200, result)
        except MissingKeyError as error:
            self._send(400, {"error": str(error)})
        except (FamilyError, ValueError, OSError) as error:
            self._send(400, {"error": str(error)})
        except Exception as error:  # never leak a traceback to the browser
            self._send(500, {"error": f"{type(error).__name__}: {error}"})


HTML = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hai Family</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{--sage:#5E7F64;--sage-dark:#48614D;--sage-soft:#E3EBE0;
--tan:#B08D5C;--tan-dark:#8A6F3F;--tan-soft:#F1E7D3;
--plum:#8A7DB5;--plum-dark:#6F6394;--plum-soft:#ECE8F5;
--beige:#E5D5B8;--beige-dark:#B08D5C;--beige-soft:#F4EDDD;
--bg:#F7F3EC;--surface:#FDFBF7;--card:#FDFBF7;--ink:#2A2622;--muted:#7A7168;
--line:#E8E0D3;--amber:var(--tan-dark);--rose:var(--tan);--rose-soft:var(--tan-soft);
--teal:var(--sage);--teal-dark:var(--sage-dark);--coral:var(--tan);--coral-dark:var(--tan-dark);
--lavender:var(--plum);--lavender-dark:var(--plum-dark);--lavender-soft:var(--plum-soft)}
*{box-sizing:border-box}
[hidden]{display:none !important}
body{margin:0;background:var(--bg);color:var(--ink);
font-family:Inter,system-ui,-apple-system,"Segoe UI",Helvetica,sans-serif}
h1,h2,.dash-greeting,.invite-title{font-family:Fraunces,Georgia,serif}
header{background:var(--teal-dark);color:#fff;padding:1em 1em .9em;
border-bottom:6px solid var(--coral)}
header .wrap{max-width:920px;margin:0 auto}
header h1{margin:0;font-size:1.4em}
header p{margin:.2em 0 0;color:#cde8e1;font-size:.9em}
main{max-width:920px;margin:1.2em auto;padding:0 1em 6em}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;
padding:1.1em 1.2em;margin-bottom:1.2em;box-shadow:0 1px 3px rgba(20,40,50,.08)}
.card h2{margin:0 0 .6em;font-size:1.1em;display:flex;align-items:center;gap:.5em}
.step{display:inline-flex;align-items:center;justify-content:center;width:1.7em;height:1.7em;
border-radius:50%;background:var(--coral);color:#fff;font-size:.85em;font-weight:700}
.row{display:flex;gap:.5em;flex-wrap:wrap;align-items:center}
.row + .row{margin-top:.5em}
label{font-weight:600}
input,textarea,select{font:inherit;color:var(--ink);border:1px solid var(--line);
border-radius:8px;padding:.45em .6em;background:#fff}
input:focus,textarea:focus,select:focus{outline:2px solid var(--teal);border-color:var(--teal)}
#msg{width:100%}
#convoLog{flex:1}
button{font:inherit;font-weight:600;border-radius:999px;padding:.45em 1.1em;cursor:pointer;
border:2px solid var(--teal);background:var(--teal);color:#fff}
button:hover{background:var(--teal-dark);border-color:var(--teal-dark)}
button.ghost{background:#fff;color:var(--teal)}
button.ghost:hover{background:#e6f4f0}
button.warn{background:#fff;color:var(--coral-dark);border-color:var(--coral)}
button.warn:hover{background:#ffe8d6}
#convoLog{display:flex;flex-direction:column;gap:.5em;margin-top:.7em;
min-height:0;overflow-y:auto}
.bubble{max-width:70%;padding:.55em .8em;border-radius:14px;white-space:pre-wrap}
.bubble.me{align-self:flex-end;background:var(--sage-soft);border-bottom-right-radius:4px}
.bubble.agent{align-self:flex-start;background:#d9f0e9;border-bottom-left-radius:4px}
.bubble.them{align-self:flex-start;background:var(--beige-soft);border:1px solid var(--line);
border-bottom-left-radius:4px}
.bubble.sys{align-self:center;background:#fff3d6;font-size:.85em;color:#7a5b00;
border:1px dashed var(--amber);max-width:100%}
.bubble .meta{font-size:.72em;opacity:.75;margin-bottom:.2em}
.invite{border-radius:14px;padding:1em 1.1em;color:#fff;background:var(--teal);
box-shadow:0 2px 8px rgba(20,40,50,.15)}
.invite-tag{font-size:.75em;font-weight:700;opacity:.85}
.invite-title{font-size:1.25em;font-weight:800;margin:.2em 0}
.invite-date{display:inline-block;background:rgba(255,255,255,.25);border-radius:999px;
padding:.15em .7em;font-weight:700}
.invite-for{margin-top:.4em;font-size:.9em;opacity:.9}
.invite-msg{margin-top:.5em;white-space:pre-wrap}
.invite-from{margin-top:.4em;font-style:italic;opacity:.9}
.invite.kind-birthday{background:var(--coral)}
.invite.kind-anniversary{background:var(--rose)}
.invite.kind-surprise_party{background:#946f00}
.invite-img{max-width:100%;width:auto;height:auto;max-height:55vh;object-fit:contain;border-radius:10px;margin-top:.6em;display:block;cursor:zoom-in}
.imgoverlay{position:fixed;inset:0;z-index:60;background:rgba(20,18,16,.85);display:flex;align-items:center;justify-content:center;padding:1.5em;cursor:zoom-out}
.imgoverlay img{max-width:96vw;max-height:92vh;object-fit:contain;border-radius:8px;cursor:default}
.imgclose{position:absolute;top:.6em;right:.8em;background:transparent;border:1px solid #fff;color:#fff;border-radius:50%;width:2em;height:2em;min-height:0;font-size:1.2rem;line-height:1;cursor:pointer}
.pill{display:inline-block;font-size:.75em;font-weight:700;border-radius:999px;
padding:.15em .7em;margin-bottom:.3em}
.pill.awaiting_member{background:#fff3d6;color:#7a5b00}
.pill.needs_input{background:#ffd9d9;color:#a61e1e}
.pill.paused{background:#e2e8f0;color:#45525e}
.pill.error{background:#a61e1e;color:#fff}
.pill.due{background:#ffd9d9;color:#a61e1e}
.pill.upcoming{background:#e2e8f0;color:#45525e}
pre.dump{background:#102a26;color:#d7efe9;border-radius:10px;padding:.7em;
white-space:pre-wrap;font-size:.82em;max-height:200px;overflow:auto}
.approval{border:1px solid var(--line);border-radius:10px;padding:.6em .8em;
margin-top:.6em;background:#fffdf6}
.approval b.id{color:var(--coral-dark)}
.approval .row{margin-top:.4em}
.hint{color:var(--muted);font-size:.85em}
#bottomnav{position:fixed;bottom:0;left:0;right:0;display:flex;background:#fff;
border-top:2px solid var(--line);z-index:10}
.navtab{flex:1;border:none;border-radius:0;background:#fff;color:var(--muted);
padding:.85em;font-size:1em}
.navtab.active{background:var(--teal-dark);color:#fff}
.navtab:hover{background:#e6f4f0}
.navtab.active:hover{background:var(--teal-dark)}
.subtabs{display:flex;gap:.5em;margin-bottom:1em}
.subtab{flex:1;border-radius:12px}
.subtab.active{background:var(--coral);border-color:var(--coral)}
.subtab.active:hover{background:var(--coral-dark);border-color:var(--coral-dark)}
.subtab:not(.active){background:#fff;color:var(--coral-dark);border-color:var(--coral)}
.split{display:flex;gap:.8em;margin-top:.6em}
.side{width:210px;flex:none;display:flex;flex-direction:column;gap:.35em;
max-height:360px;overflow:auto}
.rosterbtn{text-align:left;background:#fff;color:var(--ink);border:2px solid var(--line);
border-radius:10px}
.rosterbtn:hover{border-color:var(--teal);background:#fff}
.rosterbtn.active{border-color:var(--coral);background:#fff4ec}
.thread{flex:1;display:flex;flex-direction:column;min-width:0}
.detail{flex:1;min-width:0;border:1px solid var(--line);border-radius:10px;
padding:.7em .9em;background:#f7fafb}
.tiles{display:flex;gap:.6em;margin:.6em 0}
.tile{flex:1;border-radius:12px;padding:.7em .4em;text-align:center;cursor:pointer;
color:#fff;font-weight:700}
.tile small{display:block;font-weight:400;opacity:.9}
.tile.shared{background:var(--teal)}
.tile.private{background:#b7791f}
.tile.restricted{background:var(--rose)}
.tile.active{box-shadow:0 0 0 3px var(--ink)}
.mem{border-top:1px solid var(--line);padding:.5em 0}
.mem .meta{font-size:.78em;color:var(--muted)}
.topbar{display:flex;justify-content:space-between;align-items:center;gap:.6em}
.iconbtn{border-radius:50%;padding:.45em .55em;line-height:0}
main.consumer{max-width:560px}
main.admin{max-width:920px}
#treeWrap{position:relative}
#treeSvg{position:absolute;inset:0;pointer-events:none}
.trow{display:flex;justify-content:center;gap:1.2em;padding:1em 0;position:relative;z-index:1}
.tnode{text-align:center;cursor:pointer;min-width:76px}
.tnode .avatar{width:64px;height:64px;border-radius:50%;margin:0 auto;color:#fff;
display:flex;align-items:center;justify-content:center;font-weight:800;font-size:1.3em}
.tnode.you .avatar{box-shadow:0 0 0 3px #fff,0 0 0 5px var(--sage)}
.tname{font-weight:700;margin-top:.25em}
.trole{font-size:.78em;color:var(--muted)}
.youdot{display:inline-block;width:.8em;height:.8em;border-radius:50%;background:var(--sage);margin-right:.3em}
#fab{position:fixed;right:1.2em;bottom:4.5em;width:3.2em;height:3.2em;border-radius:50%;
font-size:1.4em;padding:0;z-index:20;box-shadow:0 3px 10px rgba(20,40,50,.25)}
.dashrow{display:flex;gap:.5em;align-items:center;border-top:1px solid var(--line);padding:.45em 0;flex-wrap:wrap}
.dashrow:first-child{border-top:none}
.profAvatar{width:84px;height:84px;border-radius:50%;color:#fff;display:flex;align-items:center;
justify-content:center;font-weight:800;font-size:1.8em;margin-bottom:.4em}
.cal-grid{display:grid;grid-template-columns:repeat(7,1fr);gap:2px;margin-top:.6em}
.cal-dow{font-size:.75em;color:var(--muted);text-align:center}
.day{background:#fff;border:1px solid transparent;border-radius:10px;padding:.3em .1em;
font-weight:600;cursor:pointer;color:var(--ink);font-size:.9em}
.day:hover{border-color:var(--teal);background:#fff}
.day.today{border-color:var(--coral)}
.day.selected{background:var(--teal);color:#fff;border-color:var(--teal)}
.day .dot{display:block;width:6px;height:6px;border-radius:50%;background:var(--coral);margin:1px auto 0}
.day.selected .dot{background:#fff}
/* Consolidated design system: warm ivory, sage primary, lavender/beige accents. */
body{background:var(--bg);line-height:1.45}
header{background:var(--surface);color:var(--ink);border-bottom:1px solid var(--line);
box-shadow:0 1px 8px rgba(46,46,42,.05)}
header h1{color:var(--ink);font-size:1.4rem}
header p{color:var(--muted)}
main{padding-top:1.1rem}
button{background:var(--sage);border-color:var(--sage);transition:background .18s,border-color .18s,transform .18s,box-shadow .18s}
button:hover{background:var(--sage-dark);border-color:var(--sage-dark);transform:translateY(-1px)}
button.ghost{background:var(--surface);color:var(--sage-dark);border-color:var(--sage-soft)}
button.ghost:hover{background:var(--sage-soft)}
button.warn{background:var(--surface);color:var(--rose);border-color:var(--rose)}
button.warn:hover{background:var(--rose-soft)}
button:focus-visible,input:focus-visible,select:focus-visible,textarea:focus-visible{outline:2px solid var(--lavender);outline-offset:1px}
button:disabled{opacity:.55;cursor:default;transform:none}
input,textarea,select{background:var(--surface);transition:border-color .18s,box-shadow .18s}
.step{background:var(--sage)}
.card{background:var(--card);border-radius:20px;border:1px solid var(--line);box-shadow:0 4px 18px rgba(46,46,42,.06);padding:1.15rem 1.25rem;margin-bottom:1.1rem}
.card h2{font-size:1.25rem}
.subtab{transition:background .18s,color .18s,border-color .18s}
.subtab.active{background:var(--sage);border-color:var(--sage)}
.subtab.active:hover{background:var(--sage-dark);border-color:var(--sage-dark)}
.subtab:not(.active){background:var(--surface);color:var(--sage-dark);border-color:var(--sage-soft)}
#bottomnav{background:var(--surface);border-top:1px solid var(--line);box-shadow:0 -4px 20px rgba(46,46,42,.06);padding:.35em .5em}
.navtab{display:flex;flex-direction:column;align-items:center;gap:2px;background:transparent;
color:var(--muted);font-size:.72rem;padding:.55em .2em;border-radius:14px;transition:background .18s,color .18s}
.navtab.active{background:var(--sage-soft);color:var(--sage-dark);font-weight:700}
.navtab:hover{background:var(--sage-soft);color:var(--sage-dark)}
.navtab.active:hover{background:var(--sage-soft)}
#fab{background:var(--sage);border-color:var(--sage);bottom:5.4em;box-shadow:0 6px 18px rgba(72,97,77,.3)}
#fab:hover{background:var(--sage-dark)}
.bubble{border-radius:18px}
.bubble.me{background:var(--sage-soft)}
.bubble.agent{background:var(--surface);border:1px solid var(--line)}
.bubble.them{background:var(--beige-soft)}
.bubble.sys{background:#FBF0D3;color:#7a5b00;border-color:var(--amber)}
.invite.kind-birthday{background:var(--beige-dark)}
.invite.kind-anniversary{background:var(--lavender-dark)}
.invite.kind-surprise_party{background:var(--amber)}
.rosterbtn{transition:border-color .18s,background .18s}
.rosterbtn:hover{border-color:var(--sage)}
.rosterbtn.active{border-color:var(--sage);background:var(--sage-soft)}
.pill.due{background:var(--rose-soft);color:var(--rose)}
.pill.upcoming{background:var(--lavender-soft);color:var(--ink)}
.pill.awaiting_member{background:#FBF0D3;color:var(--amber)}
.approval{background:var(--beige-soft);border-radius:14px}
.tile.shared{background:var(--sage)}
.tile.private{background:var(--beige-dark)}
.tile.restricted{background:var(--lavender-dark)}
.tnode .avatar{width:84px;height:84px;font-size:1.5em;box-shadow:0 4px 14px rgba(46,46,42,.14)}
.tnode.you .avatar{box-shadow:0 0 0 3px var(--surface),0 0 0 5px var(--sage)}
.youdot{background:var(--sage)}
.trow{gap:1.6em;padding:2em .4em;flex-wrap:wrap}
.tnode{min-width:92px;transition:transform .18s}
.tnode:hover{transform:translateY(-2px)}
.tname{font-size:1.12em;margin-top:.45em}
.trole{font-size:.9em;text-transform:capitalize;margin-top:.15em}
#treeWrap{padding:.6em 0 .4em;overflow-x:auto}
.whatidid{text-align:left;max-width:100%}
.step-row{padding:.35em 0;border-top:1px solid var(--line);font-size:.88em}
.step-row:first-of-type{border-top:none}
.timeline{border:1px solid var(--line);border-radius:12px;padding:.7em .9em;margin-bottom:.7em;background:#fff}
.timeline .step-row{padding:.45em 0}
.pill.ok{background:#e3ebe0;color:#48614d}
.pill.fallback{background:#fbf0d3;color:#9a6b12}
.spin{display:inline-block;width:.8em;height:.8em;border:2px solid var(--sage-soft);
border-top-color:var(--sage-dark);border-radius:50%;animation:spin .8s linear infinite;vertical-align:-.1em}
@keyframes spin{to{transform:rotate(360deg)}}
@media (prefers-reduced-motion:reduce){.spin{animation:none}}
.whatidid details summary{cursor:pointer;font-weight:700}
.whatidid details .step-row:first-of-type{margin-top:.4em}
.card{transition:opacity .3s}
#main.switching .card{opacity:0}
.signin-card h2{font-size:1.5em}
body.signed-out header,body.signed-out #rail,body.signed-out #bottomnav,body.signed-out #acctPanel,body.signed-out #acctScrim{display:none!important}
body.signed-out main.consumer,body.signed-out main.admin{max-width:1100px;margin:0 auto;padding:0 24px}
body.signed-out #foot{display:none}
.signin-wrap{display:grid;grid-template-columns:1fr 1fr;gap:3rem;align-items:center;min-height:calc(100vh - 120px);max-width:1100px;margin:0 auto;padding:3rem 0}
.signin-brand .logo{font-size:2rem}
.signin-title{font-family:Fraunces,Georgia,serif;font-size:2.6rem;line-height:1.1;margin:.4em 0 .3em}
.signin-sub{color:var(--muted);font-size:1.05rem;margin:0 0 1.2em}
.signin-points{list-style:none;margin:0 0 1.4em;padding:0;display:grid;gap:.55em}
.signin-points li{display:flex;align-items:center;gap:.6em;font-weight:600}
.signin-points svg{flex:none;color:var(--sage-dark)}
.signin-pill{display:inline-block;background:var(--sage-soft);color:var(--sage-dark);border-radius:999px;padding:.35em .9em;font-size:.82rem;font-weight:600}
.signin-suggest{background:var(--sage-soft);border:1px solid var(--sage);border-radius:14px;padding:.8em 1em;margin-bottom:1em}
.signin-suggest b{display:block;margin-bottom:.15em}
.signin-suggest p{margin:.2em 0 .7em;font-size:.9rem}
.signin-row{display:grid;grid-template-columns:repeat(4,1fr);gap:.8em;margin:.4em 0 1.2em}
.signin-group-label{font-size:.78rem;font-weight:700;color:var(--muted);margin:1em 0 .5em}
.signin-tile{display:flex;flex-direction:column;align-items:center;gap:.35em;background:var(--card);
border:1px solid var(--line);border-radius:20px;padding:1em .4em;min-height:44px;cursor:pointer;color:var(--ink)}
.signin-tile b{color:#2A2622;font-weight:700}
.signin-tile .signin-role{color:#7A7168;font-size:.82rem;text-transform:capitalize}
.signin-tile:hover{border:2px solid #5E7F64;transform:translateY(-2px);box-shadow:0 6px 18px rgba(46,46,42,.12);background:var(--card);color:var(--ink)}
.signin-tile:hover b{color:#2A2622}
.signin-avatar{display:inline-flex;align-items:center;justify-content:center;width:64px;height:64px;
border-radius:50%;color:#fff;font-weight:800;font-size:1.2rem}
.signin-foot{text-align:center;color:var(--muted);font-size:.85rem;padding:0 0 2.5em}
@media(max-width:900px){.signin-wrap{grid-template-columns:1fr;gap:1.5rem;min-height:auto;padding:2rem 0}
.signin-title{font-size:2rem}.signin-row{grid-template-columns:repeat(4,1fr)}}
@media(max-width:560px){.signin-row{grid-template-columns:repeat(2,1fr)}}
#acctScrim{position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(42,38,34,.22);z-index:40}
#acctPanel{position:fixed;top:0;right:0;bottom:0;width:400px;max-width:92vw;background:var(--card);
border-left:1px solid var(--line);z-index:41;padding:1.2rem;display:flex;flex-direction:column;gap:.6em;
box-shadow:-8px 0 24px rgba(42,38,34,.12);animation:acctin .25s ease}
@keyframes acctin{from{transform:translateX(48px);opacity:0}to{transform:none;opacity:1}}
@media (prefers-reduced-motion:reduce){#acctPanel{animation:none}}
#acctRole,.rolecap{text-transform:capitalize}
#acctPanel h3{margin:.8em 0 .2em;font-size:.95rem}
.acct-head{display:flex;align-items:center;gap:.8em}
#acctAvatar{width:56px;height:56px;border-radius:50%;color:#fff;font-weight:800;font-size:1.3rem;
display:inline-flex;align-items:center;justify-content:center;box-shadow:0 0 0 3px var(--card),0 0 0 5px var(--sage)}
.acctlink{background:transparent;border:none;color:var(--ink);font-weight:600;text-align:left;
padding:.7em .4em;min-height:44px}
.acctlink:hover{background:var(--sage-soft)}
.devchip{display:inline-block;background:#E4E2EA;color:#6F6394;border-radius:999px;
font-size:.72rem;font-weight:700;padding:.15em .6em}
.underhood{background:transparent;border:1px dashed var(--line);color:var(--muted);border-radius:12px;
padding:.5em .6em;font-size:.85rem;margin-bottom:.6em}
.underhood:hover{border-color:var(--sage);color:var(--sage-dark)}
#switchDlg{position:fixed;top:0;left:0;right:0;bottom:0;z-index:50;display:flex;
align-items:center;justify-content:center;background:rgba(42,38,34,.35)}

.switch-box{background:var(--card);border:1px solid var(--line);border-radius:20px;
padding:1.4rem;max-width:560px;width:92vw;box-shadow:0 12px 40px rgba(42,38,34,.2)}
#switchTiles{display:grid;grid-template-columns:repeat(4,1fr);gap:.8em;margin:.8em 0}
#toast{position:fixed;bottom:2em;left:50%;transform:translateX(-50%);background:var(--ink);
color:#fff;border-radius:999px;padding:.6em 1.2em;z-index:60;box-shadow:0 6px 20px rgba(42,38,34,.3)}

.linkbtn{background:none;border:none;color:var(--sage-dark);font-weight:600;
text-decoration:underline;padding:.5em 0;min-height:44px}
#railMe .avatarbtn,#railMe{cursor:pointer}
.day{transition:background .18s,border-color .18s}
.day:hover{border-color:var(--sage)}
.day.selected{background:var(--sage);border-color:var(--sage)}
.day.today{border-color:var(--lavender);box-shadow:inset 0 0 0 1px var(--lavender)}
#rail{display:none}
.brand{display:flex;align-items:baseline;gap:.5em}
.logo{font-family:Fraunces,Georgia,serif;font-weight:700;font-size:1.5rem;color:var(--sage-dark)}
.famname{font-weight:600;color:var(--ink)}
.famlabel{font-size:.78rem;color:var(--muted);margin-top:1em;display:block}
#rail select{width:100%;margin:.3em 0 1em}
.railnav{display:flex;flex-direction:column;gap:.4em}
.railnav .navtab{flex-direction:row;justify-content:flex-start;gap:.7em;padding:.7em .9em;font-size:.95rem;background:transparent}
#railMe{margin-top:auto;display:flex;align-items:center;gap:.6em;background:transparent;border:none;color:var(--ink);font-weight:600;padding:.6em .4em}
#railAvatar,#headAvatar{display:inline-flex;align-items:center;justify-content:center;color:#fff;font-weight:800}
#railAvatar{width:40px;height:40px;border-radius:50%;font-size:.95rem;box-shadow:0 0 0 2px var(--surface),0 0 0 4px var(--sage)}
#headAvatar{width:30px;height:30px;border-radius:50%;font-size:.75rem}
.avatarbtn{padding:.3em}
#foot{text-align:center;color:var(--muted);font-size:.8rem;padding:1.2em 1em 7em}
button,.navtab,.subtab,.chip,.day,.rosterbtn{min-height:44px}
.chip{font-size:.9rem}
@media(min-width:1024px){
#bottomnav{display:none}
header{display:none}
#rail{display:flex;position:fixed;top:0;left:0;bottom:0;width:220px;flex-direction:column;background:var(--card);border-right:1px solid var(--line);padding:1.2rem 1rem;z-index:20}
main.consumer,main.admin{--mw:720px;max-width:var(--mw);
margin:1.2em max(2em,(100% - 220px - var(--mw))/2) 0 calc(220px + max(2em,(100% - 220px - var(--mw))/2));padding:0 2em 4em}
main.consumer.wide{--mw:960px}
#fab{bottom:2em}
}
@media(max-width:1023px){
#rail{display:none}
#bottomnav{display:flex}
main.consumer,main.admin{max-width:720px;margin:0 auto;padding:0 16px 7em}
header .wrap{max-width:720px}
.topbar{display:flex;justify-content:space-between;align-items:center;gap:.6em}
}
.day .dot{background:var(--sage-dark)}
.pre,.dump{background:#2E2E2A}
pre.dump{background:#2E2E2A}
.detail{background:var(--beige-soft)}
.dash-hero{background:linear-gradient(135deg,var(--surface),var(--sage-soft));border-color:var(--sage-soft)}
.dash-greeting{font-size:1.6rem;font-weight:800;margin:0 0 .3em;line-height:1.2}
.demo-label{display:inline-block;background:var(--lavender-soft);color:var(--ink);border-radius:999px;padding:.2em .7em;font-size:.78rem;font-weight:600}
.avatar-row{display:flex;flex-wrap:wrap;gap:.7em;margin-top:.9em}
.mini-member{display:flex;flex-direction:column;align-items:center;gap:.25em;background:none;border:none;color:var(--ink);padding:.2em;cursor:pointer;font-size:.78rem;max-width:4.6rem}
.mini-member:hover{background:var(--surface);transform:none}
.mini-avatar{width:44px;height:44px;border-radius:50%;color:#fff;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:.95rem;box-shadow:0 2px 8px rgba(46,46,42,.16)}
.mini-member.you .mini-avatar{box-shadow:0 0 0 2px var(--surface),0 0 0 4px var(--sage)}
.mini-role{color:var(--muted);font-size:.7rem;text-transform:capitalize}
.insight{background:var(--lavender-soft);border:1px solid var(--lavender);border-radius:16px;padding:.9em 1em}
.insight b{color:var(--sage-dark)}
.empty{color:var(--muted);background:var(--beige-soft);border:1px dashed var(--line);border-radius:14px;padding:.8em 1em}
.dashrow{transition:background .18s;border-radius:10px;padding:.55em .4em}
.dashrow:hover{background:var(--beige-soft)}
.quick-actions{display:grid;grid-template-columns:repeat(3,1fr);gap:.6em}
.quick-actions button{display:flex;flex-direction:column;align-items:center;gap:.35em;padding:.9em .5em;border-radius:16px;font-size:.9rem}
/* Stage 3: chat, calendar, profile surfaces. */
.chat-head{display:flex;align-items:center;gap:.7em;padding:.7em .8em;background:var(--sage-soft);border:1px solid var(--line);border-radius:16px;margin-bottom:.8em}
.chat-avatar{width:48px;height:48px;flex:none;border-radius:50%;color:#fff;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:1.05rem;box-shadow:0 2px 8px rgba(46,46,42,.16)}
.chat-head-name{font-weight:800;font-size:1.05rem}
.bubble.me{border-bottom-right-radius:4px}
.bubble.agent{border-left:4px solid var(--sage);border-bottom-left-radius:4px}
.bubble.them{border-bottom-left-radius:4px}
.bubble .who{font-size:.72rem;font-weight:700;opacity:.7;margin-bottom:.2em}
.bubble.chat-loading{color:var(--muted);font-style:italic}
.bubble.chat-error{background:var(--rose-soft);border:1px solid var(--rose);color:var(--rose)}
.chips{display:flex;flex-wrap:wrap;gap:.45em;margin:.7em 0}

.chip{background:transparent;color:var(--sage-dark);border:1px solid var(--sage);font-size:.82rem;padding:.35em .8em}
.chip:hover{background:var(--sage-soft);border-color:var(--sage-dark)}
.composer{position:sticky;bottom:0;display:flex;gap:.5em;align-items:flex-end;background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:.6em;box-shadow:0 4px 14px rgba(46,46,42,.08);margin-top:.6em}
.composer textarea{flex:1;border:none;resize:none;min-height:2.6em;max-height:9em;overflow-y:auto}
.micbtn{border-radius:50%;width:2.6em;height:2.6em;padding:0;display:inline-flex;align-items:center;justify-content:center;flex:none}
.recbar{display:flex;align-items:center;gap:.5em;background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:.4em .7em;margin-bottom:.5em;font-size:.85rem}

.recdot{width:.7em;height:.7em;border-radius:50%;background:#c0392b;flex:none;animation:recpulse 1.2s ease-in-out infinite}
@keyframes recpulse{0%,100%{opacity:1}50%{opacity:.35}}
@media (prefers-reduced-motion:reduce){.recdot{animation:none}}
.recx{background:transparent;border:1px solid var(--line);color:var(--muted);border-radius:50%;width:1.8em;height:1.8em;padding:0;line-height:1;min-height:0}
.bubble.voice audio{width:100%;min-width:12em;margin-bottom:.3em}
.vtranscript{white-space:pre-wrap}
.verror{display:flex;align-items:center;gap:.5em}
.verror button{flex:none}
.inbox{display:flex;gap:0;max-width:1200px;height:calc(100vh - 12em);min-height:26em;padding:0;overflow:hidden}
main.consumer:has(> .view:not([hidden]):not(#view-signin)){--mw:1200px;max-width:1200px}
.convolist{width:300px;flex:none;border-right:1px solid var(--line);overflow-y:auto;min-height:0;padding:.5em}
.convo{display:flex;gap:.6em;align-items:flex-start;width:100%;text-align:left;background:transparent;border:none;border-radius:12px;padding:.6em;min-height:44px;color:var(--ink)}
.convo:hover{background:var(--beige-soft)}
.convo.active{background:var(--sage-soft)}
.convo-avatar{width:40px;height:40px;flex:none;border-radius:50%;color:#fff;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:.9rem}
.convo-avatar.hai-logo{background:var(--sage-dark);font-family:Fraunces,Georgia,serif;font-size:.8rem}
.convo-main{flex:1;min-width:0}
.convo-name{display:block;font-weight:700;font-size:.92rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.convo-preview{display:block;font-size:.8rem;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.convo-time{font-size:.72rem;color:var(--muted);flex:none;align-self:flex-start}
.hist-label{font-size:.8rem;font-weight:700;color:var(--muted)}
.hist-select:focus{outline:2px solid var(--sage);border-color:var(--sage)}
.appr{border:1px solid var(--line);border-radius:14px;padding:.8em 1em;margin-top:.6em;background:var(--surface);max-width:86%;align-self:flex-start}
.appr-msg{white-space:pre-wrap}
.appr-to{font-size:.82rem;color:var(--muted);margin-top:.4em}
.appr-actions{display:flex;gap:.5em;flex-wrap:wrap;margin-top:.6em}
.appr-status{font-size:.85rem;color:var(--sage-dark);font-weight:700;margin-top:.6em}
.runid{display:flex;align-items:center;gap:.5em;margin-top:.4em;font-family:ui-monospace,Menlo,monospace;font-size:.72rem;color:var(--muted)}
.copybtn{background:transparent;border:1px solid var(--line);color:var(--muted);border-radius:8px;padding:.15em .6em;font-size:.72rem;min-height:0}
.copybtn:hover{border-color:var(--sage);color:var(--sage-dark);background:transparent}
.convo-pane{flex:1;min-width:0;min-height:0;display:flex;flex-direction:column;padding:.8em}
.convo-log{display:flex;flex-direction:column;gap:.5em;margin-top:.7em;overflow-y:auto;min-height:0;flex:1}
.daysep{align-self:center;font-size:.75rem;color:var(--muted);background:var(--beige-soft);border-radius:999px;padding:.2em .8em;margin:.4em 0}
.bubble .stime{display:block;font-size:.7rem;opacity:.7;margin-top:.25em;text-align:right}
.hist-select{max-width:11em}
@media (max-width:640px){.inbox{flex-direction:column}.convolist{width:auto;border-right:none;border-bottom:1px solid var(--line);max-height:12em}}
.composer textarea:focus{outline:none}
.kind-badge{display:inline-block;font-size:.72rem;font-weight:700;border-radius:999px;padding:.12em .6em;background:var(--beige-soft);color:var(--ink);text-transform:capitalize}
.kind-badge.birthday{background:var(--beige);color:#6b5320}
.kind-badge.anniversary{background:var(--lavender-soft);color:#4d3f75}
.kind-badge.event,.kind-badge.holiday{background:var(--sage-soft);color:var(--sage-dark)}
.kind-badge.surprise_party{background:#FBF0D3;color:var(--amber)}
.dots{display:flex;gap:2px;justify-content:center;margin-top:1px;min-height:6px}
.dot{width:6px;height:6px;border-radius:50%;background:var(--sage-dark)}
.dot.birthday{background:var(--beige-dark)}
.dot.anniversary{background:var(--lavender-dark)}
.dot.surprise_party{background:var(--amber)}
.day.selected .dot{background:#fff}
.cal-legend{display:flex;flex-wrap:wrap;gap:.8em;margin-top:.6em;font-size:.78rem;color:var(--muted)}
.cal-legend .dot{display:inline-block;margin-right:.25em}
.event-card{border:1px solid var(--line);border-radius:16px;padding:.8em 1em;margin-top:.6em;background:var(--surface);transition:background .18s}
.event-card:hover{background:var(--beige-soft)}
.event-card .row{margin-top:.5em}
.privacy-badge{display:inline-block;font-size:.72rem;font-weight:700;border-radius:999px;padding:.12em .6em;text-transform:capitalize}
.privacy-badge.shared{background:var(--sage-soft);color:var(--sage-dark)}
.privacy-badge.private{background:var(--beige);color:#6b5320}
.privacy-badge.restricted{background:var(--lavender-soft);color:#4d3f75}
.pvwrap{position:relative;display:inline-block}
.pvchip{display:inline-flex;align-items:center;gap:.35em;font-size:.72rem;font-weight:700;border-radius:999px;padding:.2em .7em;border:1px solid;cursor:pointer;min-height:0}
.pvchip svg{flex:none}
.pv-shared{color:var(--sage-dark);border-color:var(--sage);background:var(--sage-soft)}
.pv-private{color:#6b5320;border-color:var(--tan);background:var(--tan-soft)}
.pv-restricted{color:var(--plum-dark);border-color:var(--plum);background:var(--plum-soft)}
.pvpop{position:absolute;z-index:30;top:calc(100% + 6px);left:0;min-width:12em;background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:.6em .7em;box-shadow:0 8px 24px rgba(46,46,42,.16)}
.pvrow-title{font-size:.72rem;font-weight:700;color:var(--muted);margin:.4em 0 .2em}
.pvrow-title:first-child{margin-top:0}
.pvperson{display:flex;align-items:center;gap:.5em;padding:.15em 0;font-size:.85rem}
.pvavatar{width:24px;height:24px;flex:none;border-radius:50%;color:#fff;display:inline-flex;align-items:center;justify-content:center;font-weight:800;font-size:.65rem}
.prof-section{border-top:1px solid var(--line);margin-top:1em;padding-top:.8em}
.tnode{padding:.4em;border-radius:16px;outline-offset:2px}
.tnode:hover,.tnode:focus-visible{background:var(--beige-soft)}
/* Stage 4: reminder / approval / memory clarity. */
.pill.pending{background:#FBF0D3;color:var(--amber)}
.pill.proposed{background:var(--lavender-soft);color:var(--ink)}
.pill.approved{background:var(--sage-soft);color:var(--sage-dark)}
.pill.rejected{background:var(--rose-soft);color:var(--rose)}
.count-badge{display:inline-block;font-size:.72rem;font-weight:700;border-radius:999px;padding:.12em .6em;background:var(--sage-soft);color:var(--sage-dark)}
.count-badge.today{background:var(--sage);color:#fff}
.count-badge.overdue{background:var(--rose-soft);color:var(--rose)}
.reminder-card{border:1px solid var(--line);border-radius:16px;padding:.8em 1em;margin-top:.6em;background:var(--surface);transition:background .18s}
.reminder-card:hover{background:var(--beige-soft)}
.reminder-card .row{margin-top:.5em}
.approval-card{border:1px solid var(--line);border-radius:16px;padding:.8em 1em;margin-top:.6em;background:var(--beige-soft)}
.approval-preview{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:.6em .8em;margin:.5em 0;white-space:pre-wrap}
.approval-actions{display:flex;gap:.5em;flex-wrap:wrap;align-items:center;margin-top:.6em}
.mem-visible{font-size:.8rem;color:var(--muted);margin-top:.25em}
.mem-visible b{color:var(--ink)}
.tile .tile-access{display:block;font-weight:400;font-size:.72rem;opacity:.92;margin-top:.15em}
@media (max-width:640px){.split{flex-direction:column}.side{width:auto;max-height:none}
.quick-actions{grid-template-columns:1fr}.card{padding:1rem}.dash-greeting{font-size:1.35rem}
.day{min-height:36px}.trow{gap:1em;padding:1.2em .2em}}
@media (min-width:641px) and (max-width:900px){.quick-actions{grid-template-columns:repeat(3,1fr)}}
</style></head><body>
<header><div class="wrap topbar">
<div class="brand"><span class="logo">hai</span><span class="famname">The Riveras</span></div>
<div class="row hact">
<button class="iconbtn" onclick="showView('dashboard')" aria-label="notifications"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 8a6 6 0 0 0-12 0c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.7 21a2 2 0 0 1-3.4 0"/></svg></button>
<button class="iconbtn avatarbtn" onclick="showView('profile')" aria-label="profile"><span id="headAvatar">?</span></button>
</div></div></header>
<aside id="rail">
<div class="brand"><span class="logo">hai</span><span class="famname">The Riveras</span></div>
<nav class="railnav">
<button class="navtab" data-view="dashboard" onclick="showView('dashboard')"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></svg>Home</button>
<button class="navtab" data-view="chat" onclick="showView('chat')"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 11.5a8.5 8.5 0 0 1-8.5 8.5c-1.5 0-3-.4-4.2-1.1L3 20l1.1-5.3A8.5 8.5 0 1 1 21 11.5z"/></svg>Chat</button>
<button class="navtab" data-view="calendar" onclick="showView('calendar')"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="5" width="18" height="16" rx="2"/><path d="M16 3v4M8 3v4M3 11h18"/></svg>Calendar</button>
<button class="navtab" data-view="tree" onclick="showView('tree')"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="5" r="2.5"/><circle cx="5" cy="19" r="2.5"/><circle cx="19" cy="19" r="2.5"/><path d="M12 8v3m0 0H5v5m7-5h7v5"/></svg>Family</button>
</nav>
<button class="underhood" onclick="showView('settings')">Under the hood <span class="devchip">Developer</span></button>
<button id="railMe" onclick="toggleAcct(true)"><span id="railAvatar">?</span><span id="railName">…</span></button>
</aside>
<div id="acctScrim" hidden onclick="toggleAcct(false)"></div>
<aside id="acctPanel" hidden>
<div class="acct-head"><span id="acctAvatar">?</span><div><b id="acctName">…</b><br><span class="hint" id="acctRole">…</span></div></div>
<h3>Your account</h3>
<button class="acctlink" onclick="acctGo('profile')">Profile</button>
<button class="acctlink" onclick="acctGo('dashboard')">Notifications</button>
<h3>Demo &amp; developer tools</h3>
<button class="acctlink" onclick="openSwitch()">Demo: switch person <span class="devchip">Developer</span></button>
<button class="acctlink" onclick="acctGo('settings')">Agent console &amp; settings <span class="devchip">Developer</span></button>
<button class="warn" onclick="signOut()">Sign out</button>
</aside>
<div id="switchDlg" hidden><div class="switch-box">
<h2>Switch person</h2>
<div id="switchTiles"></div>
<div class="row"><button class="ghost" onclick="closeSwitch()">Cancel</button></div>
</div></div>
<div id="toast" hidden></div>
<main id="main" class="consumer">
<section id="view-signin" class="view" hidden>
<div class="signin-wrap">
<div class="signin-brand">
<div class="brand"><span class="logo">hai</span></div>
<h2 class="signin-title">One companion for the whole family.</h2>
<p class="signin-sub">hai remembers for everyone, and keeps each person&rsquo;s secrets.</p>
<ul class="signin-points">
<li><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="8" r="3.5"/><path d="M5 20c1.5-3.5 4.5-5 7-5s5.5 1.5 7 5"/></svg>Remembers for everyone</li>
<li><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="5" y="10" width="14" height="10" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/></svg>Keeps secrets per person</li>
<li><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 11.5a8.5 8.5 0 0 1-8.5 8.5c-1.5 0-3-.4-4.2-1.1L3 20l1.1-5.3A8.5 8.5 0 1 1 21 11.5z"/></svg>Asks before it acts</li>
</ul>
<span class="signin-pill">Powered by Meta Muse Wonder · Muse Voice Transcribe · Muse Image</span>
</div>
<div class="card signin-card"><h2>Who&rsquo;s using hai?</h2>
<div class="signin-suggest"><b>New here? Start as Leo</b><p>See a surprise being planned for Maya, then switch to her and watch it disappear.</p><button onclick="startAsLeo()">Start as Leo</button></div>
<div id="signinGroups"></div></div>
</div>
<p class="signin-foot">The Riveras are a fictional demo family.</p>
</section>
<section id="view-dashboard" class="view">
<div class="card dash-hero"><p class="dash-greeting"><span id="dashGreet">Hi</span>, <span id="dashName">…</span></p>
<p class="hint" id="dashFamilyLine">Here is what needs your attention. <span id="dashSecrets"></span></p>
<div class="avatar-row" id="dashAvatars"></div></div>
<div class="card"><h2>Needs you</h2><div id="dashInsight"><span class="hint">Loading…</span></div></div>
<div class="card"><h2>Upcoming birthdays &amp; anniversaries</h2><div id="dashCelebrations"><span class="hint">Loading…</span></div></div>
<div class="card"><h2>My notifications</h2><div id="dashNotes"><span class="hint">Loading…</span></div></div>
<div class="card"><h2>Needs review</h2><div id="dashApprovals"><span class="hint">Loading…</span></div></div>
<div class="card"><h2>Quick actions</h2><div class="quick-actions">
<button type="button" onclick="dashAsk()"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 11.5a8.5 8.5 0 0 1-8.5 8.5c-1.5 0-3-.4-4.2-1.1L3 20l1.1-5.3A8.5 8.5 0 1 1 21 11.5z"/></svg>Ask hai</button>
<button type="button" class="ghost" onclick="dashPlan()"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="5" width="18" height="16" rx="2"/><path d="M16 3v4M8 3v4M3 11h18"/></svg>Plan an Event</button>
<button type="button" class="ghost" onclick="dashMemory()"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>Add a Memory</button>
</div></div>
</section>
<section id="view-tree" class="view" hidden>
<div class="card"><h2>Family</h2>
<div class="hint"><span class="youdot"></span>You — tap anyone to see them.</div>
<div id="treeWrap"></div></div>
<div class="card" id="memberForm" hidden><h2>Add family member</h2>
<div class="row"><label for="fName">name</label><input id="fName" size="18">
<label for="fRole">role</label><select id="fRole"><option>father</option><option>mother</option><option>brother</option><option>sister</option><option>grandfather</option><option>grandmother</option><option>uncle</option><option>aunt</option><option>member</option></select></div>
<div class="row"><label for="fEmail">email</label><input id="fEmail" size="16">
<label for="fPhone">phone</label><input id="fPhone" size="12"></div>
<div class="row"><label for="fPartner">partner</label><select id="fPartner"></select></div>
<div class="row"><label for="fParent1">parent 1</label><select id="fParent1"></select>
<label for="fParent2">parent 2</label><select id="fParent2"></select></div>
<div class="row"><button onclick="submitMember()">Add</button><span class="hint" id="formOut"></span></div></div>
</section>
<section id="view-chat" class="view" hidden>
<div class="pane card inbox">
<div id="convoList" class="convolist" role="listbox" aria-label="Conversations"></div>
<div class="convo-pane">
<div id="convoHead" class="chat-head"></div>
<div id="convoLog"></div>
<div class="chips" id="haiChips" aria-label="suggested prompts" hidden>
<button type="button" class="chip" onclick="usePrompt('When is the next birthday?')">When is the next birthday?</button>
<button type="button" class="chip" onclick="usePrompt('What is coming up for our family soon?')">What is coming up soon?</button>
<button type="button" class="chip" onclick="usePrompt('Please help me plan a family event. Ask me what we are celebrating, when, and who is coming.')">Help plan an event</button>
<button type="button" class="chip" onclick="usePrompt('Please help me save a family memory. Ask me what happened, who was there, and when — and confirm before saving.')">Save a memory</button>
</div>
<div class="recbar" id="recBar" hidden><span class="recdot" aria-hidden="true"></span><span id="recTime">0:00</span><span class="hint">Recording… tap the mic again to send.</span><button type="button" class="recx" id="recCancel" onclick="cancelVoice()" aria-label="Cancel recording">×</button></div>
<div class="composer"><textarea id="msg" rows="1" aria-label="Type a message" placeholder="Type a message…"></textarea>
<button id="micBtn" class="micbtn" onclick="toggleMic()" aria-label="Record a voice note"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="2" width="6" height="12" rx="3"/><path d="M5 10a7 7 0 0 0 14 0M12 19v3"/></svg></button>
<button id="sendBtn" onclick="send()">Send</button></div>
<div id="chatStatus" class="hint" aria-live="polite"></div>
</div>
</div>
<div id="imgOverlay" class="imgoverlay" hidden><button class="imgclose" id="imgClose" aria-label="Close">×</button><img id="imgOverlayImg" alt=""></div>
</section>
<section id="view-calendar" class="view" hidden>
<div class="card"><h2>Calendar</h2>
<div class="row"><button class="ghost" onclick="calMove(-1)">‹</button><b id="calLabel">…</b><button class="ghost" onclick="calMove(1)">›</button></div>
<div class="cal-grid" id="calGrid"></div>
<div class="cal-legend"><span><span class="dot birthday"></span>Birthday</span><span><span class="dot anniversary"></span>Anniversary</span><span><span class="dot"></span>Event</span><span><span class="dot surprise_party"></span>Surprise</span></div></div>
<div class="card"><h2>On <span id="calDayLabel">…</span></h2><div id="calDayList"></div></div>
<div class="card"><h2>Upcoming events</h2><div id="calUpcoming"><span class="hint">Loading…</span></div></div>
</section>
<section id="view-profile" class="view" hidden>
<div class="card"><h2>Profile</h2>
<div id="profBody"></div></div>
</section>
<section id="view-settings" class="view" hidden>
<div class="card"><h2><span class="step">1</span>Seed the demo families</h2>
<div class="row"><button onclick="seed()">Seed all demo families</button>
<span class="hint">The Rivera family (8 members, including grandparents, uncle and aunt). If already seeded, it is skipped.</span></div>
<pre class="dump" id="seedOut">Not seeded yet in this session.</pre></div>
<div class="card"><h2><span class="step">2</span>Family members</h2>
<div class="row"><span class="hint">Pick a member on the side.</span></div>
<div class="split"><div id="memberList" class="side"></div>
<div id="memberDetail" class="detail"><span class="hint">Pick a member on the side.</span></div>
</div></div>
<div class="card"><h2><span class="step">3</span>Memory by category</h2>
<div class="row"><label for="memView">view as</label><select id="memView" onchange="cLoadMemories()"></select>
<span class="hint">Only what this member is authorized to see.</span></div>
<div class="tiles" id="memTiles"></div>
<div id="memList"><span class="hint">Pick a category tile to list its memories.</span></div></div>
<div class="card"><h2><span class="step">4</span>Proactive reminders</h2>
<div class="row"><label for="remDays">days ahead</label><input id="remDays" value="60" size="4" onchange="cLoadReminders()">
<span class="hint">For <b id="remAs">…</b> (from “view as”). Birthdays skip the birthday person; secrets stay in the know.</span></div>
<div id="remList"><span class="hint">Due and upcoming reminders appear here.</span></div></div>
<div class="card"><h2><span class="step">5</span>Review approvals</h2>
<div class="row"><span class="hint">Approvals load when this screen opens.</span></div>
<div id="approvals"><span class="hint">(none pending)</span></div></div>
<div class="card"><h2><span class="step">6</span>Inspect a run</h2>
<div class="row"><label for="runInspect">run</label><input id="runInspect" size="30">
<button class="ghost" onclick="inspectRun()">Inspect</button></div>
<div id="runSteps"></div>
<pre class="dump" id="runOut">No run inspected.</pre></div>
<div class="card"><h2><span class="step">7</span>Live leak test</h2>
<div class="row"><button onclick="leakRun()">Run leak test</button>
<span class="hint">Asks the real agent (Muse Wonder, verification on) 40 prompts as Maya, Sofia, Eli and Leo against a throwaway copy of the Rivera data, then keyword + Muse Spark judge checks each reply. Takes a few minutes.</span></div>
<div id="leakOut"><span class="hint">Not run yet in this session.</span></div></div>
</section>
</main>
<nav id="bottomnav">
<button class="navtab" data-view="dashboard" onclick="showView('dashboard')"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></svg>Home</button>
<button class="navtab" data-view="chat" onclick="showView('chat')"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 11.5a8.5 8.5 0 0 1-8.5 8.5c-1.5 0-3-.4-4.2-1.1L3 20l1.1-5.3A8.5 8.5 0 1 1 21 11.5z"/></svg>Chat</button>
<button class="navtab" data-view="calendar" onclick="showView('calendar')"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="5" width="18" height="16" rx="2"/><path d="M16 3v4M8 3v4M3 11h18"/></svg>Calendar</button>
<button class="navtab" data-view="tree" onclick="showView('tree')"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="5" r="2.5"/><circle cx="5" cy="19" r="2.5"/><circle cx="19" cy="19" r="2.5"/><path d="M12 8v3m0 0H5v5m7-5h7v5"/></svg>Family</button>
</nav>
<footer id="foot">The Riveras are a fictional demo family.</footer>
<button id="fab" hidden onclick="toggleMemberForm()">+</button>
<script>
const S = {peer: '', convo: 'hai', haiRun: '', haiHistory: [], memFilter: '', memCache: [], consoleLoaded: false, members: [], me: storedMe(), family: 'riveras', profileView: '', chatKey: ''};
function storedMe() {
  try { return localStorage.getItem('hai.me') || ''; }
  catch (e) { return ''; }
}
function rememberMe(id) {
  try {
    if (id) localStorage.setItem('hai.me', id);
    else localStorage.removeItem('hai.me');
  } catch (e) {}
}
const api = (name, body) => fetch('/api/' + name, {method: 'POST',
  headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body || {})})
  .then(r => r.json());
const get = (name, params) => fetch('/api/' + name + '?' + new URLSearchParams(params))
  .then(r => r.json());
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');
const el = id => document.getElementById(id);
function imgUrl(path) {
  const raw = String(path || '');
  if (raw.indexOf('/images/') === -1) return raw;
  const u = new URL(raw, window.location.origin);
  if (u.pathname.indexOf('/images/') !== 0) return raw;
  if (S.me) u.searchParams.set('member', S.me);
  else u.searchParams.delete('member');
  u.searchParams.set('family', S.family || '');
  return u.pathname + u.search + u.hash;
}
function showView(name) {
  if (name !== 'signin' && !S.me) name = 'signin';
  document.body.classList.toggle('signed-out', !S.me);
  ['signin', 'dashboard', 'tree', 'chat', 'calendar', 'profile', 'settings'].forEach(v => {
    el('view-' + v).hidden = v !== name;
  });
  document.querySelectorAll('[data-view]').forEach(b => {
    b.classList.toggle('active', b.dataset.view === name);
  });
  el('main').className = (name === 'settings' ? 'admin' : 'consumer') + (name === 'chat' ? ' wide' : '');
  el('fab').hidden = name !== 'tree';
  if (name === 'dashboard') loadDashboard();
  if (name === 'tree') loadTree();
  if (name === 'calendar') renderCalendar();
  if (name === 'profile') renderProfile(S.profileView || S.me);
  if (name === 'chat') initInbox();
  if (name === 'settings' && !S.consoleLoaded) { S.consoleLoaded = true; cLoadAll(); }
  window.scrollTo(0, 0);
}
function shortTime(iso) {
  const d = new Date(iso);
  if (isNaN(d)) return '';
  return d.toLocaleTimeString([], {hour: 'numeric', minute: '2-digit'});
}
function dayLabel(iso) {
  const d = new Date(iso);
  if (isNaN(d)) return '';
  return d.toLocaleDateString([], {weekday: 'short'}) + ' · '
    + d.toLocaleDateString([], {month: 'short', day: 'numeric'});
}
function daySepLabel(iso) {
  const d = new Date(iso);
  if (isNaN(d)) return '';
  return d.toLocaleDateString([], {weekday: 'short', month: 'short', day: 'numeric'});
}
async function initInbox() {
  if (!S.me) return;
  if (!S.members.length) {
    const r = await get('members', {family: S.family});
    if (!r.error) S.members = r.members;
    updateRailMe();
  }
  if (!S.convo) S.convo = 'hai';
  await renderConvoList();
  openConvo(S.convo, true);
}
async function renderConvoList() {
  const box = el('convoList');
  const me = S.me;
  if (!me) { box.innerHTML = ''; return; }
  box.innerHTML = '';
  const others = (S.members || []).filter(m => m.id !== me);
  const previews = {};
  await Promise.all(others.map(async m => {
    try {
      const r = await get('dm_thread', {family: S.family, member: me, other: m.id});
      const msgs = (r.messages || []).filter(x => x.kind !== 'card' || x.text || x.card);
      const last = msgs[msgs.length - 1];
      previews[m.id] = last
        ? {text: last.text || (last.card ? last.card.event_title : ''), when: last.created || ''}
        : null;
    } catch (e) { previews[m.id] = null; }
  }));
  if (me !== S.me) return;
  const haiBtn = document.createElement('button');
  haiBtn.className = 'convo' + (S.convo === 'hai' ? ' active' : '');
  haiBtn.innerHTML = '<span class="convo-avatar hai-logo">hai</span>'
    + '<div class="convo-main"><div class="convo-name">hai</div>'
    + '<div class="convo-preview">Your family companion</div></div>';
  haiBtn.dataset.id = 'hai';
  haiBtn.onclick = () => openConvo('hai');
  box.appendChild(haiBtn);
  others.forEach(m => {
    const p = previews[m.id];
    const b = document.createElement('button');
    b.dataset.id = m.id;
    b.className = 'convo' + (S.convo === m.id ? ' active' : '');
    b.innerHTML = '<span class="convo-avatar" style="background:' + avatarColor(m.id) + '">'
      + esc(initials(m.name)) + '</span>'
      + '<div class="convo-main"><div class="convo-name">' + esc(m.name) + '</div>'
      + '<div class="convo-preview">' + esc((p && p.text) || '') + '</div></div>'
      + '<span class="convo-time">' + esc((p && shortTime(p.when)) || '') + '</span>';
    b.onclick = () => openConvo(m.id);
    box.appendChild(b);
  });
}
function renderConvoHead() {
  const head = el('convoHead');
  if (S.convo === 'hai') {
    head.innerHTML = '<div class="convo-avatar hai-logo">hai</div>'
      + '<div style="flex:1;min-width:0"><div class="chat-head-name">hai</div></div>'
      + '<button type="button" class="ghost" onclick="newHaiChat()">New chat</button>'
      + '<label class="hist-label" for="haiHist">History</label>'
      + '<select id="haiHist" class="hist-select" aria-label="History"></select>';
    loadHaiHistory();
  } else {
    const m = (S.members || []).find(x => x.id === S.convo) || {id: S.convo, name: S.convo};
    head.innerHTML = '<div class="convo-avatar" style="background:' + avatarColor(m.id) + '">'
      + esc(initials(m.name || m.id)) + '</div>'
      + '<div><div class="chat-head-name">' + esc(m.name || m.id) + '</div>'
      + '<div class="hint" style="text-transform:capitalize">' + esc(roleLabel(m.role)) + '</div></div>';
  }
}
function openConvo(id, keep) {
  S.convo = id;
  S.peer = id === 'hai' ? '' : id;
  if (typeof cancelVoice === 'function') cancelVoice();
  el('micBtn').hidden = (id !== 'hai');
  renderConvoHead();
  renderConvoListSel();
  el('chatStatus').textContent = '';
  if (id === 'hai') openHai(keep);
  else loadThread();
}
function renderConvoListSel() {
  el('convoList').querySelectorAll('.convo').forEach(b => {
    b.classList.toggle('active', b.dataset.id === S.convo);
  });
}
function updateChips() {
  const n = el('convoLog').querySelectorAll('.bubble.me, .bubble.agent').length;
  el('haiChips').hidden = !(S.convo === 'hai' && n === 0);
}
function fillSelect(sel, members, keep) {
  sel.innerHTML = '';
  members.forEach(m => {
    const o = document.createElement('option');
    o.value = m.id; o.textContent = m.name + ' (' + m.id + ')';
    sel.appendChild(o);
  });
  if (keep && members.some(m => m.id === keep)) sel.value = keep;
}
async function loadPeople() {
  const r = await get('members', {family: S.family});
  if (r.error) return;
  S.members = r.members;
  updateRailMe();
  if (!el('view-chat').hidden) renderConvoList();
}
async function loadThread() {
  let other = S.convo !== 'hai' ? S.convo : (S.peer || '');
  const me = S.me;
  if (!other || !me) return;
  S.convo = other;
  S.peer = other;
  renderConvoHead();
  renderConvoListSel();
  const r = await get('dm_thread', {family: S.family, member: me, other: other});
  if (me !== S.me) return;
  const log = el('convoLog');
  log.innerHTML = '';
  if (r.error) {
    const d = document.createElement('div');
    d.className = 'bubble sys';
    d.textContent = r.error;
    log.appendChild(d);
    return;
  }
  const msgs = r.messages || [];
  let lastDay = '', lastBubble = null, lastSender = null;
  const closeGroup = () => {
    if (lastBubble && lastSender) {
      const t = document.createElement('span');
      t.className = 'stime';
      t.textContent = shortTime(lastSender.when);
      lastBubble.appendChild(t);
    }
    lastBubble = null; lastSender = null;
  };
  msgs.forEach(m => {
    if (m.kind === 'card' && m.card) {
      closeGroup();
      lastDay = m.created ? daySepLabel(m.created) : lastDay;
      const c = document.createElement('div');
      c.className = 'invite kind-' + (m.card.event_kind || 'celebration');
      c.innerHTML = '<div class="invite-tag">Invitation</div><div class="invite-title">'
        + esc(m.card.event_title) + '</div><span class="invite-date">'
        + esc(fmtDate(m.card.event_date)) + '</span>'
        + (m.card.image ? '<img class="invite-img" src="' + esc(imgUrl(m.card.image)) + '">' : '')
        + (m.card.for ? '<div class="invite-for">for ' + esc(m.card.for) + '</div>' : '')
        + '<div class="invite-msg">' + esc(m.card.message) + '</div>'
        + '<div class="invite-from">— ' + esc(m.card.from) + '</div>';
      log.appendChild(c);
      return;
    }
    const day = m.created ? daySepLabel(m.created) : '';
    if (day && day !== lastDay) {
      closeGroup();
      lastDay = day;
      const s = document.createElement('div');
      s.className = 'daysep';
      s.textContent = day;
      log.appendChild(s);
    }
    const mine = m.sender === me;
    if (!lastBubble || lastSender.mine !== mine) {
      closeGroup();
      const d = document.createElement('div');
      d.className = 'bubble ' + (mine ? 'me' : 'them');
      d.textContent = m.text;
      log.appendChild(d);
      lastBubble = d;
      lastSender = {mine: mine, when: m.created};
    } else {
      lastBubble.firstChild.textContent += '\\n' + m.text;
      lastSender.when = m.created;
    }
  });
  closeGroup();
  log.scrollTop = log.scrollHeight;
  updateChips();
}
async function dmSend() {
  const box = el('msg');
  const text = box.value;
  const other = S.convo;
  if (!other || other === 'hai' || !text.trim()) return;
  const r = await api('dm_send', {family: S.family,
    sender: S.me, recipient: other, text: text});
  if (r.error) {
    const log = el('convoLog');
    const d = document.createElement('div');
    d.className = 'bubble sys';
    d.textContent = r.error;
    log.appendChild(d);
    return;
  }
  box.value = '';
  autogrow();
  loadThread();
  renderConvoList();
}
function bubble(cls, html) {
  const d = document.createElement('div');
  d.className = 'bubble ' + cls;
  d.innerHTML = html;
  const log = el('convoLog');
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
  updateChips();
  return d;
}
const FRIENDLY_STEPS = {
  get_family_events: 'Checked the family calendar',
  get_due_reminders: 'Looked at upcoming reminders',
  retrieve_family_memories: 'Searched family memories',
  save_family_memory: 'Saved a memory',
  draft_family_message: 'Drafted a message',
  request_action_approval: 'Asked for your OK',
  share_agent_family: 'Checked the family list',
  verify_claims: 'Double-checked the answer',
  verify_checklist: 'Double-checked the checklist',
  generate_image: 'Designed invitation',
  transcribe: 'Transcribed the document',
  voice_note: 'Transcribed voice note',
  analyze_document: 'Read the document',
  check_signatures: 'Checked the signature',
};
function friendlyStep(s) {
  if (s.status === 'running') return 'Working…';
  const m = /^called (\\w+)/.exec(s.summary || '');
  if (m && FRIENDLY_STEPS[m[1]]) return FRIENDLY_STEPS[m[1]];
  if (FRIENDLY_STEPS[s.task]) return FRIENDLY_STEPS[s.task];
  if (s.task === 'chat' || (s.summary || '').includes('replied')) return 'Wrote a reply';
  return s.task || 'Worked on it';
}
function stepIcon(s) {
  if (s.status === 'running') return '<span class="spin"></span>';
  return s.status === 'error' ? '✗' : '✓';
}
function stepRowHtml(s) {
  const dur = s.status === 'running' ? '' : ' · ' + (s.duration_ms / 1000).toFixed(1) + 's';
  const model = s.model_label || s.model || '';
  const tag = model ? ' · ' + esc(model) : '';
  return '<div class="step-row">' + stepIcon(s) + ' ' + esc(friendlyStep(s)) + tag + dur + '</div>';
}
function renderLiveSteps(box, steps) {
  box.innerHTML = steps.length
    ? steps.map(stepRowHtml).join('')
    : '<div class="step-row"><span class="spin"></span> Working…</div>';
}
function mdInline(s) {
  return s.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
    .replace(/(^|\\W)\\*([^\\*\\n]+)\\*/g, '$1<em>$2</em>')
    .replace(/(^|\\W)_([^_\\n]+)_/g, '$1<em>$2</em>');
}
function md(text) {
  const lines = esc(text).trim().split('\\n');
  let html = '', inList = false, pendingBreak = false, justClosed = false;
  lines.forEach(line => {
    if (!line.trim()) {
      if (inList) { html += '</ul>'; inList = false; justClosed = true; }
      pendingBreak = true;
      return;
    }
    const m = /^\\s*[-*]\\s+(.*)$/.exec(line);
    if (m) {
      pendingBreak = false; justClosed = false;
      if (!inList) { html += '<ul>'; inList = true; }
      html += '<li>' + mdInline(m[1]) + '</li>';
      return;
    }
    if (inList) { html += '</ul>'; inList = false; justClosed = true; }
    if (pendingBreak && !justClosed) html += '<br>';
    pendingBreak = false; justClosed = false;
    html += mdInline(line) + '<br>';
  });
  if (inList) html += '</ul>';
  return html.replace(/(<br>)+$/, '');
}
function fmtDatesInText(s) {
  return String(s || '').replace(/\\b(\\d{4})-(\\d{2})-(\\d{2})\\b/g, m => {
    const p = m.split('-').map(Number);
    if (p[1] < 1 || p[1] > 12 || p[2] < 1 || p[2] > 31) return m;
    const d = parseISODate(m);
    return d ? fmtDate(m) : m;
  });
}
function haiReplyHtml(r) {
  const text = String(r.reply || r.note || '').trim();
  if (text) return md(fmtDatesInText(text));
  if (r.status === 'needs_input' && r.approval)
    return 'I drafted something for you to review below.';
  return 'hai finished without a written reply — see What I did below.';
}
function copyRunId(btn) {
  const id = btn.getAttribute('data-run') || '';
  const done = () => { btn.textContent = 'Copied'; setTimeout(() => { btn.textContent = 'Copy'; }, 1500); };
  if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(id).then(done, done);
  else {
    const t = document.createElement('textarea');
    t.value = id;
    document.body.appendChild(t);
    t.select();
    try { document.execCommand('copy'); } catch (e) {}
    t.remove();
    done();
  }
}
function haiApprovalCard(a) {
  const d = document.createElement('div');
  d.className = 'appr';
  const msg = document.createElement('div');
  msg.className = 'appr-msg';
  msg.textContent = fmtDatesInText(a.message) || '(no message)';
  const to = document.createElement('div');
  to.className = 'appr-to';
  to.textContent = 'To: ' + humanNames(a.recipients);
  const parts = [msg, to];
  if (a.image) {
    const img = document.createElement('img');
    img.className = 'invite-img';
    img.src = imgUrl(a.image);
    img.alt = '';
    parts.push(img);
  }
  const actions = document.createElement('div');
  actions.className = 'appr-actions';
  const status = document.createElement('div');
  status.className = 'appr-status';
  status.hidden = true;
  const approveBtn = document.createElement('button');
  approveBtn.type = 'button';
  approveBtn.textContent = 'Approve';
  const declineBtn = document.createElement('button');
  declineBtn.type = 'button';
  declineBtn.className = 'warn';
  declineBtn.textContent = 'Decline';
  const finish = text => { actions.remove(); status.hidden = false; status.textContent = text; };
  const tryDecide = async (decision, doneText) => {
    approveBtn.disabled = true;
    declineBtn.disabled = true;
    const r = await decide(a.id, S.me, decision, '', document.createElement('span'));
    if (r && r.error) {
      approveBtn.disabled = false;
      declineBtn.disabled = false;
      retryBubble(r.error, () => tryDecide(decision, doneText));
    } else finish(doneText);
  };
  approveBtn.onclick = () => tryDecide('approved', 'Sent to ' + humanNames(a.recipients));
  declineBtn.onclick = () => tryDecide('rejected', 'Declined');
  actions.append(approveBtn, declineBtn);
  parts.push(actions, status);
  d.append(...parts);
  return d;
}
async function renderHaiApproval(approvalId) {
  try {
    const r = await get('approvals', {family: S.family, member: S.me});
    const a = (r.approvals || []).find(x => x.id === approvalId);
    if (!a || a.status !== 'pending') return;
    const log = el('convoLog');
    log.appendChild(haiApprovalCard(a));
    log.scrollTop = log.scrollHeight;
  } catch (e) {}
}
async function renderPendingApprovals(runId) {
  try {
    const r = await get('approvals', {family: S.family, member: S.me});
    const log = el('convoLog');
    (r.approvals || []).filter(a => a.run_id === runId && a.status === 'pending')
      .forEach(a => log.appendChild(haiApprovalCard(a)));
    log.scrollTop = log.scrollHeight;
  } catch (e) {}
}
function whatIDid(steps, runId) {
  if (!steps || !steps.length) return;
  const total = steps.reduce((a, s) => a + (s.duration_ms || 0), 0) / 1000;
  const link = runId
    ? '<br><button class="linkbtn" onclick="openRunInConsole(\\\'' + runId + '\\\')">View full run</button>'
    : '';
  const runLine = runId
    ? '<div class="runid"><span>run ' + esc(runId) + '</span>'
      + '<button type="button" class="copybtn" data-run="' + esc(runId) + '" onclick="copyRunId(this)">Copy</button></div>'
    : '';
  const rows = steps.map(stepRowHtml).join('');
  bubble('sys whatidid', '<details><summary>What I did · '
    + (steps.length === 1 ? '1 step' : steps.length + ' steps')
    + ' · ' + total.toFixed(1) + 's</summary>' + rows + link + runLine + '</details>');
}
function updateRailMe() {
  if (!S.me) {
    if (el('railAvatar')) el('railAvatar').textContent = '?';
    if (el('railName')) el('railName').textContent = '…';
    if (el('headAvatar')) el('headAvatar').textContent = '?';
    return;
  }
  const m = (S.members || []).find(x => x.id === S.me) || {id: S.me, name: S.me};
  const av = el('railAvatar'), nm = el('railName'), ha = el('headAvatar');
  if (av) { av.textContent = initials(m.name || m.id); av.style.background = avatarColor(m.id); }
  if (nm) nm.textContent = m.name || m.id;
  if (ha) { ha.textContent = initials(m.name || m.id); ha.style.background = avatarColor(m.id); }
}
function toggleAcct(open) {
  el('acctPanel').hidden = !open;
  el('acctScrim').hidden = !open;
  if (open) fillAcct();
}
document.addEventListener('keydown', ev => {
  if (ev.key === 'Escape' && !el('acctPanel').hidden) toggleAcct(false);
});
function acctGo(view) {
  toggleAcct(false);
  if (view === 'dashboard') { showView('dashboard'); setTimeout(() => el('dashNotes').scrollIntoView(), 50); return; }
  showView(view);
}
function fillAcct() {
  const m = (S.members || []).find(x => x.id === S.me) || {id: S.me, name: S.me};
  const av = el('acctAvatar');
  av.textContent = initials(m.name || m.id);
  av.style.background = avatarColor(m.id);
  el('acctName').textContent = m.name || m.id;
  el('acctRole').textContent = roleLabel(m.role);
}
function signOut() {
  toggleAcct(false);
  S.me = '';
  rememberMe('');
  S.profileView = '';
  S.peer = '';
  S.convo = 'hai';
  S.chatKey = '';
  S.haiRun = '';
  S.haiHistory = [];
  newChat();
  el('convoLog').innerHTML = '';
  updateRailMe();
  renderSignin();
  showView('signin');
}
const SIGNIN_GROUPS = [['Grandparents', ['grandmother', 'grandfather']],
  ['Parents', ['mother', 'father']],
  ['Kids', ['sister', 'brother']],
  ['Extended family', ['uncle', 'aunt']]];
function signinTile(m) {
  const b = document.createElement('button');
  b.className = 'signin-tile';
  b.innerHTML = '<span class="signin-avatar" style="background:' + signinAvatarColor(m.id) + '">'
    + esc(initials(m.name)) + '</span><b>' + esc(firstName(m.name)) + '</b><span class="signin-role">'
    + esc(roleLabel(m.role)) + '</span>';
  b.onclick = () => { setMe(m.id, true); showView('dashboard'); refreshAll(); refresh(); };
  return b;
}
function startAsLeo() {
  const ms = S.members || [];
  const leo = ms.find(m => firstName(m.name).toLowerCase() === 'leo') || ms.find(m => m.id === 'leo');
  if (!leo) return;
  setMe(leo.id, true); showView('dashboard'); refreshAll(); refresh();
}
function renderSignin() {
  const box = el('signinGroups');
  box.innerHTML = '';
  SIGNIN_GROUPS.forEach(([label, roles]) => {
    const ms = (S.members || []).filter(m => roles.includes(m.role));
    if (!ms.length) return;
    const g = document.createElement('div');
    g.innerHTML = '<h3 class="signin-group-label">' + label + '</h3>';
    const row = document.createElement('div');
    row.className = 'signin-row';
    ms.forEach(m => row.appendChild(signinTile(m)));
    g.appendChild(row);
    box.appendChild(g);
  });
  const rest = (S.members || []).filter(m => !SIGNIN_GROUPS.some(([, roles]) => roles.includes(m.role)));
  if (rest.length) {
    const g = document.createElement('div');
    g.innerHTML = '<h3 class="signin-group-label">Extended family</h3>';
    const row = document.createElement('div');
    row.className = 'signin-row';
    rest.forEach(m => row.appendChild(signinTile(m)));
    g.appendChild(row);
    box.appendChild(g);
  }
}
function openSwitch() {
  toggleAcct(false);
  const box = el('switchTiles');
  box.innerHTML = '';
  (S.members || []).forEach(m => {
    const b = signinTile(m);
    b.onclick = () => { closeSwitch(); doSwitch(m.id); };
    box.appendChild(b);
  });
  el('switchDlg').hidden = false;
}
function closeSwitch() {
  el('switchDlg').hidden = true;
}
function doSwitch(id) {
  if (!id || id === S.me) return;
  const main = el('main');
  main.classList.add('switching');
  setTimeout(() => {
    setMe(id, true);
    S.profileView = id;
    main.classList.remove('switching');
    showToast('Now signed in as ' + firstName(memberName(id)) + '.');
  }, 300);
}
let toastTimer = null;
function showToast(text) {
  const t = el('toast');
  t.textContent = text;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 2600);
}
function updateAgentHead() {
  if (!S.me) return;
  renderConvoHead();
  renderConvoListSel();
}
function subtab(name) {
  if (name === 'agent') S.convo = 'hai';
  else if (S.peer) S.convo = S.peer;
  if (!el('view-chat').hidden) openConvo(S.convo, true);
}
function openHai(keep) {
  if (!keep || !el('convoLog').children.length) newHaiChat(true);
  updateChips();
}
function newHaiChat(silent) {
  S.haiRun = '';
  el('convoLog').innerHTML = '';
  const sel = el('haiHist');
  if (sel) sel.value = '';
  el('chatStatus').textContent = '';
  updateChips();
}
function newChat() { newHaiChat(); }
function usePrompt(text) {
  el('msg').value = text;
  send();
}
async function seed() {
  const r = await api('seed', {});
  el('seedOut').textContent = JSON.stringify(r, null, 2);
  loadPeople();
  if (S.consoleLoaded) cLoadAll();
}
function firstMemberText(turns) {
  const t = (turns || []).find(t => t.from === 'member');
  return t ? t.text : '';
}
function histLabel(first, started) {
  const short = first.length > 42 ? first.slice(0, 42) + '…' : first;
  return (short || 'Chat') + ' · ' + dayLabel(started);
}
async function loadHaiHistory(autoOpen) {
  const key = S.family + ':' + S.me;
  if (S.chatKey !== key) { S.chatKey = key; S.haiRun = ''; }
  if (!S.me) return;
  const r = await get('conversations', {family: S.family, member: S.me});
  if (S.chatKey !== key) return;
  S.haiHistory = r.conversations || [];
  const sel = el('haiHist');
  if (sel) {
    sel.innerHTML = '';
    const cur = document.createElement('option');
    cur.value = '';
    cur.textContent = S.haiHistory.length ? 'Select a chat…' : '(no past chats yet)';
    sel.appendChild(cur);
    S.haiHistory.forEach(c => {
      const o = document.createElement('option');
      o.value = c.run_id;
      o.textContent = c.run_id;
      sel.appendChild(o);
      get('chat_history', {family: S.family, member: S.me, run_id: c.run_id}).then(h => {
        if (h && !h.error && o.isConnected)
          o.textContent = histLabel(firstMemberText(h.turns), c.started);
      });
    });
    sel.onchange = () => { if (sel.value) openChat(sel.value); };
    if (S.haiRun && S.haiHistory.some(c => c.run_id === S.haiRun)) sel.value = S.haiRun;
  }
  if (autoOpen && S.haiHistory.length && !S.haiRun) openChat(S.haiHistory[0].run_id);
}
function loadPastChats(autoOpen) { return loadHaiHistory(autoOpen); }
async function openChat(runId) {
  const key = S.chatKey;
  S.haiRun = runId;
  const sel = el('haiHist');
  if (sel) sel.value = runId;
  const r = await get('chat_history', {family: S.family, member: S.me, run_id: runId});
  if (S.chatKey !== key) return;
  const log = el('convoLog');
  log.innerHTML = '';
  if (r.error) { bubble('sys', esc(r.error)); return; }
  let pendingAgent = null;
  const flushAgent = () => {
    if (pendingAgent) bubble('agent', md(pendingAgent));
    pendingAgent = null;
  };
  (r.turns || []).forEach(t => {
    if (t.from === 'member') { flushAgent(); bubble('me', esc(t.text)); }
    else if (t.from === 'agent') pendingAgent = t.text;
  });
  flushAgent();
  whatIDid(r.steps, runId);
  await renderPendingApprovals(runId);
  updateChips();
}
function agentReminders() {}
function autogrow() {
  const box = el('msg');
  box.style.height = 'auto';
  box.style.height = Math.min(box.scrollHeight, 144) + 'px';
}
async function send(priorText, isRetry) {
  if (S.convo !== 'hai') { dmSend(); return; }
  const msgEl = el('msg');
  const text = isRetry ? priorText : msgEl.value;
  if (!text || !text.trim()) return;
  if (!isRetry) {
    bubble('me', esc(text));
    msgEl.value = '';
    autogrow();
  }
  const btn = el('sendBtn');
  btn.disabled = true;
  btn.textContent = 'Sending…';
  el('chatStatus').textContent = 'hai is thinking…';
  const live = bubble('sys whatidid live-steps',
    '<b>hai is working…</b><div class="live-steps"></div>');
  const liveBox = live.querySelector('.live-steps');
  renderLiveSteps(liveBox, []);
  const runId = (S.haiRun || '').trim();
  const gen = S.sendGen || 0;
  let pollRun = runId, prevRun = '', base = 0, stopped = false;
  if (!pollRun) {
    try {
      const c = await get('conversations', {family: S.family, member: S.me});
      prevRun = ((c.conversations || [])[0] || {}).run_id || '';
    } catch (e) {}
  } else {
    try {
      const r0 = await get('run_steps', {family: S.family, member: S.me, run_id: pollRun});
      base = (r0.steps || []).length;
    } catch (e) {}
  }
  const poll = async () => {
    while (!stopped && gen === (S.sendGen || 0)) {
      try {
        if (!pollRun) {
          const c = await get('conversations', {family: S.family, member: S.me});
          const newest = (c.conversations || [])[0];
          if (newest && newest.run_id !== prevRun) pollRun = newest.run_id;
        }
        if (pollRun) {
          const rs = await get('run_steps', {family: S.family, member: S.me, run_id: pollRun});
          if (!stopped) renderLiveSteps(liveBox, (rs.steps || []).slice(base));
        }
      } catch (e) {}
      if (!stopped) await new Promise(r => setTimeout(r, 500));
    }
  };
  poll();
  console.log('[hai] run', runId || 'new');
  try {
    const r = runId
      ? await api('message', {family: S.family, member: S.me, run_id: runId, message: text})
      : await api('start', {family: S.family, member: S.me, message: text});
    stopped = true;
    live.remove();
    if (gen !== (S.sendGen || 0)) return;
    console.log('[hai] run', r.run_id || runId || 'done');
    if (r.error) {
      retryBubble(r.error, () => send(text, true));
      el('chatStatus').textContent = 'Something went wrong — try again.';
    } else {
      if (r.run_id) S.haiRun = r.run_id;
      bubble('agent', haiReplyHtml(r));
      if (r.status === 'needs_input' && r.approval)
        renderHaiApproval(r.approval);
      whatIDid(r.steps, r.run_id);
      el('chatStatus').textContent = '';
    }
  } catch (err) {
    stopped = true;
    live.remove();
    if (gen !== (S.sendGen || 0)) return;
    retryBubble(String(err && err.message || err), () => send(text, true));
    el('chatStatus').textContent = 'Something went wrong — try again.';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Send';
    refresh();
    loadHaiHistory(false);
  }
}
let recState = null;
function fmtElapsed(ms) {
  const s = Math.floor(ms / 1000);
  return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0');
}
function updateRecTime() {
  if (recState) el('recTime').textContent = fmtElapsed(Date.now() - recState.started);
}
async function toggleMic() {
  if (S.convo !== 'hai') return;
  if (recState) { stopVoiceSend(false); return; }
  var Ctx = window.AudioContext || window.webkitAudioContext;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !Ctx) {
    el('chatStatus').textContent = 'Voice notes need a desktop browser with microphone access.';
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({audio: true});
    const ctx = new Ctx();
    const src = ctx.createMediaStreamSource(stream);
    const proc = ctx.createScriptProcessor(4096, 1, 1);
    const quiet = ctx.createGain();
    quiet.gain.value = 0;
    const chunks = [];
    proc.onaudioprocess = ev => { chunks.push(new Float32Array(ev.inputBuffer.getChannelData(0))); };
    src.connect(proc);
    proc.connect(quiet);
    quiet.connect(ctx.destination);
    recState = {stream: stream, ctx: ctx, proc: proc, src: src, chunks: chunks,
      inRate: ctx.sampleRate, started: Date.now(), timer: setInterval(updateRecTime, 500)};
    el('recTime').textContent = '0:00';
    el('recBar').hidden = false;
    el('chatStatus').textContent = '';
  } catch (e) {
    recState = null;
    el('recBar').hidden = true;
    el('chatStatus').textContent = 'Microphone unavailable or permission denied.';
  }
}
function encodeWav16k(chunks, inRate) {
  const outRate = 16000;
  let len = 0;
  chunks.forEach(c => { len += c.length; });
  const data = new Float32Array(len);
  let off = 0;
  chunks.forEach(c => { data.set(c, off); off += c.length; });
  const ratio = inRate / outRate;
  const outLen = Math.floor(len / ratio);
  const out = new Int16Array(outLen);
  for (let i = 0; i < outLen; i++) {
    let v = data[Math.round(i * ratio)] || 0;
    v = Math.max(-1, Math.min(1, v));
    out[i] = v < 0 ? v * 32768 : v * 32767;
  }
  const buf = new ArrayBuffer(44 + outLen * 2);
  const v = new DataView(buf);
  const wstr = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  wstr(0, 'RIFF');
  v.setUint32(4, 36 + outLen * 2, true);
  wstr(8, 'WAVE');
  wstr(12, 'fmt ');
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, outRate, true);
  v.setUint32(28, outRate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  wstr(36, 'data');
  v.setUint32(40, outLen * 2, true);
  for (let i = 0; i < outLen; i++) v.setInt16(44 + i * 2, out[i], true);
  return new Blob([buf], {type: 'audio/wav'});
}
function stopVoiceSend(cancelled) {
  if (!recState) return;
  const st = recState;
  recState = null;
  clearInterval(st.timer);
  el('recBar').hidden = true;
  try { st.proc.disconnect(); } catch (e) {}
  try { st.src.disconnect(); } catch (e) {}
  st.stream.getTracks().forEach(t => t.stop());
  try { st.ctx.close(); } catch (e) {}
  if (cancelled) return;
  const blob = encodeWav16k(st.chunks, st.inRate || 44100);
  if (!blob.size) {
    el('chatStatus').textContent = 'No audio captured — try again.';
    return;
  }
  sendVoice(blob, URL.createObjectURL(blob));
}
function cancelVoice() {
  stopVoiceSend(true);
}
function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => {
      const s = String(fr.result || '');
      resolve(s.indexOf(',') >= 0 ? s.slice(s.indexOf(',') + 1) : s);
    };
    fr.onerror = () => reject(fr.error || new Error('read failed'));
    fr.readAsDataURL(blob);
  });
}
function voiceBubble(playUrl, transcript) {
  const d = document.createElement('div');
  d.className = 'bubble me voice';
  const a = document.createElement('audio');
  a.controls = true;
  a.src = playUrl;
  a.preload = 'metadata';
  const t = document.createElement('div');
  t.className = 'vtranscript';
  t.textContent = transcript;
  d.append(a, t);
  const log = el('convoLog');
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
  updateChips();
  return d;
}
function retryBubble(message, retryFn) {
  const d = document.createElement('div');
  d.className = 'bubble sys';
  const row = document.createElement('div');
  row.className = 'verror';
  const t = document.createElement('span');
  t.textContent = message;
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'ghost';
  b.textContent = 'Retry';
  b.onclick = () => { d.remove(); retryFn(); };
  row.append(t, b);
  d.appendChild(row);
  const log = el('convoLog');
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
  return d;
}
function voiceErrorBubble(message, retryFn) {
  return retryBubble(message, retryFn);
}
async function sendVoice(blob, playUrl) {
  if (S.convo !== 'hai') return;
  const btn = el('sendBtn');
  const mic = el('micBtn');
  btn.disabled = true;
  mic.disabled = true;
  btn.textContent = 'Sending…';
  el('chatStatus').textContent = 'Transcribing voice note…';
  const live = bubble('sys whatidid live-steps',
    '<b>hai is working…</b><div class="live-steps"></div>');
  const liveBox = live.querySelector('.live-steps');
  renderLiveSteps(liveBox, []);
  const runId = (S.haiRun || '').trim();
  const gen = S.sendGen || 0;
  let pollRun = runId, prevRun = '', base = 0, stopped = false;
  if (!pollRun) {
    try {
      const c = await get('conversations', {family: S.family, member: S.me});
      prevRun = ((c.conversations || [])[0] || {}).run_id || '';
    } catch (e) {}
  } else {
    try {
      const r0 = await get('run_steps', {family: S.family, member: S.me, run_id: pollRun});
      base = (r0.steps || []).length;
    } catch (e) {}
  }
  const poll = async () => {
    while (!stopped && gen === (S.sendGen || 0)) {
      try {
        if (!pollRun) {
          const c = await get('conversations', {family: S.family, member: S.me});
          const newest = (c.conversations || [])[0];
          if (newest && newest.run_id !== prevRun) pollRun = newest.run_id;
        }
        if (pollRun) {
          const rs = await get('run_steps', {family: S.family, member: S.me, run_id: pollRun});
          if (!stopped) renderLiveSteps(liveBox, (rs.steps || []).slice(base));
        }
      } catch (e) {}
      if (!stopped) await new Promise(r => setTimeout(r, 500));
    }
  };
  poll();
  console.log('[hai] run', runId || 'new voice note');
  try {
    const audio_b64 = await blobToBase64(blob);
    const r = await api('voice_note', {family: S.family, member: S.me,
      audio_b64: audio_b64, mime: blob.type || 'audio/webm', run_id: runId});
    stopped = true;
    live.remove();
    if (gen !== (S.sendGen || 0)) return;
    if (r.error) {
      voiceErrorBubble(r.error, () => sendVoice(blob, playUrl));
      el('chatStatus').textContent = 'Something went wrong — try again.';
    } else {
      if (r.run_id) S.haiRun = r.run_id;
      voiceBubble(playUrl, r.transcript || '');
      bubble('agent', haiReplyHtml(r));
      if (r.status === 'needs_input' && r.approval)
        renderHaiApproval(r.approval);
      whatIDid(r.steps, r.run_id);
      el('chatStatus').textContent = '';
    }
  } catch (err) {
    stopped = true;
    live.remove();
    if (gen !== (S.sendGen || 0)) return;
    voiceErrorBubble(String(err && err.message || err), () => sendVoice(blob, playUrl));
    el('chatStatus').textContent = 'Something went wrong — try again.';
  } finally {
    btn.disabled = false;
    mic.disabled = false;
    btn.textContent = 'Send';
    refresh();
    loadHaiHistory(false);
  }
}
el('msg').addEventListener('input', autogrow);
el('msg').addEventListener('keydown', ev => {
  if (ev.key === 'Enter' && !ev.shiftKey) { ev.preventDefault(); send(); }
});
el('convoLog').addEventListener('click', ev => {
  const img = ev.target.closest ? ev.target.closest('img.invite-img') : null;
  if (!img) return;
  el('imgOverlayImg').src = imgUrl(img.currentSrc || img.src);
  el('imgOverlay').hidden = false;
});
function closeImgOverlay() {
  el('imgOverlay').hidden = true;
  el('imgOverlayImg').removeAttribute('src');
}
function refreshImageUrls() {
  document.querySelectorAll('img.invite-img').forEach(img => {
    img.setAttribute('src', imgUrl(img.getAttribute('src') || img.src));
  });
  const overlay = el('imgOverlayImg');
  if (overlay && overlay.getAttribute('src')) {
    overlay.setAttribute('src', imgUrl(overlay.getAttribute('src')));
  }
}
el('imgOverlay').addEventListener('click', ev => {
  if (ev.target === el('imgOverlay') || (ev.target.closest && ev.target.closest('.imgclose'))) closeImgOverlay();
});
document.addEventListener('keydown', ev => {
  if (ev.key === 'Escape' && el('imgOverlay') && !el('imgOverlay').hidden) closeImgOverlay();
});
new MutationObserver(muts => {
  const log = el('convoLog');
  muts.forEach(m => m.addedNodes.forEach(n => {
    if (!n || n.nodeType !== 1) return;
    const imgs = (n.matches && n.matches('img.invite-img')) ? [n]
      : (n.querySelectorAll ? Array.from(n.querySelectorAll('img.invite-img')) : []);
    imgs.forEach(img => {
      if (img.complete && img.naturalWidth) return;
      img.addEventListener('load', () => {
        if (log.scrollHeight - log.scrollTop - log.clientHeight < 40) log.scrollTop = log.scrollHeight;
      }, {once: true});
    });
  }));
}).observe(el('convoLog'), {childList: true, subtree: true});
async function cLoadAll() {
  await cLoadFamily();
  await cLoadMemories();
  await cLoadReminders();
  await refresh();
}
async function cLoadFamily() {
  const r = await get('members', {family: S.family});
  const box = el('memberList');
  box.innerHTML = '';
  if (r.error) { box.textContent = r.error; return; }
  r.members.forEach(m => {
    const b = document.createElement('button');
    b.className = 'rosterbtn';
    b.textContent = m.name;
    b.onclick = () => {
      box.querySelectorAll('.rosterbtn').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      el('memberDetail').innerHTML = '<b>' + esc(m.name) + '</b> (' + esc(m.id) + ')'
        + '<br>role: <span class="rolecap">' + esc(roleLabel(m.role)) + '</span>'
        + '<br>email: ' + esc(m.email || '-')
        + '<br>phone: ' + esc(m.phone || '-');
    };
    box.appendChild(b);
  });
  fillSelect(el('memView'), r.members, el('memView').value || S.me);
}
async function cLoadMemories() {
  const r = await get('memories', {family: S.family,
    member: el('memView').value || (el('memView').options[0] || {}).value || ''});
  const tiles = el('memTiles');
  tiles.innerHTML = '';
  if (r.error) { tiles.textContent = r.error; return; }
  S.memCache = r.memories || [];
  S.memFilter = '';
  const catAccess = {shared: 'Everyone in the family', private: 'Owner only (+ anyone they shared it with)', restricted: 'Owner + named members only'};
  ['shared', 'private', 'restricted'].forEach(cat => {
    const n = S.memCache.filter(m => m.category === cat).length;
    const t = document.createElement('div');
    t.className = 'tile ' + cat;
    t.id = 'tile-' + cat;
    t.innerHTML = '<span class="privacy-badge ' + esc(cat) + '">' + esc(cat) + '</span>'
      + '<small>' + (n === 1 ? '1 memory' : n + ' memories') + '</small><span class="tile-access">' + esc(catAccess[cat]) + '</span>';
    t.onclick = () => memFilter(cat);
    tiles.appendChild(t);
  });
  el('memList').innerHTML = '<span class="hint">Pick a category tile to list its memories.</span>';
}
function memFilter(cat) {
  S.memFilter = S.memFilter === cat ? '' : cat;
  ['shared', 'private', 'restricted'].forEach(c =>
    el('tile-' + c).classList.toggle('active', S.memFilter === c));
  const list = S.memCache.filter(m => !S.memFilter || m.category === S.memFilter);
  const box = el('memList');
  box.innerHTML = '';
  if (!list.length) { box.textContent = '(no memories)'; return; }
  list.forEach(m => {
    const d = document.createElement('div');
    d.className = 'mem';
    d.innerHTML = esc(m.text) + '<div class="meta">' + esc(m.id) + ' &middot; owner ' + esc(memberName(m.owner))
      + (m.people && m.people.length ? ' &middot; about: ' + esc(humanNames(m.people)) : '')
      + (m.event_date ? ' &middot; ' + esc(m.event_date) : '') + '</div>';
    d.insertBefore(privacyChip(privacyInfoForMemory(m)), d.firstChild);
    box.appendChild(d);
  });
}
async function cLoadReminders() {
  const member = el('memView').value || S.me;
  el('remAs').textContent = member;
  const r = await get('due_reminders', {family: S.family,
    member: member, days_ahead: el('remDays').value || '60'});
  const box = el('remList');
  box.innerHTML = '';
  if (r.error) { box.textContent = r.error; return; }
  (r.items || []).forEach(it => {
    const d = reminderCard(it, false);
    const row = document.createElement('div');
    row.className = 'row';
    const btn = document.createElement('button');
    btn.textContent = 'Propose';
    const out = document.createElement('span');
    out.className = 'hint';
    btn.onclick = async () => {
      const p = await api('propose_reminder', {family: S.family, by: member,
        key: it.key, title: it.title, date: it.date, recipients: it.recipients,
        message: it.suggested_message});
      out.textContent = ' ' + JSON.stringify(p);
      if (!p.error) { btn.disabled = true; refresh(); }
    };
    row.append(btn, out);
    d.appendChild(row);
    box.appendChild(d);
  });
  if (!box.children.length) box.textContent = '(no reminders in this window)';
}
function consoleMember() {
  return (el('memView') && el('memView').value) || S.me || '';
}
async function refresh() {
  const member = consoleMember();
  const box = el('approvals');
  if (!member) { box.innerHTML = ''; box.textContent = 'Choose a member first.'; return; }
  const r = await get('approvals', {family: S.family, member: member});
  box.innerHTML = '';
  (r.approvals || []).forEach(a => {
    const card = approvalCard(a, {
      useInput: true,
      defaultBy: el('memView').value || S.me,
      onDecide: (ap, decision, by, out) => {
        if (decision === 'rejected') decide(ap.id, by, 'rejected', prompt('rejection note:', '') || '', out);
        else decide(ap.id, by, 'approved', '', out);
      },
    });
    const idLine = document.createElement('div');
    idLine.className = 'hint';
    idLine.innerHTML = 'approval <b class="id">' + esc(a.id) + '</b>';
    card.insertBefore(idLine, card.firstChild);
    box.appendChild(card);
  });
  if (!box.children.length) box.textContent = r.error || '(none pending)';
}
async function decide(id, by, decision, note, out) {
  let r;
  try {
    r = await api('decide', {family: S.family, member: by,
      approval_id: id, by: by, decision: decision, note: note || ''});
  } catch (err) {
    r = {error: String(err && err.message || err)};
  }
  if (out) out.textContent = ' ' + (r.error || JSON.stringify(r));
  refresh();
  return r;
}
function openRunInConsole(runId) {
  el('runInspect').value = runId || '';
  showView('settings');
  inspectRun();
}
async function inspectRun() {
  const member = consoleMember();
  const box = el('runSteps');
  box.innerHTML = '';
  if (!member) { el('runOut').textContent = 'Choose a member first.'; return; }
  const r = await get('run', {family: S.family, member: member, run_id: el('runInspect').value.trim()});
  if (r.error) {
    el('runOut').textContent = /not owned by/.test(r.error)
      ? 'This run belongs to another family member. Switch to them to inspect it.'
      : r.error;
    return;
  }
  if ((r.steps || []).length) {
    const tl = document.createElement('div');
    tl.className = 'timeline';
    tl.innerHTML = '<b>Model steps</b>' + r.steps.map(s =>
      '<div class="step-row"><span class="pill ' + esc(s.status) + '">' + esc(s.status)
      + '</span> <b>#' + s.step + ' ' + esc(s.task) + '</b> · '
      + esc(s.model_label || s.model || '') + ' · ' + (s.duration_ms / 1000).toFixed(1)
      + 's · in ' + s.input_tokens + ' / out ' + s.output_tokens + ' tokens<br>'
      + '<span class="hint">' + esc(s.started || '') + ' — ' + esc(s.summary || '')
      + '</span></div>').join('');
    box.appendChild(tl);
  } else {
    box.innerHTML = '<span class="hint">No model steps recorded for this run yet.</span>';
  }
  el('runOut').textContent = JSON.stringify(r, null, 2);
}
async function leakRun() {
  const r = await api('leak_test_start', {});
  el('leakOut').innerHTML = '<span class="hint">Running… ' + (r.done || 0) + '/' + (r.total || 40) + ' prompts</span>';
  leakPoll();
}
async function leakPoll() {
  const r = await get('leak_test_status', {});
  const box = el('leakOut');
  if (r.state === 'running') {
    box.innerHTML = '<span class="hint">Running… ' + r.done + '/' + r.total + ' prompts (live model calls, please wait)</span>';
    setTimeout(leakPoll, 3000);
    return;
  }
  if (r.state === 'error') { box.textContent = 'Leak test failed: ' + r.error; return; }
  if (r.state === 'done' && r.report) { leakRender(r.report); return; }
  box.innerHTML = '<span class="hint">Not run yet in this session.</span>';
}
function leakRender(report) {
  const box = el('leakOut');
  box.innerHTML = '';
  const s = report.summary;
  const head = document.createElement('div');
  head.innerHTML = '<b>' + s.overall.passed + '/' + s.overall.total + ' passed</b>'
    + ' · ' + report.total_seconds + 's · agent: ' + esc(report.models.agent)
    + ' · judge: ' + esc(report.models.judge) + ' · verification ' + esc(report.verification)
    + '<br><span class="hint">Saved to ' + esc(report.saved_to || '') + '</span>';
  box.appendChild(head);
  const mk = (title, tallies) => {
    const d = document.createElement('div');
    d.innerHTML = '<b>' + title + '</b><br>' + Object.keys(tallies).map(k =>
      esc(k) + ': ' + tallies[k].passed + '/' + tallies[k].total).join('<br>');
    box.appendChild(d);
  };
  mk('Per person', s.by_person);
  mk('Per secret', s.by_secret);
  const fails = (report.failures || []).concat(report.control_failures || []);
  if (!fails.length) {
    const ok = document.createElement('div');
    ok.innerHTML = '<span class="pill ok">no leaks</span>';
    box.appendChild(ok);
  }
  fails.forEach(f => {
    const d = document.createElement('div');
    d.className = 'approval';
    d.innerHTML = '<b>' + esc(f.member) + '</b> · ' + esc(f.secret)
      + (f.control ? ' · <span class="pill error">control should have mentioned it</span>' : ' · <span class="pill error">leak</span>')
      + '<br><b>Prompt:</b> ' + esc(f.prompt)
      + '<br><b>Reply:</b> ' + esc(f.reply)
      + (f.judge ? '<br><b>Judge:</b> ' + esc(f.judge.verdict) + ' — ' + esc(f.judge.reason) : '')
      + '<br><span class="hint">models: ' + esc((f.models || []).join(', ') || '—')
      + ' · steps: ' + (f.steps || []).length + '</span>';
    box.appendChild(d);
  });
}
function memberName(id) {
  const m = (S.members || []).find(x => x.id === id);
  return m ? m.name : id;
}
function humanNames(ids) {
  return (ids || []).map(id => memberName(id)).join(', ') || '—';
}
function parseISODate(s) {
  const p = String(s || '').split('-').map(Number);
  if (p.length !== 3 || p.some(n => !Number.isFinite(n))) return null;
  return new Date(p[0], p[1] - 1, p[2]);
}
function countdownInfo(dateStr) {
  const d = parseISODate(dateStr);
  if (!d) return {text: '', cls: ''};
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const days = Math.round((d - today) / 86400000);
  if (days === 0) return {text: 'today', cls: 'today'};
  if (days === 1) return {text: 'tomorrow', cls: ''};
  if (days === -1) return {text: 'yesterday', cls: 'overdue'};
  if (days > 1) return {text: 'in ' + days + ' days', cls: ''};
  return {text: Math.abs(days) + ' days ago', cls: 'overdue'};
}
function statusPillHtml(status) {
  const s = String(status || 'upcoming');
  const label = {due: 'due', upcoming: 'upcoming', pending: 'pending',
    approved: 'approved', rejected: 'rejected', proposed: 'proposed'}[s] || s;
  return '<span class="pill ' + esc(s) + '">' + esc(label) + '</span>';
}
function fmtDate(iso) {
  const d = parseISODate(iso);
  if (!d) return iso || '';
  return d.toLocaleDateString(undefined, {weekday: 'short', month: 'short', day: 'numeric'});
}
const KIND_LABELS = {birthday: 'Birthday', anniversary: 'Anniversary', event: 'Event',
  holiday: 'Holiday', public_holiday: 'Holiday', surprise_party: 'Surprise party',
  celebration: 'Celebration', shared: 'Shared', private: 'Private', restricted: 'Restricted'};
function kindName(kind) {
  return KIND_LABELS[kind] || String(kind || 'event').replace(/_/g, ' ')
    .replace(/^./, c => c.toUpperCase());
}
function reminderCard(it, showAsk) {
  const cd = countdownInfo(it.date);
  const d = document.createElement('div');
  d.className = 'reminder-card';
  const pill = '<span class="pill ' + esc(it.status || 'upcoming') + '">'
    + esc((cd.text ? cd.text + ' · ' : '') + (it.status || 'upcoming')) + '</span>';
  d.innerHTML = pill + ' ' + kindBadge(it) + ' <b>' + esc(it.title) + '</b> <span class="pvslot"></span>'
    + '<br><span class="hint">' + esc(fmtDate(it.date)) + '</span>'
    + (it.image ? '<img class="invite-img" src="' + esc(imgUrl(it.image)) + '" alt="">' : '')
    + (it.suggested_message ? '<br>' + esc(it.suggested_message) : '');
  d.querySelector('.pvslot').replaceWith(privacyChip(privacyInfoForItem(it)));
  if (showAsk) {
    const row = document.createElement('div');
    row.className = 'row';
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'ghost';
    b.textContent = 'Ask hai to help plan';
    b.onclick = () => calAskHai(it);
    row.appendChild(b);
    d.appendChild(row);
  }
  return d;
}
const PV_ICONS = {
  shared: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
  private: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>',
  restricted: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/></svg>'
};
function rosterIds() { return (S.members || []).map(m => m.id); }
function memberIdFor(p) {
  const s = String(p || '').trim().toLowerCase();
  if (!s) return null;
  const hit = (S.members || []).find(m => m.id.toLowerCase() === s
    || String(m.name || '').toLowerCase() === s
    || firstName(m.name).toLowerCase() === s);
  return hit ? hit.id : null;
}
function titleMentions(title, id) {
  const m = (S.members || []).find(x => x.id === id);
  if (!m) return false;
  const first = firstName(m.name || '').toLowerCase();
  if (first.length < 3) return false;
  const words = String(title || '').toLowerCase().split(/[^a-z]+/);
  return words.indexOf(first) !== -1;
}
function itemHonoree(it) {
  const all = rosterIds();
  const excluded = all.filter(id => !(it.recipients || []).includes(id));
  if (!excluded.length) return null;
  if (it.kind === 'birthday' || it.kind === 'surprise_party') {
    const named = excluded.filter(id => titleMentions(it.title, id));
    if (named.length) return named[0];
    if (excluded.length === 1) return excluded[0];
  }
  return null;
}
function memoryHonoree(m, excluded) {
  const about = (m.people || []).map(memberIdFor).filter(id => id && excluded.includes(id));
  if (about.length === 1) return about[0];
  return null;
}
function privacyLabel(scope, canSee, cantSee, honoree) {
  const firstNames = ids => ids.map(id => firstName(memberName(id)));
  if (scope === 'shared' || !cantSee.length) return 'Everyone';
  if (scope === 'restricted' && honoree) return 'Secret from ' + firstName(memberName(honoree));
  if (scope === 'private') {
    const others = canSee.filter(id => id !== S.me);
    if (!others.length) return 'Only you';
    return 'You + ' + firstNames(others).join(', ');
  }
  if (cantSee.length > 2)
    return 'Only ' + firstNames(canSee).join(', ') + (canSee.length === 1 ? ' knows' : ' know');
  return 'Hidden from ' + firstNames(cantSee).join(', ');
}
function privacyInfoForMemory(m) {
  const all = rosterIds();
  if (m.category === 'shared' || !all.length)
    return {scope: 'shared', label: 'Everyone', canSee: all.slice(), cantSee: []};
  const auth = [...new Set([m.owner].concat(m.shared_with || []))];
  const canSee = auth.filter(id => all.includes(id));
  const cantSee = all.filter(id => !auth.includes(id));
  if (!cantSee.length) return {scope: 'shared', label: 'Everyone', canSee: canSee, cantSee: []};
  const scope = m.category === 'private' ? 'private' : 'restricted';
  const honoree = scope === 'restricted' ? memoryHonoree(m, cantSee) : null;
  return {scope: scope, label: privacyLabel(scope, canSee, cantSee, honoree), canSee: canSee, cantSee: cantSee};
}
function privacyInfoForItem(it) {
  const all = rosterIds();
  const r = (it.recipients || []).slice();
  const coversAll = all.length > 0 && all.every(id => r.includes(id));
  if (coversAll || !all.length)
    return {scope: 'shared', label: 'Everyone', canSee: all.slice(), cantSee: []};
  const canSee = r.filter(id => all.includes(id));
  const cantSee = all.filter(id => !r.includes(id));
  if (!cantSee.length) return {scope: 'shared', label: 'Everyone', canSee: canSee, cantSee: []};
  const honoree = itemHonoree(it);
  let scope;
  if (it.kind === 'surprise_party' || it.kind === 'restricted') scope = 'restricted';
  else if (it.kind === 'birthday' && honoree && !r.includes(honoree)) scope = 'restricted';
  else scope = (r.length <= 2 && r.includes(S.me)) ? 'private' : 'restricted';
  return {scope: scope, label: privacyLabel(scope, canSee, cantSee, honoree), canSee: canSee, cantSee: cantSee};
}
function pvPerson(id) {
  const row = document.createElement('div');
  row.className = 'pvperson';
  const av = document.createElement('span');
  av.className = 'pvavatar';
  av.style.background = avatarColor(id);
  av.textContent = initials(memberName(id));
  const nm = document.createElement('span');
  nm.textContent = firstName(memberName(id)) + (id === S.me ? ' (you)' : '');
  row.append(av, nm);
  return row;
}
function privacyChip(info) {
  const wrap = document.createElement('span');
  wrap.className = 'pvwrap';
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'pvchip pv-' + info.scope;
  b.innerHTML = PV_ICONS[info.scope] + '<span>' + esc(info.label) + '</span>';
  const pop = document.createElement('div');
  pop.className = 'pvpop';
  pop.hidden = true;
  const t1 = document.createElement('div');
  t1.className = 'pvrow-title';
  t1.textContent = 'Can see';
  pop.appendChild(t1);
  info.canSee.forEach(id => pop.appendChild(pvPerson(id)));
  const t2 = document.createElement('div');
  t2.className = 'pvrow-title';
  t2.textContent = "Can't see";
  pop.appendChild(t2);
  if (info.cantSee.length) info.cantSee.forEach(id => pop.appendChild(pvPerson(id)));
  else {
    const n = document.createElement('div');
    n.className = 'hint';
    n.textContent = 'Nobody — everyone can see this.';
    pop.appendChild(n);
  }
  b.onclick = ev => {
    ev.stopPropagation();
    const wasHidden = pop.hidden;
    document.querySelectorAll('.pvpop').forEach(p => { p.hidden = true; });
    pop.hidden = !wasHidden;
  };
  wrap.append(b, pop);
  return wrap;
}
document.addEventListener('click', ev => {
  if (!ev.target.closest || !ev.target.closest('.pvwrap'))
    document.querySelectorAll('.pvpop').forEach(p => { p.hidden = true; });
});
function reAskApproval(a) {
  showView('chat');
  subtab('agent');
  el('msg').value = 'I rejected approval ' + a.id + ' (' + a.action + '). The message was: "'
    + (a.message || '') + '" to ' + humanNames(a.recipients)
    + (a.note ? ' (' + a.note + ')' : '')
    + '. Please help me re-ask for this with any changes I want.';
  el('msg').focus();
}
function approvalHeadline(a) {
  const recips = a.recipients || [];
  const all = rosterIds();
  const who = (all.length && recips.length && all.every(id => recips.includes(id)))
    ? 'the family'
    : recips.map(id => firstName(memberName(id)) || id).join(', ') || 'the family';
  if (a.action === 'schedule_reminder') return 'Reminder to ' + who;
  if (a.action === 'send_message') return 'Message to ' + who;
  if (a.action === 'share_memory') return 'Share with ' + who;
  return String(a.action || 'approval').replace(/_/g, ' ');
}
function approvalCard(a, opts) {
  opts = opts || {};
  const d = document.createElement('div');
  d.className = 'approval-card';
  const status = a.status || 'pending';
  let head = statusPillHtml(status) + ' <b>' + esc(approvalHeadline(a)) + '</b>'
    + ' &middot; requested by ' + esc(memberName(a.requested_by || ''))
    + '<br><span class="hint">to: ' + esc(humanNames(a.recipients)) + '</span>';
  if (a.note) head += '<br><span class="hint">' + esc(a.note) + '</span>';
  if (a.memory_ids && a.memory_ids.length) head += '<br><span class="hint">memories: ' + esc(a.memory_ids.join(', ')) + '</span>';
  const top = document.createElement('div');
  top.innerHTML = head;
  top.appendChild(privacyChip(privacyInfoForItem(a)));
  const preview = document.createElement('div');
  preview.className = 'approval-preview';
  preview.textContent = fmtDatesInText(a.message) || '(no message)';
  const hint = document.createElement('div');
  hint.className = 'hint';
  hint.textContent = 'Approve sends this as-is. To change it, Reject and re-ask hai in Chat.';
  const actions = document.createElement('div');
  actions.className = 'approval-actions';
  const out = document.createElement('span');
  out.className = 'hint';
  let byInput = null;
  if (opts.useInput) {
    byInput = document.createElement('input');
    byInput.placeholder = 'decide as';
    byInput.value = opts.defaultBy || 'maya';
    byInput.size = 10;
    actions.appendChild(byInput);
  }
  const ok = document.createElement('button');
  ok.type = 'button';
  ok.textContent = 'Approve';
  const no = document.createElement('button');
  no.type = 'button';
  no.textContent = 'Reject';
  no.className = 'warn';
  const decider = () => byInput ? byInput.value : (opts.by || S.me);
  if (opts.onDecide) {
    ok.onclick = () => opts.onDecide(a, 'approved', decider(), out);
    no.onclick = () => opts.onDecide(a, 'rejected', decider(), out);
  }
  const reask = document.createElement('button');
  reask.type = 'button';
  reask.className = 'ghost';
  reask.textContent = 'Re-ask hai in Chat';
  reask.onclick = () => reAskApproval(a);
  actions.append(ok, no, reask, out);
  if (a.image) {
    const img = document.createElement('img');
    img.className = 'invite-img';
    img.src = imgUrl(a.image);
    img.alt = '';
    d.append(top, preview, img, hint, actions);
  } else {
    d.append(top, preview, hint, actions);
  }
  return d;
}
function initials(name) {
  const parts = String(name || '?').trim().split(/\\s+/);
  return ((parts[0] || '?')[0] + ((parts[1] || '')[0] || '')).toUpperCase();
}
function firstName(name) { return String(name || '').split(/\\s+/)[0] || ''; }
function roleLabel(role) {
  const r = String(role || 'member');
  if (r === 'sister') return 'Daughter';
  if (r === 'brother') return 'Son';
  return r;
}
function avatarColor(id) {
  return signinAvatarColor(id);
}
function signinAvatarColor(id) {
  const palette = ['#5E7F64', '#B08D5C', '#B35A45'];
  let h = 0;
  for (const ch of String(id)) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return palette[h % palette.length];
}
async function bootMembers() {
  const r = await get('members', {family: S.family});
  if (r.error || !(r.members || []).length) {
    el('signinGroups').innerHTML = '<span class="hint">Could not load the family: ' + esc(r.error || 'empty') + '</span>';
    return;
  }
  S.members = r.members;
  if (S.members.some(m => m.id === S.me)) {
    showView('dashboard');
    refreshAll();
    refresh();
  } else {
    S.me = '';
    rememberMe('');
    updateRailMe();
    renderSignin();
    showView('signin');
  }
}
function refreshAll() {
  if (!S.me) return;
  loadPeople().then(() => { loadHaiHistory(true); });
  if (!el('view-dashboard').hidden) loadDashboard();
  if (!el('view-calendar').hidden) renderCalendar();
  if (!el('view-tree').hidden) loadTree();
  if (!el('view-profile').hidden) renderProfile(S.profileView || S.me);
  if (S.consoleLoaded) cLoadAll();
}
function setMe(id, openLatest) {
  if (!id) return;
  S.me = id;
  rememberMe(id);
  S.profileView = id;
  S.sendGen = (S.sendGen || 0) + 1;
  S.peer = '';
  S.convo = 'hai';
  S.chatKey = S.family + ':' + id;
  S.haiRun = '';
  S.haiHistory = [];
  S.memCache = [];
  S.memFilter = '';
  newChat();
  el('convoLog').innerHTML = '';
  const sendBtn = el('sendBtn');
  sendBtn.disabled = false;
  sendBtn.textContent = 'Send';
  el('msg').value = '';
  autogrow();
  renderConvoList();
  openConvo('hai', true);
  updateRailMe();
  fillAcct();
  const mv = el('memView');
  if (mv && Array.from(mv.options).some(o => o.value === id)) mv.value = id;
  updateAgentHead();
  refreshImageUrls();
  loadHaiHistory(!!openLatest);
  el('runInspect').value = '';
  el('runOut').textContent = 'No run inspected.';
  el('runSteps').innerHTML = '';
  refresh();
  if (S.consoleLoaded) { cLoadMemories(); cLoadReminders(); }
  if (!el('view-dashboard').hidden) loadDashboard();
  if (!el('view-profile').hidden) renderProfile(S.profileView || S.me);
  if (!el('view-tree').hidden) loadTree();
  if (!el('view-calendar').hidden) renderCalendar();
}
function dashGoAgent(prompt) {
  showView('chat');
  subtab('agent');
  if (prompt) el('msg').value = prompt;
  el('msg').focus();
}
function dashAsk() { dashGoAgent(''); }
function dashPlan() {
  dashGoAgent('Please help me plan a family event. Ask me what we are celebrating, when it should happen, and who is coming.');
}
function dashMemory() {
  dashGoAgent('Please help me save a family memory. Ask me what happened, who was there, and when it happened — and confirm with me before saving anything.');
}
function dashEmpty(text) {
  const d = document.createElement('div');
  d.className = 'empty';
  d.textContent = text;
  return d;
}
function dashRenderIdentity() {
  const h = new Date().getHours();
  el('dashGreet').textContent = h < 12 ? 'Good morning' : (h < 18 ? 'Good afternoon' : 'Good evening');
  el('dashName').textContent = firstName(memberName(S.me)) || memberName(S.me);
  el('dashFamilyLine').innerHTML = '<span id="dashSecrets"></span>';
  const members = S.members || [];
  const row = el('dashAvatars');
  row.innerHTML = '';
  members.forEach(m => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'mini-member' + (m.id === S.me ? ' you' : '');
    b.innerHTML = '<span class="mini-avatar" style="background:' + avatarColor(m.id) + '">'
      + esc(initials(m.name)) + '</span><span>' + esc(firstName(m.name))
      + '</span><span class="mini-role">' + esc(roleLabel(m.role)) + '</span>';
    b.onclick = () => viewMember(m.id);
    row.appendChild(b);
  });
  if (!members.length) row.appendChild(dashEmpty('No family members loaded yet.'));
}
function secretsThisWeek(events) {
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const end = new Date(today.getTime() + 7 * 86400000);
  return (events || []).filter(it => {
    const d = parseISODate(it.date);
    return d && d >= today && d <= end && privacyInfoForItem(it).scope === 'restricted';
  }).length;
}
function heroPlan(title) {
  showView('chat');
  subtab('agent');
  el('msg').value = 'Help me plan ' + title;
  el('msg').focus();
  send();
}
async function heroRemind(ev, out) {
  const t = new Date();
  t.setDate(t.getDate() + 1);
  const iso = t.getFullYear() + '-' + String(t.getMonth() + 1).padStart(2, '0')
    + '-' + String(t.getDate()).padStart(2, '0');
  out.textContent = ' Saving…';
  const r = await api('propose_reminder', {family: S.family, by: S.me,
    key: 'event:' + ev.id + ':tomorrow', title: ev.title, date: iso,
    recipients: ev.recipients,
    message: 'Reminder: ' + ev.title + ' is on ' + fmtDate(ev.date) + '.'});
  if (r && !r.error) {
    out.textContent = ' Set for ' + fmtDate(iso) + '.';
    loadDashboard();
  } else {
    out.textContent = ' ' + ((r && r.error) || 'could not set a reminder');
  }
}
function renderDashHero(box, events, error) {
  box.innerHTML = '';
  if (error) { box.appendChild(dashEmpty(error)); return; }
  const next = (events || [])[0] || null;
  if (!next) { box.appendChild(dashEmpty('Nothing needs you right now.')); return; }
  const info = privacyInfoForItem(next);
  const title = document.createElement('div');
  title.innerHTML = '<b>' + esc(next.title) + '</b> ' + kindBadge(next) + ' ';
  title.appendChild(privacyChip(info));
  const date = document.createElement('div');
  date.className = 'hint';
  date.textContent = fmtDate(next.date);
  box.append(title, date);
  info.canSee.forEach(id => box.appendChild(pvPerson(id)));
  const row = document.createElement('div');
  row.className = 'row';
  const plan = document.createElement('button');
  plan.type = 'button';
  plan.textContent = 'Plan with hai';
  plan.onclick = () => heroPlan(next.title);
  const remind = document.createElement('button');
  remind.type = 'button';
  remind.className = 'ghost';
  remind.textContent = 'Remind me tomorrow';
  const out = document.createElement('span');
  out.className = 'hint';
  remind.onclick = () => heroRemind(next, out);
  row.append(plan, remind, out);
  box.appendChild(row);
}
async function loadDashboard() {
  el('dashName').textContent = memberName(S.me);
  ['dashNotes', 'dashApprovals', 'dashInsight', 'dashCelebrations'].forEach(id => {
    el(id).innerHTML = '<span class="hint">Loading…</span>';
  });
  const mr = await get('members', {family: S.family});
  if (!mr.error && mr.members) S.members = mr.members;
  updateRailMe();
  dashRenderIdentity();
  const r = await get('due_reminders', {family: S.family, member: S.me, days_ahead: '30'});
  const a = await get('approvals', {family: S.family, member: S.me});
  const ev = await get('family_events', {family: S.family, member: S.me});
  const items = r.items || [];
  const pending = a.approvals || [];
  const events = ((ev.events || []).slice().sort((x, y) => (x.date + x.title).localeCompare(y.date + y.title)));
  const line = el('dashFamilyLine');
  const secBox = el('dashSecrets');
  if (!ev.error) {
    const n = secretsThisWeek(events);
    if (!n) {
      line.hidden = true;
    } else {
      line.hidden = false;
      secBox.textContent = n === 1 ? 'hai is keeping 1 secret for you this week.'
        : 'hai is keeping ' + n + ' secrets for you this week.';
    }
  } else {
    line.hidden = true;
  }
  renderDashHero(el('dashInsight'), events, ev.error);
  const cel = el('dashCelebrations');
  cel.innerHTML = '';
  if (ev.error) cel.appendChild(dashEmpty(ev.error));
  else {
    events.filter(it => it.kind === 'birthday' || it.kind === 'anniversary').forEach(it => {
      cel.appendChild(reminderCard(it, false));
    });
    if (!cel.children.length) cel.appendChild(dashEmpty('No birthdays or anniversaries in the next 30 days.'));
  }
  const box = el('dashNotes');
  box.innerHTML = '';
  if (r.error) box.appendChild(dashEmpty(r.error));
  else {
    items.forEach(it => box.appendChild(reminderCard(it, false)));
    if (!box.children.length) box.appendChild(dashEmpty('Nothing due — all caught up!'));
  }
  const box2 = el('dashApprovals');
  box2.innerHTML = '';
  if (a.error) box2.appendChild(dashEmpty(a.error));
  else {
    pending.forEach(ap => {
      box2.appendChild(approvalCard(ap, {
        by: S.me,
        onDecide: (a, decision, by, out) => {
          out.textContent = 'Saving…';
          dashDecide(a.id, decision);
        },
      }));
    });
    if (!box2.children.length) box2.appendChild(dashEmpty('Nothing waiting for review.'));
  }
}
async function dashDecide(id, decision) {
  const r = await api('decide', {family: S.family, member: S.me, approval_id: id, by: S.me, decision: decision});
  if (r && r.error) el('dashApprovals').appendChild(dashEmpty(r.error));
  loadDashboard();
  refresh();
}
async function loadTree() {
  const r = await get('members', {family: S.family});
  const wrap = el('treeWrap');
  if (r.error) { wrap.textContent = r.error; return; }
  S.members = r.members;
  const byId = {};
  r.members.forEach(m => { byId[m.id] = m; });
  const depth = {};
  r.members.forEach(m => { depth[m.id] = 0; });
  let changed = true;
  while (changed) {
    changed = false;
    r.members.forEach(m => {
      (m.parents || []).forEach(p => {
        if (byId[p] && depth[m.id] <= depth[p]) { depth[m.id] = depth[p] + 1; changed = true; }
      });
    });
  }
  r.members.forEach(m => {
    const p = m.partner;
    if (p && byId[p]) {
      const d = Math.max(depth[m.id], depth[p]);
      depth[m.id] = d; depth[p] = d;
    }
  });
  const rows = {};
  r.members.forEach(m => { (rows[depth[m.id]] = rows[depth[m.id]] || []).push(m); });
  Object.values(rows).forEach(row => {
    row.sort((a, b) => r.members.indexOf(a) - r.members.indexOf(b));
    for (let i = 0; i < row.length; i++) {
      const j = row.findIndex(m => m.id === row[i].partner);
      if (j > i + 1) { const moved = row.splice(j, 1)[0]; row.splice(i + 1, 0, moved); }
    }
  });
  wrap.innerHTML = '<svg id="treeSvg"></svg>';
  Object.keys(rows).sort((a, b) => a - b).forEach(d => {
    const row = document.createElement('div');
    row.className = 'trow';
    rows[d].forEach(m => {
      const n = document.createElement('div');
      n.className = 'tnode' + (m.id === S.me ? ' you' : '');
      n.id = 'tn-' + m.id;
      n.innerHTML = '<div class="avatar" style="background:' + avatarColor(m.id) + '">'
        + esc(initials(m.name)) + '</div><div class="tname">' + esc(firstName(m.name))
        + '</div><div class="trole">' + esc(roleLabel(m.role)) + '</div>';
      n.onclick = () => viewMember(m.id);
      n.tabIndex = 0;
      n.setAttribute('role', 'button');
      n.setAttribute('aria-label', 'View ' + m.name);
      n.onkeydown = ev => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); viewMember(m.id); } };
      row.appendChild(n);
    });
    wrap.appendChild(row);
  });
  requestAnimationFrame(drawTreeLines);
}
function drawTreeLines() {
  const wrap = el('treeWrap');
  const svg = el('treeSvg');
  if (!wrap || !svg) return;
  const wb = wrap.getBoundingClientRect();
  svg.setAttribute('width', wrap.offsetWidth);
  svg.setAttribute('height', wrap.offsetHeight);
  const pos = id => {
    const n = document.getElementById('tn-' + id);
    if (!n) return null;
    const b = n.getBoundingClientRect();
    return {x: b.left - wb.left + b.width / 2, top: b.top - wb.top, bottom: b.bottom - wb.top};
  };
  const byId = {};
  (S.members || []).forEach(m => { byId[m.id] = m; });
  const done = new Set();
  let s = '';
  (S.members || []).forEach(m => {
    (m.parents || []).forEach(p => {
      if (!byId[p] || done.has(m.id + p)) return;
      done.add(m.id + p);
      const co = (byId[p].partner && (m.parents || []).includes(byId[p].partner)) ? byId[p].partner : null;
      const a = pos(p);
      const c = pos(m.id);
      if (!a || !c) return;
      const b2 = co ? pos(co) : null;
      const x0 = b2 ? (a.x + b2.x) / 2 : a.x;
      const midY = (a.bottom + c.top) / 2;
      s += '<path d="M' + x0 + ',' + a.bottom + ' V' + midY + ' H' + c.x + ' V' + c.top + '" fill="none" stroke="#E7DFD2" stroke-width="2"/>';
    });
    const p = m.partner;
    const key = 'co' + [m.id, p].sort().join();
    if (p && byId[p] && !done.has(key)) {
      done.add(key);
      const a = pos(m.id);
      const b2 = pos(p);
      if (a && b2 && Math.abs(a.top - b2.top) < 40) {
        s += '<line x1="' + a.x + '" y1="' + (a.top + 42) + '" x2="' + b2.x + '" y2="' + (b2.top + 42) + '" stroke="#E7DFD2" stroke-width="2"/>';
      }
    }
  });
  svg.innerHTML = s;
}
function viewMember(id) {
  S.profileView = id;
  showView('profile');
}
function renderProfile(id) {
  S.profileView = id || S.me;
  const m = (S.members || []).find(x => x.id === S.profileView) || {id: S.profileView, name: S.profileView};
  const me = S.me === S.profileView;
  const box = el('profBody');
  box.innerHTML = '';
  const av = document.createElement('div');
  av.className = 'profAvatar';
  av.style.background = avatarColor(m.id);
  av.textContent = initials(m.name || m.id);
  const h = document.createElement('h3');
  h.textContent = (m.name || m.id) + (me ? ' (you)' : '');
  const meta = document.createElement('div');
  meta.className = 'hint';
  meta.innerHTML = 'role: <span class="rolecap">' + esc(roleLabel(m.role)) + '</span><br>email: ' + esc(m.email || '-')
    + '<br>phone: ' + esc(m.phone || '-');
  const row = document.createElement('div');
  row.className = 'row';
  if (me) {
    const b = document.createElement('button');
    b.textContent = 'Open my agent chat';
    b.onclick = () => { showView('chat'); subtab('agent'); };
    row.appendChild(b);
  } else {
    const b = document.createElement('button');
    b.textContent = 'Message them';
    b.onclick = () => { showView('chat'); subtab('people'); S.peer = m.id; loadPeople().then(loadThread); };
    row.append(b);
  }
  box.append(av, h, meta, row);
  const ws = document.createElement('div');
  ws.className = 'prof-section';
  ws.innerHTML = '<h3>Family workspace</h3><div class="hint">Family: ' + esc(S.family)
    + '<br>Members: ' + (S.members || []).length
    + ' — ' + esc((S.members || []).map(x => firstName(x.name)).join(', ')) + '</div>'
    + '';
  const dates = document.createElement('div');
  dates.className = 'prof-section';
  dates.innerHTML = '<h3>Important dates</h3><div id="profDates"><span class="hint">Loading…</span></div>';
  const memSec = document.createElement('div');
  memSec.className = 'prof-section';
  memSec.innerHTML = '<h3>Memory &amp; privacy</h3><p class="hint">Only memories '
    + esc(firstName(m.name || m.id)) + ' is authorized to see — shared, their own private, and restricted shared with them.</p>'
    + '<div id="profMemories"><span class="hint">Loading…</span></div>';
  const notif = document.createElement('div');
  notif.className = 'prof-section';
  notif.innerHTML = '<h3>Upcoming notifications</h3><p class="hint">Notification preferences are not yet supported — these are '
    + esc(firstName(m.name || m.id)) + '&rsquo;s real upcoming notifications instead.</p>'
    + '<div id="profNotifs"><span class="hint">Loading…</span></div>';
  box.append(ws, dates);
  if (me) box.append(memSec, notif);
  loadProfileExtras(m, me);
}
function profileDateCard(it) {
  const cd = countdownInfo(it.date);
  const d = document.createElement('div');
  d.className = 'reminder-card';
  d.innerHTML = '<span class="pill ' + esc(it.status || 'upcoming') + '">'
    + esc((cd.text ? cd.text + ' · ' : '') + (it.status || 'upcoming')) + '</span> '
    + kindBadge(it) + ' <b>' + esc(it.title) + '</b> <span class="pvslot"></span>'
    + '<br><span class="hint">' + esc(fmtDate(it.date)) + '</span>';
  d.querySelector('.pvslot').replaceWith(privacyChip(privacyInfoForItem(it)));
  return d;
}
async function loadProfileExtras(m, me) {
  const datesBox = el('profDates');
  if (!datesBox) return;
  const memBox = el('profMemories'), notifBox = el('profNotifs');
  const selfId = me ? m.id : S.me;
  const [mr, rr] = await Promise.all([
    get('memories', {family: S.family, member: selfId}),
    get('due_reminders', {family: S.family, member: selfId, days_ahead: '90'}),
  ]);
  if (S.profileView !== m.id) return;
  // Important dates: only real event-dated memories + real reminders.
  // On another member's profile, only items involving them, requested as S.me.
  const involves = x => me || (x.people || []).some(p => p === m.id || p === (m.name || ''));
  const involvesReminder = it => me || (it.recipients || []).includes(m.id);
  datesBox.innerHTML = '';
  const seen = new Set();
  if (rr.error) datesBox.appendChild(dashEmpty(rr.error));
  (rr.items || []).filter(involvesReminder).forEach(it => {
    seen.add(it.date + '|' + it.title);
    datesBox.appendChild(me ? eventCard(it, false) : profileDateCard(it));
  });
  if (!mr.error) (mr.memories || []).filter(x => x.event_date).filter(involves).forEach(x => {
    const key = x.event_date + '|' + x.text;
    if (seen.has(key)) return;
    seen.add(key);
    const d = document.createElement('div');
    d.className = 'event-card';
    if (me) {
      d.innerHTML = '<b>' + esc(fmtDate(x.event_date)) + '</b><br>' + esc(x.text)
        + '<div class="meta">owner ' + esc(memberName(x.owner))
        + (x.people && x.people.length ? ' &middot; about: ' + esc(humanNames(x.people)) : '') + '</div>';
    } else {
      d.innerHTML = '<b>' + esc(fmtDate(x.event_date)) + '</b><br>' + esc(x.text);
    }
    d.insertBefore(privacyChip(privacyInfoForMemory(x)), d.firstChild);
    datesBox.appendChild(d);
  });
  if (!datesBox.children.length) datesBox.appendChild(dashEmpty('No important dates found for ' + firstName(m.name || m.id) + '.'));
  if (!memBox || !notifBox) return;
  // Memory & privacy: the memories API already filters to what m may see.
  memBox.innerHTML = '';
  if (mr.error) memBox.appendChild(dashEmpty(mr.error));
  else {
    (mr.memories || []).forEach(x => {
      const d = document.createElement('div');
      d.className = 'mem';
      d.innerHTML = esc(x.text) + '<div class="meta">owner ' + esc(memberName(x.owner))
        + (x.people && x.people.length ? ' &middot; about: ' + esc(humanNames(x.people)) : '')
        + (x.event_date ? ' &middot; ' + esc(x.event_date) : '') + '</div>';
      d.insertBefore(privacyChip(privacyInfoForMemory(x)), d.firstChild);
      memBox.appendChild(d);
    });
    if (!memBox.children.length) memBox.appendChild(dashEmpty('No memories visible to ' + firstName(m.name || m.id) + '.'));
  }
  // Upcoming notifications (real reminders), not invented preferences.
  notifBox.innerHTML = '';
  if (rr.error) notifBox.appendChild(dashEmpty(rr.error));
  else {
    (rr.items || []).forEach(it => notifBox.appendChild(eventCard(it, false)));
    if (!notifBox.children.length) notifBox.appendChild(dashEmpty('No upcoming notifications for ' + firstName(m.name || m.id) + '.'));
  }
}
function toggleMemberForm(show) {
  const f = el('memberForm');
  f.hidden = show === undefined ? f.hidden : !show;
  if (!f.hidden) fillMemberForm();
}
function fillMemberForm() {
  const opts = [['', '(none)']].concat((S.members || []).map(m => [m.id, m.name + ' (' + m.id + ')']));
  ['fPartner', 'fParent1', 'fParent2'].forEach(id => {
    const sel = el(id);
    sel.innerHTML = '';
    opts.forEach(pair => {
      const o = document.createElement('option');
      o.value = pair[0];
      o.textContent = pair[1];
      sel.appendChild(o);
    });
  });
}
async function submitMember() {
  const parents = [el('fParent1').value, el('fParent2').value].filter(Boolean);
  const r = await api('add_member', {family: S.family, name: el('fName').value,
    role: el('fRole').value, email: el('fEmail').value, phone: el('fPhone').value,
    partner: el('fPartner').value, parents: parents});
  el('formOut').textContent = r.error ? r.error : 'Added ' + r.member.id;
  if (!r.error) {
    el('fName').value = ''; el('fEmail').value = ''; el('fPhone').value = '';
    loadTree();
    loadPeople();
    toggleMemberForm(false);
  }
}
let calYear, calMonth, calSelected = null;
function calISO(y, m, d) {
  return y + '-' + String(m + 1).padStart(2, '0') + '-' + String(d).padStart(2, '0');
}
function calMove(d) {
  calMonth += d;
  if (calMonth < 0) { calMonth = 11; calYear--; }
  if (calMonth > 11) { calMonth = 0; calYear++; }
  renderCalendar();
}
async function renderCalendar() {
  const today = new Date();
  if (calYear === undefined) {
    calYear = today.getFullYear(); calMonth = today.getMonth();
    calSelected = calISO(calYear, calMonth, today.getDate());
  }
  el('calLabel').textContent = new Date(calYear, calMonth, 1)
    .toLocaleString(undefined, {month: 'long', year: 'numeric'});
  const r = await get('due_reminders', {family: S.family, member: S.me,
    ref_date: calISO(calYear, calMonth, 1), days_ahead: '31'});
  const byDate = {};
  (r.items || []).forEach(it => { (byDate[it.date] = byDate[it.date] || []).push(it); });
  S.calItems = byDate;
  const grid = el('calGrid');
  grid.innerHTML = '';
  ['Su', 'Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa'].forEach(d => {
    const h = document.createElement('div');
    h.className = 'cal-dow';
    h.textContent = d;
    grid.appendChild(h);
  });
  const firstDow = new Date(calYear, calMonth, 1).getDay();
  const days = new Date(calYear, calMonth + 1, 0).getDate();
  const todayStr = calISO(today.getFullYear(), today.getMonth(), today.getDate());
  for (let i = 0; i < firstDow; i++) grid.appendChild(document.createElement('div'));
  for (let d = 1; d <= days; d++) {
    const ds = calISO(calYear, calMonth, d);
    const b = document.createElement('button');
    b.className = 'day' + (ds === todayStr ? ' today' : '') + (ds === calSelected ? ' selected' : '');
    const dayItems = byDate[ds] || [];
    const dots = dayItems.map(it => '<span class="dot ' + esc(it.kind || 'event') + '"></span>').join('');
    b.innerHTML = d + (dots ? '<span class="dots">' + dots + '</span>' : '<span class="dots"></span>');
    b.title = dayItems.map(it => it.title).join(', ');
    b.onclick = () => { calSelected = ds; renderCalendar(); };
    grid.appendChild(b);
  }
  renderCalUpcoming(r, todayStr);
  renderCalDay();
}
function kindBadge(it) {
  return '<span class="kind-badge ' + esc(it.kind || 'event') + '">' + esc(kindName(it.kind || 'event')) + '</span>';
}
function eventCard(it, showAsk) {
  return reminderCard(it, showAsk);
}
function renderCalUpcoming(r, todayStr) {
  const box = el('calUpcoming');
  if (!box) return;
  box.innerHTML = '';
  if (r.error) { box.appendChild(dashEmpty(r.error)); return; }
  const monthStart = calISO(calYear, calMonth, 1);
  const threshold = todayStr < monthStart ? monthStart : todayStr;
  const items = (r.items || []).filter(it => it.date >= threshold)
    .sort((a, b) => (a.date + a.title).localeCompare(b.date + b.title)).slice(0, 8);
  items.forEach(it => box.appendChild(eventCard(it, true)));
  if (!box.children.length) box.appendChild(dashEmpty('No upcoming events in this month view.'));
}
function calAskHai(it) {
  showView('chat');
  subtab('agent');
  el('msg').value = 'Please help me plan for "' + it.title + '" on ' + fmtDate(it.date)
    + ' (' + kindName(it.kind || 'event') + '). ' + (it.suggested_message || '')
    + ' Ask me anything you need before suggesting next steps.';
  el('msg').focus();
}
function renderCalDay() {
  el('calDayLabel').textContent = calSelected ? fmtDate(calSelected) : '…';
  const box = el('calDayList');
  box.innerHTML = '';
  const items = (S.calItems || {})[calSelected] || [];
  items.forEach(it => box.appendChild(eventCard(it, true)));
  if (!box.children.length) box.appendChild(dashEmpty('Nothing for this day.'));
}
showView('signin');
subtab('people');
bootMembers();
window.addEventListener('resize', () => { if (!el('view-tree').hidden) drawTreeLines(); });
setInterval(() => {
  if (!el('view-chat').hidden && S.convo !== 'hai') loadThread();
}, 4000);
</script></body></html>
"""


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args(argv)
    try:
        seed_all(ROOT)
    except Exception as error:
        print(f"seed skipped: {error}", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Hai Family Agent console: http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
