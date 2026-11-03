"""The Hai Family Agent: the agentic loop for one member's conversation.

run_agent starts a conversation; resume_run continues one that stopped. Each
turn the model either calls exactly one family tool or replies to the member in
plain text. A text reply stops the run as awaiting_member until the member
writes back; a proposed action stops it as needs_input until a person reviews
it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from openai import OpenAI

from agent_kit import progress
from agent_kit.settings import agent_model, fast_model

from . import tools
from .store import FamilyStore

LAST_TURN_MESSAGE = (
    "This is your last turn. Reply to the member in plain text with what you have."
)
MORE_TURNS_MESSAGE = "You have more turns. Continue."


def _instructions() -> str:
    return Path(__file__).with_name("system_instructions.md").read_text()


def _identity(root: Path, run_id: str) -> dict:
    path = root / "workspace/runs" / run_id / "family.json"
    if not path.is_file():
        raise ValueError(f"{run_id} is not a family run")
    return json.loads(path.read_text())


def _field(call, key: str):
    return call.get(key) if isinstance(call, dict) else getattr(call, key, None)


def _item_type(item) -> str | None:
    return item.get("type") if isinstance(item, dict) else getattr(item, "type", None)


def _find_call(output) -> object | None:
    for item in output or []:
        if _item_type(item) == "function_call":
            return item
    return None


def _reply_text(output) -> str:
    """The assistant's plain-text reply from a model's output items."""
    parts = []
    for item in output or []:
        if _item_type(item) != "message":
            continue
        content = (
            item.get("content")
            if isinstance(item, dict)
            else getattr(item, "content", [])
        )
        for block in content or []:
            text = (
                block.get("text")
                if isinstance(block, dict)
                else getattr(block, "text", "")
            )
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def _steps_since(root: Path, run_id: str, base: int) -> list:
    try:
        return progress.load_progress(root, run_id).get("steps", [])[base:]
    except Exception:
        return []


def _run_loop(
    *,
    root: Path,
    client: OpenAI,
    run_id: str,
    messages: list,
    instructions: str,
    turns: int,
) -> dict:
    try:
        steps_base = len(progress.load_progress(root, run_id).get("steps", []))
    except Exception:
        steps_base = 0
    for index in range(max(0, turns)):
        if index == turns - 1:
            messages.append({"role": "user", "content": LAST_TURN_MESSAGE})
        primary, fallback = agent_model(), fast_model()
        used_model, fallback_from = primary, ""
        step_entry = progress.start_step(
            root, run_id, task="chat", model=primary, summary="Thinking…"
        )
        step_no = step_entry["step"] if step_entry else 0
        t0 = time.perf_counter()
        try:
            try:
                response = client.responses.create(
                    model=primary,
                    instructions=instructions,
                    input=messages,
                    tools=tools.definitions(root, run_id),
                )
            except Exception as first_error:
                if fallback == primary:
                    raise
                progress.finish_step(
                    root, run_id, step_no,
                    duration_ms=(time.perf_counter() - t0) * 1000,
                    status="error",
                    summary=f"{primary} failed ({first_error}); fell back to {fallback}",
                )
                print(
                    f"Run {run_id}: {primary} failed or timed out; "
                    f"falling back to {fallback}",
                    flush=True,
                )
                used_model, fallback_from = fallback, primary
                step_entry = progress.start_step(
                    root, run_id, task="chat", model=fallback, summary="Thinking…"
                )
                step_no = step_entry["step"] if step_entry else 0
                t0 = time.perf_counter()
                response = client.responses.create(
                    model=fallback,
                    instructions=instructions,
                    input=messages,
                    tools=tools.definitions(root, run_id),
                )
        except Exception as error:
            progress.finish_step(
                root, run_id, step_no,
                duration_ms=(time.perf_counter() - t0) * 1000,
                status="error",
                summary=f"{used_model} failed: {error}",
            )
            progress.save_progress(root, run_id, messages, status="paused")
            current = progress.load_progress(root, run_id)
            return {
                "run_id": run_id,
                "status": "paused",
                "turn": current["turn"],
                "note": str(error) or type(error).__name__,
                "steps": _steps_since(root, run_id, steps_base),
            }
        duration_ms = (time.perf_counter() - t0) * 1000
        output = list(response.output or [])
        messages.extend(output)
        call = _find_call(output)
        notes = {"model_used": used_model}
        if fallback_from:
            notes["model_fallback"] = f"{fallback_from} -> {used_model}"
        step_status = "fallback" if fallback_from else "ok"
        usage = getattr(response, "usage", None)
        progress.finish_step(
            root, run_id, step_no,
            duration_ms=duration_ms, status=step_status,
            summary=f"called {_field(call, 'name')}" if call else "replied in text",
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )
        if call is None:
            reply = _reply_text(output)
            progress.save_progress(root, run_id, messages, response, status="awaiting_member",
                                   model_name=used_model, duration_ms=duration_ms,
                                   step_status=step_status, **notes)
            current = progress.load_progress(root, run_id)
            return {
                "run_id": run_id,
                "status": "awaiting_member",
                "turn": current["turn"],
                "reply": reply,
                "steps": _steps_since(root, run_id, steps_base),
            }
        result = tools.run(client, root, run_id, call)
        messages.append(
            {
                "type": "function_call_output",
                "call_id": _field(call, "call_id"),
                "output": json.dumps(result),
            }
        )
        status = (
            result.get("status", "running") if isinstance(result, dict) else "running"
        )
        progress.save_progress(root, run_id, messages, response, status=status,
                               model_name=used_model, duration_ms=duration_ms,
                               step_status=step_status, **notes)
        if status != "running":
            current = progress.load_progress(root, run_id)
            outcome: dict = {
                "run_id": run_id,
                "status": status,
                "turn": current["turn"],
                "steps": _steps_since(root, run_id, steps_base),
            }
            if isinstance(result, dict) and result.get("approval"):
                outcome["approval"] = result["approval"].get("id")
            return outcome
    current = progress.load_progress(root, run_id)
    if current["status"] == "running":
        progress.save_progress(root, run_id, messages, status="paused")
        current = progress.load_progress(root, run_id)
    return {
        "run_id": run_id,
        "status": current["status"],
        "turn": current["turn"],
        "steps": _steps_since(root, run_id, steps_base),
    }


def run_agent(
    *,
    root: Path,
    client: OpenAI,
    family_id: str,
    member_id: str,
    message: str,
    turns: int = 10,
) -> dict:
    root = Path(root)
    store = FamilyStore(root, family_id)
    member = store.require_member(member_id)
    if not (message or "").strip():
        raise ValueError("the member's message is empty")
    instructions = _instructions()
    preferred = (member.get("preferred_language") or "English").strip() or "English"
    if preferred.lower() != "english":
        instructions += (
            f"\n\nThe member you are talking to prefers {preferred}. "
            f"Reply in {preferred} unless they ask for another language."
        )
    objective = (
        f"Family conversation with {member['name']} ({member_id}) "
        f"of family {family_id}."
    )
    run_id = progress.new_run(
        root, instructions=instructions, objective=objective, model_name=agent_model()
    )
    (root / "workspace/runs" / run_id / "family.json").write_text(
        json.dumps({"family_id": family_id, "member_id": member_id})
    )
    messages: list = [
        {
            "role": "user",
            "content": (
                f"[Talking to {member['name']} ({member_id}), family {family_id}]\n\n"
                f"{message.strip()}"
            ),
        }
    ]
    return _run_loop(
        root=root,
        client=client,
        run_id=run_id,
        messages=messages,
        instructions=instructions,
        turns=turns,
    )


def resume_run(
    *, root: Path, run_id: str, client: OpenAI, message: str = "", turns: int = 10
) -> dict:
    root = Path(root)
    identity = _identity(root, run_id)
    store = FamilyStore(root, identity["family_id"])
    saved = progress.load_progress(root, run_id)
    status = saved["status"]
    if status == "awaiting_member":
        if not (message or "").strip():
            return {
                "run_id": run_id,
                "status": status,
                "turn": saved["turn"],
                "note": "waiting for the member's next message",
            }
        follow_up = message.strip()
    elif status == "needs_input":
        pending = store.pending_approvals(run_id)
        if pending:
            return {
                "run_id": run_id,
                "status": status,
                "turn": saved["turn"],
                "note": f"waiting for review of {', '.join(a['id'] for a in pending)}",
            }
        decided = [
            a
            for a in store.approvals_for_run(run_id)
            if a["status"] in ("approved", "rejected")
        ]
        if not decided:
            return {
                "run_id": run_id,
                "status": status,
                "turn": saved["turn"],
                "note": "nothing to resume",
            }
        latest = decided[-1]
        if latest["status"] == "approved":
            follow_up = (
                f"A family member approved your {latest['action']} proposal "
                f"({latest['id']}). Tell the member what was approved. "
            )
            if latest.get("granted"):
                grants = "; ".join(
                    f"{g['memory']} is now shared with {', '.join(g['members'])}"
                    for g in latest["granted"]
                )
                follow_up += f"Access granted: {grants}. "
            elif latest.get("delivered"):
                follow_up += (
                    f"Invitations were sent to "
                    f"{', '.join(latest.get('recipients', []))}. "
                    f"Tell the member who received them. "
                )
            else:
                follow_up += (
                    "The prototype has no delivery mechanism: demonstrate the "
                    "approved message or action, do not claim it was sent. "
                )
        else:
            follow_up = (
                f"A family member rejected your {latest['action']} proposal "
                f"({latest['id']}): {latest['decision_note'] or 'no reason given'}. "
                "Tell the member, and offer an alternative."
            )
    elif status == "paused":
        follow_up = MORE_TURNS_MESSAGE
    else:
        result: dict = {
            "run_id": run_id,
            "status": status,
            "turn": saved["turn"],
        }
        if status == "running":
            result["note"] = "run is still running"
        else:
            result["note"] = f"run is {status}; nothing to resume"
        return result
    instructions_path = root / "workspace/runs" / run_id / "instructions.md"
    instructions = (
        instructions_path.read_text() if instructions_path.is_file() else _instructions()
    )
    messages: list = list(saved["messages"])
    messages.append({"role": "user", "content": follow_up})
    return _run_loop(
        root=root,
        client=client,
        run_id=run_id,
        messages=messages,
        instructions=instructions,
        turns=turns,
    )
