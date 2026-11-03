"""A run on disk: the console shows it turn by turn, and a stopped run resumes from it.

    run_id = new_run(root, instructions=..., objective=...)
    messages = [{"role": "user", "content": objective}]
    # each turn: response = client.responses.create(...); messages += response.output;
    #            result = tools.run(...) for the tool the model picked; append its function_call_output
    save_progress(root, run_id, messages, response)                     # end of every turn
    save_progress(root, run_id, messages, response, status="answered")  # the turn that ends the run

    progress = load_progress(root, run_id)  # progress["messages"], ["status"], ["turn"]
    step = resume(root, run_id)             # continue a stopped run: step["message"] for the model, or None
    new_id = branch(client, root, run_id)   # a new run with verification on, from the run's checklist

Everything goes in workspace/runs/<run-id>/progress.json.
"""

from __future__ import annotations

import datetime
import json
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .settings import model, verification_enabled

if TYPE_CHECKING:
    from openai import OpenAI

CONTROL = "workspace/control.json"

# Parallel tool calls (e.g. document analysis) record steps from threads;
# serialize their read-append-write of progress.json.
_STEP_LOCK = threading.Lock()


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def new_run(
    root: Path,
    *,
    instructions: str,
    objective: str,
    verification: bool | None = None,
    model_name: str | None = None,
) -> str:
    """Start a run: a new folder under workspace/runs/. Returns its id.

    verification defaults to VERIFICATION in .env; it is fixed for the run's lifetime.
    model_name defaults to the fallback model(); task-specific loops pass theirs.
    """
    if verification is None:
        verification = verification_enabled()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"run-{stamp}-{uuid.uuid4().hex[:6]}"
    folder = _folder(root, run_id)
    folder.mkdir(parents=True)
    (folder / "instructions.md").write_text(instructions)
    (folder / "objective.md").write_text(objective)
    _write(
        root,
        run_id,
        {
            "run_id": run_id,
            "verification": verification,
            "model": model_name or model(),
            "started": now(),
            "status": "running",
            "turn": 0,
            "messages": [],
            "turns": [],
            "steps": [],
            "notes": {},
        },
    )
    print(
        f"Run {run_id} started · verification {'on' if verification else 'off'} · model {model_name or model()}",
        flush=True,
    )
    return run_id


def save_progress(
    root: Path,
    run_id: str,
    messages: list,
    response: Any = None,
    *,
    status: str = "running",
    model_name: str = "",
    duration_ms: float | None = None,
    step_status: str = "",
    **notes: Any,
) -> None:
    """End of a turn: save the conversation and the turn's tokens, print one line.

    `status` is "running" until the run stops (answered, paused, needs_input, ...).
    Without `response`, only the status changes (no model call this time).
    Anything in `notes` comes back from load_progress.
    """
    progress = load_progress(root, run_id)
    plain = [_plain(m) for m in messages]
    entry: dict[str, Any] = {"time": now(), "status": status, "end": len(plain)}
    if response is not None:
        output = [_plain(item) for item in response.output]
        before = progress["turns"][-1]["end"] if progress["turns"] else 0
        start = next(
            (i for i in range(before, len(plain)) if output and plain[i] == output[0]),
            before,
        )
        usage = response.usage
        progress["turn"] += 1
        entry.update(
            turn=progress["turn"],
            saw=start,
            output=start + len(output),
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )
        if model_name:
            entry["model"] = model_name
            entry["model_label"] = model_label(model_name)
        if duration_ms is not None:
            entry["duration_ms"] = round(float(duration_ms), 1)
        if step_status:
            entry["step_status"] = step_status
    progress["turns"].append(entry)
    progress["notes"].update(notes)
    progress.update(messages=plain, status=status)
    _write(root, run_id, progress)
    print(_line(root, progress, entry), flush=True)
    if status == "running":
        _wait_for_step(root)


def load_progress(root: Path, run_id: str) -> dict:
    """A run's saved progress: messages, status, turn, verification, notes."""
    path = _folder(root, run_id) / "progress.json"
    if not path.is_file():
        raise FileNotFoundError(f"no such run: {run_id}")
    return json.loads(path.read_text())


_LABELS = {
    "muse-wonder-1.0": "Muse Wonder",
    "muse-spark-1.3": "Muse Spark",
    "muse-spark-1.2": "Muse Spark",
    "muse-spark-1.1": "Muse Spark",
    "muse-image-1.0": "Muse Image",
    "muse-voice-transcribe-1.0": "Muse Voice Transcribe",
    "sam-3.1": "SAM 3.1",
}


def model_label(model_id: str) -> str:
    """A display name for a model id: muse-wonder-1.0 -> Muse Wonder."""
    if model_id in _LABELS:
        return _LABELS[model_id]
    parts = [p for p in str(model_id or "").split("-") if p]
    while parts and any(ch.isdigit() for ch in parts[-1]) and "." in parts[-1]:
        parts.pop()
    return " ".join(p.capitalize() for p in parts) or str(model_id or "")


def record_step(
    root: Path,
    run_id: str,
    *,
    task: str,
    model: str,
    duration_ms: float,
    status: str = "ok",
    summary: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
    started: str = "",
) -> dict | None:
    """Append one model call to the run's steps in progress.json.

    Every model call records its own step: number, task, model, start time,
    duration, tokens, status (ok / fallback / error), one-line summary.
    Returns the step, or None when there is no run to attach it to.
    """
    if not run_id:
        return None
    with _STEP_LOCK:
        try:
            progress = load_progress(root, run_id)
        except (OSError, ValueError):
            return None
        steps = progress.setdefault("steps", [])
        entry = {
            "step": len(steps) + 1,
            "task": task,
            "model": model,
            "model_label": model_label(model),
            "started": started or now(),
            "duration_ms": round(float(duration_ms), 1),
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "status": status,
            "summary": (summary or "")[:200],
        }
        steps.append(entry)
        progress["steps"] = steps
        _write(root, run_id, progress)
        return entry


def start_step(
    root: Path, run_id: str, *, task: str, model: str, summary: str = ""
) -> dict | None:
    """Append a step in status "running" so live readers see it immediately."""
    if not run_id:
        return None
    with _STEP_LOCK:
        try:
            progress = load_progress(root, run_id)
        except (OSError, ValueError):
            return None
        steps = progress.setdefault("steps", [])
        entry = {
            "step": len(steps) + 1,
            "task": task,
            "model": model,
            "model_label": model_label(model),
            "started": now(),
            "duration_ms": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "status": "running",
            "summary": (summary or "")[:200],
        }
        steps.append(entry)
        progress["steps"] = steps
        _write(root, run_id, progress)
        return entry


def finish_step(
    root: Path,
    run_id: str,
    step_number: int,
    *,
    duration_ms: float,
    status: str = "ok",
    summary: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> dict | None:
    """Close a running step: duration, status, summary, and token counts."""
    if not run_id or not step_number:
        return None
    with _STEP_LOCK:
        try:
            progress = load_progress(root, run_id)
        except (OSError, ValueError):
            return None
        for entry in progress.get("steps", []):
            if entry.get("step") == step_number:
                entry.update(
                    duration_ms=round(float(duration_ms), 1),
                    status=status,
                    summary=(summary or entry.get("summary", ""))[:200],
                    input_tokens=int(input_tokens or 0),
                    output_tokens=int(output_tokens or 0),
                )
                _write(root, run_id, progress)
                return entry
        return None


def resume(root: Path, run_id: str) -> dict:
    """Continue a stopped run: apply what a person did since it stopped.

    Returns {"message": what to tell the model next, or None when there is nothing to
    continue, "status": the run's status, "note": why not}. Approving the memo ends the
    run as done.
    """
    from . import document_text
    from .case_record import CaseRecord

    progress = load_progress(root, run_id)
    status = progress["status"]
    case = CaseRecord(Path(root), run_id)
    if status == "paused":
        if progress["notes"].get("branched_from") and not progress["turn"]:
            return _go(BRANCH_MESSAGE)
        return _go("You have more turns. Continue, and finish with submit_memo.")
    if status == "needs_input":
        requests = case.data["requests"]
        if not requests:
            return _stop(status, "no request to answer")
        request = requests[-1]
        visible = [d["id"] for d in document_text.list_documents(Path(root))]
        fresh = [d for d in visible if d not in request["visible_ids"]]
        if not fresh and request["status"] != "answered":
            return _stop(
                status, "nothing new: release evidence or answer the request first"
            )
        if request["status"] == "open":
            request = case.answer_request(
                request["id"], f"Released {len(fresh)} new document(s)."
            )
        message = f"A person answered your request {request['id']} ({request['item']}): {request['response']}"
        if fresh:
            message += f"\nReleased documents: {', '.join(fresh)}"
        return _go(message + "\nContinue the investigation.")
    if status == "awaiting_signoff":
        review = (case.data.get("memo") or {}).get("review")
        if not review:
            return _stop(
                status, "waiting for the attorney to approve or reject the memo"
            )
        if review["decision"] == "approved":
            save_progress(root, run_id, progress["messages"], status="done")
            return _stop("done", "the attorney approved the memo")
        return _go(
            f"The attorney rejected the memo: {review['notes']}. "
            "Address the notes and submit a revised memo."
        )
    return _stop(status, f"run is {status}; nothing to resume")


BRANCH_MESSAGE = (
    "This run continues from your checklist with the verification layer on. "
    "Fix what the check says, then plan, investigate, and file your memo."
)


def branch(client: OpenAI, root: Path, run_id: str) -> str:
    """Start a new run with verification on that continues `run_id` from its first set_checklist call.

    The new run keeps the conversation up to that call, runs the call again (so the
    checklist check refuses a weak checklist with feedback), and stops as "paused":
    resume it to let the agent continue. Returns the new run's id.
    """
    from . import tools

    root = Path(root)
    messages = load_progress(root, run_id)["messages"]
    index = next(
        (
            i
            for i, m in enumerate(messages)
            if m.get("type") == "function_call" and m.get("name") == "set_checklist"
        ),
        None,
    )
    if index is None:
        raise RuntimeError(f"{run_id} never called set_checklist")
    folder = _folder(root, run_id)
    new_id = new_run(
        root,
        instructions=(folder / "instructions.md").read_text(),
        objective=(folder / "objective.md").read_text(),
        verification=True,
    )
    call = messages[index]
    result = tools.run(client, root, new_id, call)
    if (
        result.get("status") == "paused"
    ):  # a check that is not built yet: nothing to continue
        shutil.rmtree(_folder(root, new_id), ignore_errors=True)
        (root / "workspace/case_records" / f"{new_id}.json").unlink(missing_ok=True)
        raise RuntimeError(result["error"])
    output = {
        "type": "function_call_output",
        "call_id": call.get("call_id"),
        "output": json.dumps(result),
    }
    save_progress(
        root,
        new_id,
        messages[: index + 1] + [output],
        status="paused",
        branched_from=run_id,
    )
    return new_id


# ---- helpers ------------------------------------------------------------------------


def _go(message: str) -> dict:
    return {"message": message, "status": "running", "note": ""}


def _stop(status: str, note: str) -> dict:
    return {"message": None, "status": status, "note": note}


def _folder(root: Path, run_id: str) -> Path:
    return Path(root) / "workspace/runs" / run_id


def _write(root: Path, run_id: str, progress: dict) -> None:
    path = _folder(root, run_id) / "progress.json"
    tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(progress, indent=1))
    tmp.replace(path)


def _plain(value: Any) -> Any:
    """SDK objects (such as the items of response.output) as plain JSON values."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _line(root: Path, progress: dict, entry: dict) -> str:
    """The terminal line for one saved turn."""
    run_id, status = progress["run_id"], entry["status"]
    if "turn" not in entry:
        return f"Run {run_id}: {status}"
    messages = progress["messages"]
    call = next(
        (
            m
            for m in messages[entry["saw"] : entry["output"]]
            if m.get("type") == "function_call"
        ),
        None,
    )
    if call is None:
        did = "no tool call (the model answered in text)"
    else:
        output = next(
            (
                m.get("output")
                for m in messages[entry["output"] : entry["end"]]
                if m.get("type") == "function_call_output"
                and m.get("call_id") == call.get("call_id")
            ),
            "",
        )
        did = f"{call.get('name')} → {_summary(output)}"
    line = f"Turn {entry['turn']}: {did}"
    if entry.get("model"):
        line += f" · {entry.get('step_status') or 'ok'} · {entry.get('model_label') or model_label(entry['model'])}"
        if entry.get("duration_ms") is not None:
            line += f" · {entry['duration_ms'] / 1000:.1f}s"
    line += f" · facts {_facts(root, run_id)} · {status}"
    if status != "running":
        line += f"\nRun {run_id} stopped: {status} after {progress['turn']} turns"
    return line


def _summary(output: Any) -> str:  # noqa: C901
    try:
        result = json.loads(output) if isinstance(output, str) else output
    except ValueError:
        return str(output)[:80]
    if not isinstance(result, dict):
        return str(result)[:80]
    if "error" in result:
        return "refused: " + str(result["error"])[:80]
    documents = result.get("documents")
    if documents and isinstance(documents[0], dict) and "evidence" in documents[0]:
        records = [e for d in documents for e in d["evidence"]]
        return f"{len(records)} quotes, {sum(e.get('verified', False) for e in records)} verified"
    for key, text in (
        ("documents", "{n} documents"),
        ("checklist", "checklist set ({n} items)"),
        ("plan", "plan saved ({n} steps)"),
    ):
        if key in result:
            return text.format(n=len(result[key]))
    if "recorded" in result:
        return f"fact {result['recorded'].get('id')} recorded"
    if result.get("escalated"):
        return "escalated to a person"
    if result.get("filed"):
        return f"memo filed ({result.get('memo')})"
    if "verdict" in result:
        return f"signature {result['verdict']}"
    if "flagged" in result:
        return "flagged confidential"
    return "ok"


def _facts(root: Path, run_id: str) -> int:
    record = Path(root) / "workspace/case_records" / f"{run_id}.json"
    try:
        return len(json.loads(record.read_text()).get("facts", []))
    except (OSError, ValueError):
        return 0


def _wait_for_step(root: Path) -> None:
    """Step mode (ticked in the console): wait for 'Next turn' before the next turn."""
    path = Path(root) / CONTROL
    announced = False
    while True:
        try:
            control = json.loads(path.read_text())
        except (OSError, ValueError):
            return
        if not control.get("step"):
            if control.get("waiting"):
                _write_control(path, dict(control, waiting=False))
            return
        if control.get("credits", 0) > 0:
            _write_control(
                path, dict(control, credits=control["credits"] - 1, waiting=False)
            )
            return
        if not announced:
            _write_control(path, dict(control, waiting=True))
            print("Step mode: waiting for 'Next turn' in the console", flush=True)
            announced = True
        time.sleep(0.3)


def _write_control(path: Path, control: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(control))
    tmp.replace(path)
