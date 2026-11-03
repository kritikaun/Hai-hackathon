"""The Legal Agent: the agentic loop.

run_agent works the case; resume_run continues a run that stopped.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from openai import OpenAI

from agent_kit import progress, tools
from agent_kit.settings import case_dir, model

LAST_TURN_MESSAGE = (
    "This is your last turn. File the memo now with submit_memo "
    "using what you have."
)
NO_TOOL_MESSAGE = (
    "Call exactly one tool per turn. Never answer in plain text. "
    "Continue with the next tool call."
)


def _instructions() -> str:
    return Path(__file__).with_name("system_instructions.md").read_text()


def _objective(root: Path) -> str:
    return (case_dir(root) / "objective.md").read_text()


def _field(call, key: str):
    return call.get(key) if isinstance(call, dict) else getattr(call, key, None)


def _find_call(output) -> object | None:
    for item in output or []:
        item_type = (
            item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        )
        if item_type == "function_call":
            return item
    return None


def _run_loop(
    *,
    root: Path,
    client: OpenAI,
    run_id: str,
    messages: list,
    instructions: str,
    turns: int,
) -> dict:
    for index in range(max(0, turns)):
        if index == turns - 1:
            messages.append({"role": "user", "content": LAST_TURN_MESSAGE})
        step_entry = progress.start_step(
            root, run_id, task="agent_turn", model=model(), summary="Thinking…"
        )
        step_no = step_entry["step"] if step_entry else 0
        t0 = time.perf_counter()
        try:
            response = client.responses.create(
                model=model(),
                instructions=instructions,
                input=messages,
                tools=tools.definitions(root, run_id),
            )
        except Exception as error:
            progress.finish_step(
                root, run_id, step_no,
                duration_ms=(time.perf_counter() - t0) * 1000,
                status="error",
                summary=f"{model()} failed: {error}",
            )
            progress.save_progress(root, run_id, messages, status="paused")
            current = progress.load_progress(root, run_id)
            return {
                "run_id": run_id,
                "status": "paused",
                "turn": current["turn"],
                "note": str(error) or type(error).__name__,
            }
        duration_ms = (time.perf_counter() - t0) * 1000
        output = list(response.output or [])
        messages.extend(output)
        call = _find_call(output)
        usage = getattr(response, "usage", None)
        progress.finish_step(
            root, run_id, step_no,
            duration_ms=duration_ms, status="ok",
            summary=f"called {_field(call, 'name')}" if call else "replied in text",
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )
        if call is None:
            messages.append({"role": "user", "content": NO_TOOL_MESSAGE})
            progress.save_progress(root, run_id, messages, response, status="running",
                                   model_name=model(), duration_ms=duration_ms,
                                   step_status="ok")
            continue
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
                               model_name=model(), duration_ms=duration_ms,
                               step_status="ok")
        if status != "running":
            current = progress.load_progress(root, run_id)
            return {"run_id": run_id, "status": status, "turn": current["turn"]}
    current = progress.load_progress(root, run_id)
    if current["status"] == "running":
        progress.save_progress(root, run_id, messages, status="paused")
        current = progress.load_progress(root, run_id)
    return {"run_id": run_id, "status": current["status"], "turn": current["turn"]}


def run_agent(*, root: Path, client: OpenAI, turns: int = 30) -> dict:
    root = Path(root)
    instructions = _instructions()
    objective = _objective(root)
    run_id = progress.new_run(root, instructions=instructions, objective=objective)
    messages: list = [{"role": "user", "content": objective}]
    return _run_loop(
        root=root,
        client=client,
        run_id=run_id,
        messages=messages,
        instructions=instructions,
        turns=turns,
    )


def resume_run(*, root: Path, run_id: str, client: OpenAI, turns: int = 15) -> dict:
    root = Path(root)
    saved = progress.load_progress(root, run_id)
    step = progress.resume(root, run_id)
    if step["message"] is None:
        current = progress.load_progress(root, run_id)
        result: dict = {
            "run_id": run_id,
            "status": current["status"],
            "turn": current["turn"],
        }
        if step.get("note"):
            result["note"] = step["note"]
        return result
    instructions_path = root / "workspace/runs" / run_id / "instructions.md"
    instructions = (
        instructions_path.read_text() if instructions_path.is_file() else _instructions()
    )
    messages: list = list(saved["messages"])
    messages.append({"role": "user", "content": step["message"]})
    return _run_loop(
        root=root,
        client=client,
        run_id=run_id,
        messages=messages,
        instructions=instructions,
        turns=turns,
    )
