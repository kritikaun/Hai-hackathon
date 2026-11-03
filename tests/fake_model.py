"""An offline Model API for the tests: `Scripted(model).client` is a real OpenAI client
whose requests never leave the machine. Nothing here is part of the agent.

ScriptedModel recognises:
- agent turns (requests with `tools`): the next scripted tool call;
- text extraction ("Transcribe document ..."): the transcript you supply;
- document reading ("VERIFIED PAGE TEXT"): evidence quoting the transcript;
- the checklist check ("REFERENCE CHECKLIST"): coverage by matching item ids, plus feedback;
- the claims check ("CLAIMS TO CHECK"): every claim supported unless told otherwise;
plus SAM segmentation calls, answered from recorded box outputs.
"""

from __future__ import annotations

import json
import re
import threading
from typing import Callable

import httpx
from agent_kit.client import DEFAULT_BASE_URL
from agent_kit.settings import sam_model
from openai import OpenAI

FEEDBACK = (
    "The checklist leaves out an area the question turns on. Look again at who was "
    "involved and what their role means for the answer."
)


DOC_ID = re.compile(r"(?:Transcribe document|case document) ([A-Z]+-[A-Z0-9]+)")


class Refused(Exception):
    """Raise from a responder to make the scripted API answer HTTP 400 with this message."""


def _text_of(request: dict) -> str:
    if isinstance(request.get("input"), str):
        return request["input"]
    parts = []
    for item in request.get("input", []):
        for block in (
            item.get("content", []) if isinstance(item.get("content"), list) else []
        ):
            if isinstance(block, dict) and block.get("type") == "input_text":
                parts.append(block["text"])
    return "\n".join(parts)


def history(request: dict) -> list:
    """Function calls and outputs seen so far in a local-mode conversation."""
    items = request.get("input")
    return [
        item
        for item in (items if isinstance(items, list) else [])
        if item.get("type") in ("function_call", "function_call_output")
    ]


def tool_call(name: str, arguments: dict, n: int) -> dict:
    return {
        "output": [
            {
                "type": "function_call",
                "call_id": f"call-{n}",
                "name": name,
                "arguments": json.dumps(arguments),
            }
        ],
        "usage": {"input_tokens": 1500 + 50 * n, "output_tokens": 80},
    }


class ScriptedModel:
    def __init__(
        self,
        transcripts: dict,
        planner: Callable[[dict], tuple | None],
        sam_outputs: dict | None = None,
        evidence: Callable[[str, list], list] | None = None,
        unsupported: Callable[[str], list] | None = None,
        feedback: str = FEEDBACK,
    ) -> None:
        self.transcripts = transcripts
        self.planner = planner
        self.sam_outputs = sam_outputs or {}
        self.evidence = evidence or default_evidence
        self.unsupported = unsupported or (lambda text: [])
        self.feedback = feedback
        self.n = 0

    def __call__(self, kind: str, payload) -> dict:
        self.n += 1
        if kind == "segment":
            image_url, _prompt = payload
            return {
                "output_text": self.sam_outputs.get(image_url, ""),
                "usage": {"input_tokens": 800, "output_tokens": 40},
            }
        request = payload
        if "tools" in request:
            action = self.planner(request)
            if action is None:
                return {"output_text": "(script finished)"}
            name, arguments = action
            return tool_call(name, arguments, self.n)
        text = _text_of(request)
        if "REFERENCE CHECKLIST" in text:
            covered = checklist_coverage(text)
            return {
                "output_text": json.dumps(
                    {
                        "covered": covered,
                        "feedback": "" if all(covered.values()) else self.feedback,
                    }
                ),
                "usage": {"input_tokens": 900, "output_tokens": 60},
            }
        if "CLAIMS TO CHECK" in text:
            return {
                "output_text": json.dumps({"unsupported": self.unsupported(text)}),
                "usage": {"input_tokens": 1800, "output_tokens": 60},
            }
        match = DOC_ID.search(text)
        doc = match.group(1) if match else ""
        pages = self.transcripts.get(doc, [{"page": 1, "text": ""}])
        if text.startswith("Transcribe document"):
            return {
                "output_text": json.dumps({"pages": pages}),
                "usage": {"input_tokens": 3000, "output_tokens": 900},
            }
        return {
            "output_text": json.dumps(
                {"evidence": self.evidence(doc, pages), "note": ""}
            ),
            "usage": {"input_tokens": 3500, "output_tokens": 300},
        }


def _json_lists(text: str) -> list:
    decoder, found, i = json.JSONDecoder(), [], 0
    while (i := text.find("[", i)) != -1:
        try:
            value, end = decoder.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        if isinstance(value, list):
            found.append(value)
        i = end
    return found


def checklist_coverage(text: str) -> dict:
    """Scripted judge: a reference item is covered (True) by an agent item whose id contains it (or vice versa)."""
    lists = _json_lists(text)
    if len(lists) < 2:
        return {}
    reference, agent = lists[0], lists[1]
    agent_ids = [str(a.get("id", "")) for a in agent if isinstance(a, dict)]
    coverage = {}
    for ref in reference:
        rid = str(ref.get("id", ""))
        coverage[rid] = any(a and (a == rid or rid in a or a in rid) for a in agent_ids)
    return coverage


def default_evidence(doc: str, pages: list) -> list:
    for page in pages:
        lines = [
            line.strip() for line in page["text"].splitlines() if len(line.strip()) > 25
        ]
        if lines:
            return [
                {
                    "page": page["page"],
                    "quote": lines[0],
                    "claim": f"{doc} says: {lines[0][:80]}",
                    "interpretation": "Scripted reading.",
                    "ambiguity": "",
                }
            ]
    return []


def _response_body(answer: dict, n: int) -> dict:
    """A Responses API body for a scripted answer (output items, or output_text)."""
    output = answer.get("output") or []
    if not output and answer.get("output_text"):
        output = [
            {
                "type": "message",
                "id": f"msg-{n}",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": answer["output_text"],
                        "annotations": [],
                    }
                ],
            }
        ]
    usage = answer.get("usage") or {"input_tokens": 100, "output_tokens": 20}
    return {
        "id": f"resp-scripted-{n}",
        "object": "response",
        "created_at": 0,
        "model": "scripted",
        "status": "completed",
        "output": [dict(item, id=item.get("id", f"item-{n}")) for item in output],
        "usage": dict(
            usage, total_tokens=usage["input_tokens"] + usage["output_tokens"]
        ),
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }


class ScriptedTransport(httpx.BaseTransport):
    """Answers /responses, /files and /models offline, and records every call in `calls`.

    Each call is (kind, payload): ("respond", request body), ("segment", (image_url, prompt))
    for SAM, or ("upload", file name). `responder(kind, payload)` gives the answer.
    """

    def __init__(self, responder: Callable[[str, object], dict]) -> None:
        self.responder = responder
        self.calls: list = []
        self.headers: list = []
        self._lock = threading.Lock()
        self._n = 0

    def _next(self) -> int:
        with self._lock:
            self._n += 1
            return self._n

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        path, n = request.url.path, self._next()
        if path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": []})
        if path.endswith("/files"):
            name = re.search(rb'filename="([^"]*)"', request.read())
            self.calls.append(("upload", name.group(1).decode() if name else ""))
            return httpx.Response(
                200,
                json={
                    "id": f"file-scripted-{n}",
                    "object": "file",
                    "bytes": 0,
                    "created_at": 0,
                    "filename": name.group(1).decode() if name else "file",
                    "purpose": "user_data",
                    "status": "uploaded",
                },
            )
        self.headers.append(dict(request.headers))
        body = json.loads(request.content or b"{}")
        if body.get("model") == sam_model():
            content = body["input"][0]["content"]
            prompt = next(b["text"] for b in content if b["type"] == "input_text")
            image = next(b["image_url"] for b in content if b["type"] == "input_image")
            kind, payload = "segment", (image, prompt)
        else:
            kind, payload = "respond", body
        self.calls.append((kind, payload))
        try:
            answer = self.responder(kind, payload) or {}
        except Refused as error:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": str(error),
                        "type": "invalid_request_error",
                        "code": None,
                        "param": None,
                    }
                },
            )
        return httpx.Response(200, json=_response_body(answer, n))


class Scripted:
    """An OpenAI client answered by a ScriptedModel, plus the calls it received."""

    def __init__(self, model: Callable[[str, object], dict]) -> None:
        self.transport = ScriptedTransport(model)
        self.client: OpenAI = OpenAI(
            api_key="offline-test",
            base_url=DEFAULT_BASE_URL,
            max_retries=0,
            http_client=httpx.Client(transport=self.transport),
        )

    @property
    def calls(self) -> list:
        return self.transport.calls

    @property
    def responder(self) -> Callable[[str, object], dict]:
        return self.transport.responder

    @responder.setter
    def responder(self, value: Callable[[str, object], dict]) -> None:
        self.transport.responder = value
