"""Shared helpers for the offline tests: a tiny case on disk and a scripted client."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from agent_kit.tools import check_signatures
from tests.fake_model import Scripted, ScriptedModel


# A tiny case: document id -> page texts.
INITIAL = {
    "DOC-REPORT": [
        "Weekly status report. Overall status is green. Open Severity 1 defects: 21 at period end."
    ],
    "DOC-MINUTES": [
        "Steering minutes. Mr. Lindgren confirmed the system is ready for go-live. Erik Lindgren, Engagement Partner."
    ],
    "DOC-MEMO": [
        "PRIVILEGED AND CONFIDENTIAL. Attorney work product for the review team."
    ],
    "DOC-REF": ["Test strategy approval page signed by Dana Okafor, Test Manager."],
}
FOLLOW_UP = {
    "DOC-DRAFT": [
        "Draft readiness assessment dated December 19. Recommendation: NO-GO. Nine of fourteen criteria not met."
    ],
    "DOC-CERT": [
        "UAT Phase 2 exit certification. I certify that all exit criteria have been satisfied. D. Okafor."
    ],
}
PNG = bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000000")


# The tiny case's ground-truth checklist (verification data outside the case; never sent to the model).
GROUND_TRUTH = [
    {
        "id": "not_ready",
        "question": "Do internal records show the system was not ready?",
    },
    {"id": "ready_statement", "question": "Did they say it was ready?"},
    {"id": "timing", "question": "Did they know before they said it?"},
    {"id": "seniority", "question": "Were the people involved senior?"},
    {
        "id": "signatures",
        "question": "Was the certification signed by the named person?",
    },
    {"id": "counterevidence", "question": "Is there evidence against the above?"},
]


def instructions_file() -> str:
    """legal_agent/system_instructions.md: the agent must send it unchanged as its instructions."""
    return (
        Path(__file__).absolute().parents[1] / "legal_agent/system_instructions.md"
    ).read_text()


def first_agent_request(fake: Scripted) -> dict:
    return next(c[1] for c in fake.calls if c[0] == "respond" and "tools" in c[1])


def checklist_item(item_id: str, signature: bool = False) -> dict:
    return {
        "id": item_id,
        "question": f"Question for {item_id}",
        "evidence_needed": f"A document showing {item_id}",
        "needs_signature_check": signature,
    }


CHECKLIST_ITEMS = [
    checklist_item(g["id"], signature=g["id"] == "signatures") for g in GROUND_TRUTH
]

# Scripted tool calls shared by the tests.
CHECKLIST = ("set_checklist", {"items": CHECKLIST_ITEMS})
# 4 of 6 ground-truth items: below the 80% bar, so the checklist check refuses it.
PARTIAL_CHECKLIST = (
    "set_checklist",
    {"items": [i for i in CHECKLIST_ITEMS if i["id"] not in ("seniority", "timing")]},
)
PLAN = (
    "update_plan",
    {
        "plan": [
            "Read every document",
            "Record evidence per checklist item",
            "File the memo",
        ]
    },
)
LIST = ("list_documents", {})
READ = (
    "analyze_document",
    {
        "document_ids": ["DOC-REPORT", "DOC-MINUTES"],
        "request": "readiness and who said it",
    },
)
FACT = (
    "record_fact",
    {
        "item": "ready_statement",
        "document_id": "DOC-MINUTES",
        "page": 1,
        "quote": "Mr. Lindgren confirmed the system is ready for go-live",
        "claim": "the vendor said it was ready",
    },
)
MEMO = (
    "submit_memo",
    {
        "conclusion": "Probably yes.",
        "findings": [
            {
                "item": "ready_statement",
                "status": "proven",
                "summary": "Said ready.",
                "citations": [
                    {
                        "document_id": "DOC-MINUTES",
                        "page": 1,
                        "quote": "Mr. Lindgren confirmed the system is ready for go-live",
                    }
                ],
            }
        ],
        "gaps": "Timing unknown.",
    },
)


def _write_doc(folder: Path, doc_id: str, pages: list) -> dict:
    content = ("\n".join(pages)).encode()
    (folder / f"{doc_id}.pdf").write_bytes(content)
    (folder / "pages").mkdir(exist_ok=True)
    for n in range(1, len(pages) + 1):
        (folder / "pages" / f"{doc_id}-p{n}.png").write_bytes(
            PNG + doc_id.encode() + bytes([n])
        )
    return {"id": doc_id, "file": f"{doc_id}.pdf"}


def make_root() -> tuple[Path, tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    (root / "case").mkdir()
    os.environ["CASE_DIR"] = str(root / "case")
    (root / "case/objective.md").write_text(
        "Did the vendor know the system was not ready, and say it was?"
    )
    (root / "ground_truth").mkdir()
    os.environ["GROUND_TRUTH_DIR"] = str(root / "ground_truth")
    (root / "ground_truth/case.json").write_text(json.dumps({"items": GROUND_TRUTH}))
    inbox = root / "workspace/inbox"
    inbox.mkdir(parents=True)
    for doc_id, pages in INITIAL.items():
        _write_doc(inbox, doc_id, pages)
    later = root / "follow_up"
    later.mkdir()
    for doc_id, pages in FOLLOW_UP.items():
        _write_doc(later, doc_id, pages)
    return root, tmp


def release(root: Path) -> None:
    """Release the follow-up documents into the inbox, for the tiny case."""
    inbox = root / "workspace/inbox"
    for path in (root / "follow_up").glob("*.pdf"):
        shutil.copy(path, inbox / path.name)
    for path in (root / "follow_up/pages").glob("*.png"):
        shutil.copy(path, inbox / "pages" / path.name)


def transcripts() -> dict:
    return {
        d: [{"page": i + 1, "text": t} for i, t in enumerate(p)]
        for d, p in {**INITIAL, **FOLLOW_UP}.items()
    }


def scripted(root: Path, actions: list, unsupported=None) -> Scripted:
    """A scripted OpenAI client (fake.client) that plays `actions` (name, args) in order, one per agent turn.

    unsupported(text) -> list: what the claims check reports (default: every claim supported).
    """
    sam_outputs = {}
    box = "<0f>0<|box;x1=10;y1=20;x2=110;y2=60;w=850;h=1100|><|mask;x=0;y=0;data=100,40,~x|>"
    for folder in (root / "workspace/inbox/pages", root / "follow_up/pages"):
        for image in folder.glob("*.png"):
            if image.name.startswith(("DOC-REF", "DOC-CERT")):
                sam_outputs[check_signatures.data_url(image)] = box
    queue = list(actions)

    def planner(request: dict):
        return queue.pop(0) if queue else None

    return Scripted(
        ScriptedModel(transcripts(), planner, sam_outputs, unsupported=unsupported)
    )


def agent_requests(fake: Scripted) -> list:
    return [c[1] for c in fake.calls if c[0] == "respond" and "tools" in c[1]]


def progress(root: Path, run_id: str) -> dict:
    return json.loads((root / "workspace/runs" / run_id / "progress.json").read_text())


def tool_calls(root: Path, run_id: str) -> list:
    """(name, arguments, result) for every tool call in a run, from its saved conversation."""
    messages = progress(root, run_id)["messages"]
    outputs = {
        m.get("call_id"): json.loads(m.get("output") or "{}")
        for m in messages
        if m.get("type") == "function_call_output"
    }
    return [
        (
            m["name"],
            json.loads(m.get("arguments") or "{}"),
            outputs.get(m.get("call_id")),
        )
        for m in messages
        if m.get("type") == "function_call"
    ]


def case(root: Path, run_id: str) -> dict:
    return json.loads((root / "workspace/case_records" / f"{run_id}.json").read_text())
