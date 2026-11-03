"""The case record: the simulated case-management system the agent acts on.

One JSON record per run at workspace/case_records/<run-id>.json, written only
through this module and append-audited. The agent changes it through tools
(set_checklist, record_fact, flag_privileged, request_evidence, submit_memo); people change
it through ./helper review (answer a request, approve or reject the memo).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from . import document_text
from .progress import now

ITEM_STATUSES = ("open", "proven", "escalated")


MAX_FACTS_PER_ITEM = 3
MAX_CITATIONS_PER_FINDING = 3


class CaseError(ValueError):
    """A write the case record refused. The message is safe to show the model."""


class FactLimitError(CaseError):
    """The item already has MAX_FACTS_PER_ITEM facts. Not a rejected quote."""


ITEM_ID = re.compile(r"^[a-z][a-z0-9_]{0,47}$")


class CaseRecord:
    def __init__(self, root: Path, run_id: str) -> None:
        self.root = root
        self.path = root / "workspace/case_records" / f"{run_id}.json"
        if self.path.is_file():
            self.data: dict[str, Any] = json.loads(self.path.read_text())
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.data = {
                "run_id": run_id,
                "created": now(),
                "checklist": {},
                "checklist_checks": [],
                "memo_checks": [],
                "read_ids": [],
                "facts": [],
                "rejected_facts": [],
                "flags": [],
                "signature_checks": [],
                "requests": [],
                "memo": None,
                "audit": [],
            }
            self._save("created", "runtime", {})

    # ---- persistence ----
    def _save(self, action: str, actor: str, detail: dict) -> None:
        self.data["audit"].append(
            {"at": now(), "actor": actor, "action": action, "detail": detail}
        )
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(self.path)

    def _item(self, item: str) -> dict:
        if not self.data["checklist"]:
            raise CaseError("there is no checklist yet: call set_checklist first")
        if item not in self.data["checklist"]:
            raise CaseError(
                f"unknown checklist item {item!r}; valid items: {', '.join(self.data['checklist'])}"
            )
        return self.data["checklist"][item]

    def flagged(self, document_id: str) -> bool:
        return any(
            flag["document_id"] == document_id and flag["kind"] == "privileged"
            for flag in self.data["flags"]
        )

    # ---- agent writes ----
    def parse_checklist(self, items: list) -> dict:
        """Validate checklist items and return the new checklist, without saving it."""
        if not isinstance(items, list) or not items:
            raise CaseError("the checklist needs at least one item")
        new: dict[str, dict] = {}
        for raw in items:
            if not isinstance(raw, dict):
                raise CaseError("each checklist item must be an object")
            item_id = str(raw.get("id", "")).strip()
            if not ITEM_ID.match(item_id):
                raise CaseError(
                    f"item id {item_id!r} must be snake_case, e.g. ready_statement"
                )
            if item_id in new:
                raise CaseError(f"duplicate checklist item {item_id}")
            question = str(raw.get("question", "")).strip()
            evidence = str(raw.get("evidence_needed", "")).strip()
            if not question:
                raise CaseError(f"{item_id}: question is empty")
            if not evidence:
                raise CaseError(
                    f"{item_id}: say what evidence would settle it (evidence_needed)"
                )
            old = self.data["checklist"].get(item_id, {})
            new[item_id] = {
                "question": question,
                "evidence_needed": evidence,
                "needs_signature_check": bool(raw.get("needs_signature_check")),
                "status": old.get("status", "open"),
                "facts": old.get("facts", []),
                "request": old.get("request"),
            }
        dropped = [
            k
            for k, v in self.data["checklist"].items()
            if k not in new and (v["facts"] or v["request"])
        ]
        if dropped:
            raise CaseError(
                f"keep the items that already have facts or requests: {', '.join(dropped)}"
            )
        return new

    def set_checklist(self, items: list) -> dict:
        """Replace the checklist. Items that already have facts or requests must be kept."""
        new = self.parse_checklist(items)
        checks = self.data.get("checklist_checks", [])
        first_draft = (
            set(checks[0]["draft"]) if checks and not checks[0]["passed"] else None
        )
        for item_id, entry in new.items():
            entry["added_after_check"] = (
                first_draft is not None and item_id not in first_draft
            )
        self.data["checklist"] = new
        self._save("checklist_set", "agent", {"items": list(new)})
        return new

    def record_checklist_check(self, items: list, result: dict) -> dict:
        """Store the outcome of the ground-truth checklist check for one draft."""
        record = {
            "draft": [str(i.get("id")) for i in items if isinstance(i, dict)],
            "passed": bool(result.get("passed")),
            "score": result.get("score"),
            "feedback": result.get("feedback", ""),
            "attempt": result.get("attempt"),
            "accepted": bool(result.get("accepted", result.get("passed"))),
            "at": now(),
        }
        self.data.setdefault("checklist_checks", []).append(record)
        self._save(
            "checklist_checked",
            "verification",
            {"passed": record["passed"], "accepted": record["accepted"]},
        )
        return record

    def record_memo_check(self, check: str, problems: list) -> dict:
        """Store the outcome of a memo check (memo rules or claims) for one submitted memo."""
        record = {
            "check": check,
            "passed": not problems,
            "problems": problems,
            "at": now(),
        }
        self.data.setdefault("memo_checks", []).append(record)
        self._save(
            "memo_checked", "verification", {"check": check, "passed": not problems}
        )
        return record

    def memo_refusals(self) -> int:
        return sum(1 for c in self.data.get("memo_checks", []) if not c["passed"])

    def mark_read(self, document_ids: list) -> None:
        """Remember which documents the agent has read (the escalation rule needs it)."""
        read = self.data.setdefault("read_ids", [])
        new = [d for d in document_ids if d not in read]
        if new:
            read += new
            self._save("documents_read", "agent", {"documents": new})

    def progress(self) -> dict:
        """Where the checklist stands: proven, escalated and open items, and what to do next."""
        checklist = self.data["checklist"]
        by_status = {
            status: [k for k, v in checklist.items() if v["status"] == status]
            for status in ("proven", "escalated", "open")
        }
        done = len(by_status["proven"]) + len(by_status["escalated"])
        if by_status["open"]:
            next_step = (
                f"Open: {', '.join(by_status['open'])}. Get one fact for each open item "
                "before adding more to any item."
            )
        else:
            next_step = (
                "Every item is proven or escalated: file the memo now. Add another fact "
                f"only if one could be disputed (at most {MAX_FACTS_PER_ITEM} per item)."
            )
        return {
            "proven": len(by_status["proven"]),
            "escalated": len(by_status["escalated"]),
            "open": by_status["open"],
            "total": len(checklist),
            "summary": f"{done} of {len(checklist)} items proven or escalated. {next_step}",
        }

    def record_fact(
        self, item: str, document_id: str, page: int, quote: str, claim: str
    ) -> dict:
        """Mark evidence for a checklist item. Refused unless the quote is on that page.

        At most MAX_FACTS_PER_ITEM facts per item (FactLimitError). The returned fact
        carries `progress` (see progress()) so the model sees what is still open.
        """
        entry = self._item(item)
        if len(entry["facts"]) >= MAX_FACTS_PER_ITEM:
            raise FactLimitError(
                f"{item} already has {MAX_FACTS_PER_ITEM} facts, the most an item takes. "
                + self.progress()["summary"]
            )
        try:
            page = int(page)
            if not claim or not str(claim).strip():
                raise CaseError("claim must be non-empty")
            if self.flagged(document_id):
                raise CaseError(
                    f"{document_id} is flagged as confidential and can never be cited"
                )
            document_text.get_document(self.root, document_id)
            text = document_text.page_text(self.root, document_id, page)
            if text is None:
                raise CaseError(
                    f"{document_id} has not been analyzed yet; call analyze_document first"
                )
            if not document_text.quote_on_page(quote, text):
                raise CaseError(
                    f"quote not found on {document_id} page {page}; copy the exact words from the document"
                )
        except (CaseError, document_text.DocumentError, TypeError, ValueError) as error:
            self.data["rejected_facts"].append(
                {
                    "item": item,
                    "document_id": document_id,
                    "page": page,
                    "quote": quote,
                    "reason": str(error),
                    "at": now(),
                }
            )
            self._save(
                "fact_rejected",
                "agent",
                {"item": item, "document_id": document_id, "reason": str(error)},
            )
            raise CaseError(str(error)) from None
        fact = {
            "id": f"F{len(self.data['facts']) + 1}",
            "item": item,
            "document_id": document_id,
            "page": page,
            "quote": quote,
            "claim": str(claim),
            "at": now(),
        }
        self.data["facts"].append(fact)
        entry["facts"].append(fact["id"])
        entry["status"] = "proven"
        self._save(
            "fact_recorded",
            "agent",
            {
                "fact": fact["id"],
                "item": item,
                "document_id": document_id,
                "page": page,
            },
        )
        return dict(fact, progress=self.progress())

    def flag_privileged(self, document_id: str, reason: str) -> dict:
        document_text.get_document(self.root, document_id)
        if self.flagged(document_id):
            return next(
                f
                for f in self.data["flags"]
                if f["document_id"] == document_id and f["kind"] == "privileged"
            )
        flag = {
            "kind": "privileged",
            "document_id": document_id,
            "reason": str(reason),
            "review": "pending",
            "at": now(),
        }
        self.data["flags"].append(flag)
        withdrawn = [
            f["id"] for f in self.data["facts"] if f["document_id"] == document_id
        ]
        self.data["facts"] = [
            f for f in self.data["facts"] if f["document_id"] != document_id
        ]
        for entry in self.data["checklist"].values():
            entry["facts"] = [f for f in entry["facts"] if f not in withdrawn]
            if entry["status"] == "proven" and not entry["facts"]:
                entry["status"] = "open"
        self._save(
            "document_flagged",
            "agent",
            {"document_id": document_id, "withdrawn_facts": withdrawn},
        )
        return flag

    def add_signature_check(self, result: dict) -> dict:
        record = dict(result, at=now())
        self.data["signature_checks"].append(record)
        if result.get("verdict") in ("inconsistent", "unclear", "unsigned"):
            self.data["flags"].append(
                {
                    "kind": "signature",
                    "document_id": result.get("document_id"),
                    "reason": f"{result.get('verdict')}: {result.get('reason', '')}",
                    "review": "pending",
                    "at": now(),
                }
            )
        self._save(
            "signature_checked",
            "agent",
            {
                "document_id": result.get("document_id"),
                "verdict": result.get("verdict"),
            },
        )
        return record

    def open_request(
        self,
        item: str,
        need: str,
        rule: str,
        visible_ids: list,
        evidence_so_far: list | None = None,
    ) -> dict:
        entry = self._item(item)
        if not need or not str(need).strip():
            raise CaseError("need must say exactly what evidence is missing")
        request = {
            "id": f"R{len(self.data['requests']) + 1}",
            "item": item,
            "rule": rule,
            "need": str(need),
            "visible_ids": list(visible_ids),
            "evidence_so_far": evidence_so_far or list(self.data["facts"]),
            "status": "open",
            "response": None,
            "at": now(),
        }
        self.data["requests"].append(request)
        entry["request"] = request["id"]
        if entry["status"] == "open":
            entry["status"] = "escalated"
        self._save(
            "request_opened",
            "agent",
            {"request": request["id"], "item": item, "rule": rule},
        )
        return request

    def open_requests(self) -> list:
        return [r for r in self.data["requests"] if r["status"] == "open"]

    def file_memo(self, memo: dict, status: str) -> dict:
        """File the memo. Refused if a finding cites more than MAX_CITATIONS_PER_FINDING facts."""
        for finding in memo.get("findings") or []:
            citations = finding.get("citations") if isinstance(finding, dict) else None
            if (
                isinstance(citations, list)
                and len(citations) > MAX_CITATIONS_PER_FINDING
            ):
                raise CaseError(
                    f"the finding for {finding.get('item')} cites {len(citations)} facts; "
                    f"cite its {MAX_CITATIONS_PER_FINDING} strongest at most"
                )
        record = dict(memo, status=status, filed=now(), review=None)
        self.data["memo"] = record
        self._save("memo_filed", "agent", {"status": status})
        return record

    # ---- people ----
    def answer_request(self, request_id: str, response: str) -> dict:
        for request in self.data["requests"]:
            if request["id"] == request_id:
                request["status"] = "answered"
                request["response"] = response
                request["answered"] = now()
                self._save("request_answered", "human", {"request": request_id})
                return request
        raise CaseError(f"no request {request_id}")

    def review_memo(
        self, approve: bool, notes: str = "", flag_decisions: dict | None = None
    ) -> dict:
        memo = self.data.get("memo")
        if not memo:
            raise CaseError("no memo has been filed")
        memo["review"] = {
            "decision": "approved" if approve else "rejected",
            "notes": notes,
            "at": now(),
        }
        memo["status"] = "approved" if approve else "rejected"
        for flag in self.data["flags"]:
            flag["review"] = (flag_decisions or {}).get(
                flag["document_id"], "confirmed" if approve else flag["review"]
            )
        self._save("memo_reviewed", "human", {"decision": memo["status"]})
        return memo

    # ---- reads ----
    def summary(self) -> dict:
        return {
            "checklist": {
                k: {
                    "status": v["status"],
                    "facts": len(v["facts"]),
                    "request": v["request"],
                }
                for k, v in self.data["checklist"].items()
            },
            "facts": len(self.data["facts"]),
            "flags": [
                {"kind": f["kind"], "document_id": f["document_id"]}
                for f in self.data["flags"]
            ],
            "open_requests": [r["id"] for r in self.open_requests()],
            "memo": (self.data["memo"] or {}).get("status"),
        }
