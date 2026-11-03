"""The memo rules.

Called when the agent calls submit_memo with VERIFICATION=on, and by
./helper verify RUN --memo.

    run_verification(ctx, memo) -> {"passed": bool, "feedback": [problem, ...]}

memo = {"conclusion": str, "findings": [{"item", "status", "summary", "citations": [
{"document_id", "page", "quote"}]}], "gaps": str}. ctx.case is the run's CaseRecord:
case.data["checklist"], case.data["facts"], case.data["signature_checks"] (each with
document_id, page, verdict), case.flagged(document_id), case.open_requests(). Compare
quotes with agent_kit.document_text.normalize.

Rules (one problem string per break, which the agent reads and fixes; passed = no problems):
- every checklist item has a finding, and is proven or escalated (status "open" is a problem);
- a finding marked "proven" has a recorded fact for its item, and cites at least one;
- every citation matches a recorded fact (same document, page, and quote);
- no citation is from a document flagged confidential;
- an item with needs_signature_check that is marked proven has a signature check on a
  document it cites, and every such check's verdict is "consistent" (otherwise the item
  is not proven: say so, and name it as a gap);
- no evidence request is still open.
"""

from __future__ import annotations

from agent_kit.document_text import normalize


def _page(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _citation_matches_fact(citation: dict, fact: dict) -> bool:
    if fact.get("document_id") != str(citation.get("document_id", "")):
        return False
    if _page(fact.get("page")) != _page(citation.get("page")):
        return False
    return normalize(str(fact.get("quote", ""))) == normalize(
        str(citation.get("quote", ""))
    )


def run_verification(ctx, memo: dict) -> dict:
    case = ctx.case
    checklist: dict = case.data.get("checklist", {})
    facts: list = case.data.get("facts", [])
    raw_findings = memo.get("findings") or [] if isinstance(memo, dict) else []
    findings = [f for f in raw_findings if isinstance(f, dict)]

    by_item: dict[str, list] = {}
    for finding in findings:
        item = finding.get("item")
        if item:
            by_item.setdefault(str(item), []).append(finding)

    problems: list[str] = []

    for item in checklist:
        if item not in by_item:
            problems.append(f"no finding for {item}")

    for item, entry in checklist.items():
        if isinstance(entry, dict) and entry.get("status") == "open":
            problems.append(f"{item} is neither proven nor escalated")

    for item, item_findings in by_item.items():
        entry = checklist.get(item)
        if entry is None:
            problems.append(f"unknown checklist item {item!r}")
        for finding in item_findings:
            status = finding.get("status")
            citations = finding.get("citations") or []
            if not isinstance(citations, list):
                citations = []
            citations = [c for c in citations if isinstance(c, dict)]

            if status == "proven":
                own_facts = [f for f in facts if f.get("item") == item]
                if not own_facts:
                    problems.append(
                        f"{item} is marked proven but has no recorded fact"
                    )
                if not citations:
                    problems.append(f"{item} is marked proven but cites no fact")
                elif own_facts and not any(
                    any(_citation_matches_fact(c, f) for f in own_facts)
                    for c in citations
                ):
                    problems.append(
                        f"{item} is marked proven but cites none of its recorded facts"
                    )

            for citation in citations:
                doc = str(citation.get("document_id", ""))
                if doc and case.flagged(doc):
                    problems.append(
                        f"{item}: citation from confidential document {doc} is not allowed"
                    )
                if not any(_citation_matches_fact(citation, f) for f in facts):
                    problems.append(
                        f"{item}: citation for {doc} p{_page(citation.get('page'))} "
                        "has no matching recorded fact"
                    )

            if status == "proven" and isinstance(entry, dict):
                if entry.get("needs_signature_check"):
                    cited_docs = {
                        str(c.get("document_id", ""))
                        for c in citations
                        if c.get("document_id")
                    }
                    checks = [
                        s
                        for s in case.data.get("signature_checks", [])
                        if s.get("document_id") in cited_docs
                    ]
                    if not checks:
                        problems.append(
                            f"{item}: no cited document was checked for the signature"
                        )
                    for check in checks:
                        if check.get("verdict") != "consistent":
                            reason = check.get("reason", "")
                            message = (
                                f"{check.get('document_id')} p{_page(check.get('page'))} "
                                f"is {check.get('verdict')}: {item} is not proven, "
                                "name it as a gap"
                            )
                            if reason:
                                message += f" ({reason})"
                            problems.append(message)

    for request in case.open_requests():
        problems.append(
            f"request is still open for {request.get('item')} ({request.get('id')})"
        )

    return {"passed": not problems, "feedback": problems}
