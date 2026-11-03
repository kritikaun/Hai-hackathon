"""Pre-built: is each finding supported by the quotes it cites?

Called by the submit_memo tool after verify_memo passes, and by ./helper verify RUN --memo.

    run_verification(ctx, memo) -> {"passed": bool, "feedback": [problem, ...]}

One Muse Spark call judges every proven finding in the memo against its quotes.
"""

from __future__ import annotations

import json
import time
from typing import cast

from agent_kit import progress
from agent_kit.settings import verify_model
from openai.types.responses import Response

PROMPT = """CLAIMS TO CHECK. Each finding below states a claim and the exact quotes cited for it.
For each finding, decide whether the quotes support the claim as written. A claim that goes
beyond what its quotes say is unsupported.

{findings}

Answer only with JSON: {{"unsupported": [{{"item": "<item>", "reason": "one sentence"}}]}}
(an empty list when every claim is supported)."""


def run_verification(ctx, memo: dict) -> dict:
    findings = [
        {
            "item": f.get("item"),
            "claim": f.get("summary", ""),
            "quotes": [c.get("quote", "") for c in f.get("citations") or []],
        }
        for f in memo.get("findings") or []
        if isinstance(f, dict) and f.get("status") == "proven" and f.get("citations")
    ]
    if not findings:
        return {"passed": True, "feedback": []}
    run_id = (getattr(ctx, "case", None) and ctx.case.data.get("run_id")) or ""
    step_entry = progress.start_step(
        ctx.root, run_id, task="verify_claims", model=verify_model(), summary="Checking…"
    )
    t0 = time.perf_counter()
    response = cast(
        Response,
        ctx.client.responses.create(
            model=verify_model(),
            input=PROMPT.format(findings=json.dumps(findings, indent=1)),
            text={"format": {"type": "json_object"}},
        ),
    )
    usage = getattr(response, "usage", None)
    progress.finish_step(
        ctx.root, run_id, step_entry["step"] if step_entry else 0,
        duration_ms=(time.perf_counter() - t0) * 1000,
        summary=f"checked {len(findings)} findings against their quotes",
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
    )
    try:
        unsupported = json.loads(response.output_text).get("unsupported") or []
    except (ValueError, AttributeError):
        return {
            "passed": True,
            "feedback": [],
        }  # an unreadable judgement never blocks the memo
    problems = [
        f"{u.get('item')}: the finding is not supported by its quotes ({u.get('reason', '')})"
        for u in unsupported
        if isinstance(u, dict)
    ]
    return {"passed": not problems, "feedback": problems}
