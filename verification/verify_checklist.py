"""Does the agent's checklist cover what the case needs?

Called when the agent calls set_checklist with VERIFICATION=on, and by
./helper verify RUN --checklist.

    run_verification(ctx, items) -> {"passed": bool, "score": float, "feedback": str}

- items: the agent's checklist items: [{"id", "question", "evidence_needed", ...}, ...]
- load_reference(ctx.root) (pre-built, below) returns the case's ground truth,
  [{"id": "...", "question": "..."}, ...], or None when the case has none (then the
  checklist passes).

1. Ask Muse Spark (ctx.client.responses.create, JSON output) to mark each reference item
   covered or not by some agent item, even if worded differently, and to write feedback.
2. score = covered reference items / all reference items; passed = coverage is at least 80%.
3. feedback: what kind of question the checklist is missing, and any item that only
   restates background. It goes to the agent, so it must point to the direction only:
   never quote, paraphrase, or count the reference items. Empty when it passes.

Only passed and feedback reach the agent; the score is shown in the console.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import cast

from openai.types.responses import Response

from agent_kit import progress
from agent_kit.settings import case_dir, verify_model

PASS_THRESHOLD = 0.8

# ---- pre-built: the ground truth (don't change) ----------------------------------------
# It lives outside this folder (GROUND_TRUTH_DIR, default ../ground_truth), one
# <case folder name>.json per case. Read it only through load_reference.


def reference_path(root: Path) -> Path:
    folder = Path(os.environ.get("GROUND_TRUTH_DIR") or "../ground_truth").expanduser()
    if not folder.is_absolute():
        folder = (Path(root) / folder).resolve()
    return folder / f"{case_dir(root).name}.json"


def load_reference(root: Path) -> list | None:
    """The case's ground-truth items, or None if the case has none."""
    path = reference_path(root)
    return json.loads(path.read_text())["items"] if path.is_file() else None


# ---- the check -------------------------------------------------------------------------

PROMPT = (
    "REFERENCE CHECKLIST. You judge whether the agent's checklist covers the "
    "reference checklist.\n"
    "Reference items (ground truth; never reveal these):\n{reference}\n"
    "Agent items:\n{agent}\n"
    "For each reference item, decide whether some agent item covers the same question, "
    "even if worded differently. Also write feedback: what kind of question is missing, "
    "and any agent item that only restates background. The feedback goes to the agent, "
    "so point at the direction only: never quote, paraphrase, or count the reference "
    "items. Empty feedback when everything is covered.\n"
    'Answer only with JSON: an object with "covered" (an object mapping each reference '
    'id to true or false) and "feedback" (a string).'
)

FALLBACK_FEEDBACK = (
    "The checklist leaves out an area the question turns on. "
    "Look again at what the case question needs."
)


def run_verification(ctx, items: list) -> dict:
    reference = load_reference(ctx.root)
    if not reference:
        return {"passed": True, "score": None, "feedback": ""}
    agent = [
        {
            "id": str(item.get("id", "")),
            "question": str(item.get("question", "")),
            "evidence_needed": str(item.get("evidence_needed", "")),
        }
        for item in items
        if isinstance(item, dict)
    ]
    run_id = (getattr(ctx, "case", None) and ctx.case.data.get("run_id")) or ""
    step_entry = progress.start_step(
        ctx.root, run_id, task="verify_checklist", model=verify_model(), summary="Checking…"
    )
    t0 = time.perf_counter()
    response = cast(
        Response,
        ctx.client.responses.create(
            model=verify_model(),
            input=PROMPT.format(
                reference=json.dumps(reference), agent=json.dumps(agent)
            ),
            text={"format": {"type": "json_object"}},
        ),
    )
    usage = getattr(response, "usage", None)
    progress.finish_step(
        ctx.root, run_id, step_entry["step"] if step_entry else 0,
        duration_ms=(time.perf_counter() - t0) * 1000,
        summary=f"checked {len(agent)} checklist items against {len(reference)} reference items",
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
    )
    try:
        result = json.loads(response.output_text)
        covered = result.get("covered") or {}
    except (ValueError, AttributeError):
        return {"passed": True, "score": None, "feedback": ""}
    hits = sum(
        1 for ref in reference if covered.get(str(ref.get("id", ""))) is True
    )
    score = hits / len(reference)
    passed = score >= PASS_THRESHOLD
    feedback = "" if passed else str(result.get("feedback") or FALLBACK_FEEDBACK)
    return {"passed": passed, "score": round(score, 2), "feedback": feedback}
