#!/usr/bin/env python3
"""The reviewer's side of the case record (a person, not the agent).

    ./helper review RUN_ID --approve [--notes "..."]
    ./helper review RUN_ID --reject --notes "what to fix"
    ./helper review RUN_ID --answer R1 --notes "reply to the evidence request"

Then resume the same run: ./helper run --resume RUN_ID
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_kit.case_record import CaseError, CaseRecord  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_id")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--approve", action="store_true")
    group.add_argument("--reject", action="store_true")
    group.add_argument("--answer", metavar="REQUEST_ID")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    if not (ROOT / "workspace/case_records" / f"{args.run_id}.json").is_file():
        parser.error(f"no case record for {args.run_id}")
    case = CaseRecord(ROOT, args.run_id)
    try:
        if args.answer:
            result = case.answer_request(
                args.answer, args.notes or "Answered by reviewer."
            )
        else:
            if args.reject and not args.notes:
                parser.error("--reject needs --notes saying what to fix")
            result = case.review_memo(args.approve, args.notes)
    except CaseError as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in ("id", "status", "review", "response")
                if k in result
            },
            indent=2,
        )
    )
    print(f"Now resume the same run: ./helper run --resume {args.run_id}")


if __name__ == "__main__":
    main()
