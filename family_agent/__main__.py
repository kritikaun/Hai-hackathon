"""The Hai Family Agent's commands.

    python -m family_agent seed [--family ID]
    python -m family_agent run --family ID --member MID --message "..." [--turns N]
    python -m family_agent resume RUN_ID [--message "..."] [--turns N]
    python -m family_agent review --family ID [--list | --approve AP | --reject AP --by MID --notes "..."]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_kit.client import (
    INTERACTIVE_MAX_RETRIES,
    INTERACTIVE_TIMEOUT_SECONDS,
    MissingKeyError,
    make_client,
)

from .loop import resume_run, run_agent
from .seed import seed_all, seed_family
from .store import FamilyError, FamilyStore

ROOT = Path(__file__).resolve().parents[1]


def _print(result: dict) -> None:
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in (
                    "family_id",
                    "run_id",
                    "status",
                    "turn",
                    "reply",
                    "approval",
                    "note",
                    "members",
                    "memories",
                    "events",
                    "approvals",
                    "approval_id",
                    "decision",
                    "seeded",
                    "skipped",
                )
                if result.get(k) is not None
            },
            indent=2,
        )
    )


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    seed = sub.add_parser("seed", help="seed a family workspace with demo data")
    seed.add_argument("--family", default="")

    run = sub.add_parser("run", help="start a member's conversation")
    run.add_argument("--family", required=True)
    run.add_argument("--member", required=True)
    run.add_argument("--message", required=True)
    run.add_argument("--turns", type=int, default=10)

    resume = sub.add_parser("resume", help="continue a stopped conversation")
    resume.add_argument("run_id")
    resume.add_argument("--message", default="")
    resume.add_argument("--turns", type=int, default=10)

    review = sub.add_parser("review", help="list or decide action approvals")
    review.add_argument("--family", required=True)
    group = review.add_mutually_exclusive_group()
    group.add_argument("--list", action="store_true")
    group.add_argument("--approve", metavar="APPROVAL_ID", default="")
    group.add_argument("--reject", metavar="APPROVAL_ID", default="")
    review.add_argument("--by", default="", help="deciding member's id")
    review.add_argument("--notes", default="")

    args = parser.parse_args(argv)
    try:
        if args.command == "seed":
            if args.family:
                _print(seed_family(ROOT, args.family))
            else:
                _print(seed_all(ROOT))
            return
        if args.command == "review":
            store = FamilyStore(ROOT, args.family)
            if args.approve:
                if not args.by:
                    parser.error("--approve needs --by MEMBER")
                approval = store.decide_approval(
                    args.approve, decided_by=args.by, decision="approved"
                )
                _print({"approval_id": approval["id"], "decision": "approved"})
            elif args.reject:
                if not args.by:
                    parser.error("--reject needs --by MEMBER")
                store.decide_approval(
                    args.reject,
                    decided_by=args.by,
                    decision="rejected",
                    note=args.notes,
                )
                _print({"approval_id": args.reject, "decision": "rejected"})
            else:
                pending = store.pending_approvals()
                _print({"family_id": args.family, "approvals": pending})
            return
        try:
            client = make_client(
                timeout_seconds=INTERACTIVE_TIMEOUT_SECONDS,
                max_retries=INTERACTIVE_MAX_RETRIES,
            )
        except MissingKeyError as error:
            parser.error(str(error))
        if args.command == "run":
            _print(
                run_agent(
                    root=ROOT,
                    client=client,
                    family_id=args.family,
                    member_id=args.member,
                    message=args.message,
                    turns=args.turns,
                )
            )
        else:
            _print(
                resume_run(
                    root=ROOT, run_id=args.run_id, client=client,
                    message=args.message, turns=args.turns,
                )
            )
    except FamilyError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
