#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_kit.settings import case_dir  # noqa: E402

logger: logging.Logger = logging.getLogger(__name__)
DOCUMENT_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight(*, root: Path = ROOT, expect_release: str) -> tuple[int, int]:
    """Check the case folder, the inbox, and the trace schema before tests or a run.

    The inbox must hold every case document unchanged; with expect_release="initial"
    it must hold nothing else (released follow-up documents are allowed otherwise).
    """
    case = case_dir(root)
    if not (case / "objective.md").is_file():
        raise RuntimeError(f"objective.md is missing from the case: {case}")
    folder = case / "documents"
    case_docs = (
        {
            p.name: p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in DOCUMENT_SUFFIXES
        }
        if folder.is_dir()
        else {}
    )
    if not case_docs:
        raise RuntimeError(f"no documents (PDF or image) in {folder}")

    inbox = root / "workspace/inbox"
    actual: set[str] = set()
    if inbox.is_dir():
        actual = {
            p.name
            for p in inbox.iterdir()
            if p.is_file() and p.suffix.lower() in DOCUMENT_SUFFIXES
        }
    missing = sorted(set(case_docs) - actual)
    if missing:
        raise RuntimeError(
            f"inbox is missing case documents (run ./helper setup): {missing}"
        )
    changed = [n for n, p in case_docs.items() if sha256(p) != sha256(inbox / n)]
    if changed:
        raise RuntimeError(f"inbox copies differ from the case documents: {changed}")
    extra = sorted(actual - set(case_docs))
    if expect_release == "initial" and extra:
        raise RuntimeError(f"inbox has documents that are not in the case: {extra}")

    try:
        import openai  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "the OpenAI Python SDK is not installed: run ./helper setup"
        ) from None
    logger.info(
        "Preflight passed: %d case documents, %d in inbox", len(case_docs), len(actual)
    )
    return len(case_docs), len(actual)


def check_key() -> None:
    """Does MODEL_API_KEY work, and can it use the agent's models? (./helper setup)"""
    import openai
    from agent_kit import settings
    from agent_kit.client import make_client, MissingKeyError

    try:
        client = make_client()
    except MissingKeyError:
        logger.info(
            "Next: add MODEL_API_KEY to agent/.env, then run ./helper setup again."
        )
        return
    try:
        models = {m.id for m in client.models.list().data}
    except openai.APIError as error:
        raise RuntimeError(
            f"Model API unreachable: {error}. Check MODEL_API_KEY in agent/.env."
        ) from None
    missing = [m for m in (settings.model(), settings.sam_model()) if m not in models]
    if missing:
        raise RuntimeError(f"MODEL_API_KEY works but cannot use: {', '.join(missing)}")
    logger.info(
        "MODEL_API_KEY works: %s and %s are available.",
        settings.model(),
        settings.sam_model(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-release", choices=("initial", "auto"), default="auto")
    parser.add_argument(
        "--check-key",
        action="store_true",
        help="also check MODEL_API_KEY (./helper setup)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    preflight(expect_release=args.expect_release)
    if args.check_key:
        check_key()


if __name__ == "__main__":
    main()
