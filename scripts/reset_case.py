#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_kit.settings import case_dir  # noqa: E402

logger: logging.Logger = logging.getLogger(__name__)


def render_missing_pages(inbox: Path) -> None:
    """Best effort: render page images (pages/<ID>-p<n>.png) for PDFs that have none.

    Page images are used by check_signatures and the console; the agent reads documents
    through the Files API either way. Needs pdftoppm (poppler); skipped if it is missing.
    """
    if shutil.which("pdftoppm") is None:
        return
    pages = inbox / "pages"
    for pdf in sorted(inbox.glob("*.pdf")):
        if any(pages.glob(f"{pdf.stem}-p*.png")):
            continue
        pages.mkdir(exist_ok=True)
        prefix = pages / f".render-{pdf.stem}"
        result = subprocess.run(
            ["pdftoppm", "-r", "100", "-gray", "-png", str(pdf), str(prefix)],
            capture_output=True,
        )
        if result.returncode:
            logger.warning("could not render pages for %s", pdf.name)
            continue
        for image in pages.glob(f".render-{pdf.stem}-*.png"):
            image.rename(pages / f"{pdf.stem}-p{int(image.stem.rsplit('-', 1)[1])}.png")


def reset(*, root: Path = ROOT, clear_runs: bool) -> int:
    initial = case_dir(root) / "documents"
    if not initial.is_dir():
        raise RuntimeError(f"no documents/ folder in the case: {initial.parent}")
    workspace = root / "workspace"
    inbox = workspace / "inbox"
    runs = workspace / "runs"

    if inbox.exists():
        shutil.rmtree(inbox)
    shutil.copytree(initial, inbox)
    render_missing_pages(inbox)
    if clear_runs:
        for directory in (runs, workspace / "case_records"):
            if directory.exists():
                shutil.rmtree(directory)
    runs.mkdir(parents=True, exist_ok=True)
    count = sum(1 for path in inbox.iterdir() if path.is_file())
    logger.info("Seeded %d initial documents", count)
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clear-runs",
        action="store_true",
        help="also remove prior append-only run directories",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    reset(clear_runs=args.clear_runs)


if __name__ == "__main__":
    main()
