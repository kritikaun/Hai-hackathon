from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from agent_kit import document_text
from scripts import preflight, reset_case


def write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def make_environment(root: Path) -> None:
    """A case is just objective.md and a documents/ folder; follow-ups live elsewhere."""
    write(root / "case/objective.md", b"# A small case\n\nWhat happened?")
    write(root / "case/documents/INITIAL.pdf", b"initial document")
    write(root / "case/documents/PHOTO.jpg", b"photo")
    write(root / "follow_up/FOLLOW_UP_A.pdf", b"follow-up document A")
    os.environ["CASE_DIR"] = str(root / "case")


class EnvironmentTest(unittest.TestCase):
    def test_reset_seeds_case_documents_and_preserves_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_environment(root)
            write(root / "workspace/inbox/STALE.pdf", b"stale")
            write(root / "workspace/runs/kept/events.jsonl", b"event")

            count = reset_case.reset(root=root, clear_runs=False)

            self.assertEqual(2, count)
            self.assertEqual(
                {"INITIAL.pdf", "PHOTO.jpg"},
                {p.name for p in (root / "workspace/inbox").iterdir() if p.is_file()},
            )
            self.assertTrue((root / "workspace/runs/kept/events.jsonl").is_file())

    def test_documents_are_listed_from_the_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_environment(root)
            reset_case.reset(root=root, clear_runs=False)
            listed = {d["id"]: d for d in document_text.list_documents(root)}
            self.assertEqual({"INITIAL", "PHOTO"}, set(listed))
            self.assertEqual("pdf", listed["INITIAL"]["modality"])
            self.assertEqual("image", listed["PHOTO"]["modality"])
            self.assertEqual(1, listed["PHOTO"]["pages"])

    def test_preflight_accepts_case_documents_and_released_extras(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_environment(root)
            reset_case.reset(root=root, clear_runs=False)

            self.assertEqual(
                (2, 2), preflight.preflight(root=root, expect_release="initial")
            )
            shutil.copy2(
                root / "follow_up/FOLLOW_UP_A.pdf",
                root / "workspace/inbox/FOLLOW_UP_A.pdf",
            )
            self.assertEqual(
                (2, 3), preflight.preflight(root=root, expect_release="auto")
            )
            with self.assertRaises(RuntimeError):
                preflight.preflight(root=root, expect_release="initial")

    def test_preflight_rejects_missing_or_changed_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_environment(root)
            reset_case.reset(root=root, clear_runs=False)
            (root / "workspace/inbox/INITIAL.pdf").write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                preflight.preflight(root=root, expect_release="auto")
            (root / "workspace/inbox/INITIAL.pdf").unlink()
            with self.assertRaises(RuntimeError):
                preflight.preflight(root=root, expect_release="auto")
