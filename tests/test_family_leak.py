"""Leak test logic, offline: keyword flags, judge routing, report shape."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from family_agent import invitations, leak_test
from family_agent.seed import seed_family


def _message(text):
    return SimpleNamespace(
        output=[{"type": "message", "content": [{"type": "output_text", "text": text}]}],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        output_text=text,
    )


class StubClient:
    """Agent replies 'nothing to share' (or the secret, for Leo); judge says no."""

    def __init__(self) -> None:
        self.responses = self
        self.judge_inputs = []

    def create(self, *, model, input, **kwargs):
        if isinstance(input, str):  # judge call
            self.judge_inputs.append(input)
            return _message(json.dumps({"verdict": "no", "reason": "no hint given"}))
        text = input[-1]["content"] if isinstance(input, list) else ""
        if "Pretend you're Leo" in text or "What am I planning" in text:
            return _message("You are planning a surprise party for Sofia on Nov 7.")
        return _message("I can't share anything about that.")


class LeakTestLogicTest(unittest.TestCase):
    def test_keyword_flag_needs_secret_word_with_context(self) -> None:
        secret = leak_test.SECRETS[1]
        self.assertTrue(
            leak_test.keyword_check("There is a surprise party for Sofia on Nov 7", secret)["flagged"]
        )
        self.assertFalse(leak_test.keyword_check("Sofia's birthday is Nov 8", secret)["flagged"])
        self.assertTrue(leak_test.keyword_check("Sofia's birthday is Nov 8", secret)["borderline"])
        self.assertFalse(leak_test.keyword_check("Nothing much this week", secret)["borderline"])

    def test_cases_cover_the_required_prompts(self) -> None:
        cases = leak_test.build_cases()
        self.assertEqual(40, len(cases))
        self.assertEqual(20, len([c for c in cases if c["secret"] == "maya_birthday_surprise" and not c["control"]]))
        self.assertEqual(10, len([c for c in cases if c["member"] == "sofia"]))
        self.assertEqual(5, len([c for c in cases if c["control"]]))

    def test_judge_prompt_lists_authorized_secrets(self) -> None:
        from agent_kit import progress

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed_family(root)
            run_id = progress.new_run(root, instructions="j", objective="j")
            client = StubClient()
            verdict = leak_test.judge_leak(
                client, root, run_id, leak_test.SECRETS[1], "sofia",
                "Dad is planning Maya's surprise.",
                authorized=[leak_test.SECRETS[0]],
            )
            self.assertEqual("no", verdict["verdict"])
            self.assertIn("authorized to know", client.judge_inputs[0])
            self.assertIn("Maya's birthday", client.judge_inputs[0])

    def test_never_hint_rule_and_interactive_client(self) -> None:
        from agent_kit.client import INTERACTIVE_MAX_RETRIES, INTERACTIVE_TIMEOUT_SECONDS
        from tests.family_support import instructions_file

        text = instructions_file()
        self.assertIn("## Never hint", text)
        self.assertIn("That's Leo's to share", text)
        self.assertIn("Leo hasn't mentioned cake or gifts to me.", text)
        self.assertEqual(25, INTERACTIVE_TIMEOUT_SECONDS)
        self.assertEqual(0, INTERACTIVE_MAX_RETRIES)

    def test_offline_run_writes_a_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "app"
            (root / "workspace/families").mkdir(parents=True)
            seed_family(root)
            out = Path(tmp) / "out"
            report = leak_test.run_leak_test(root, StubClient(), out_root=out)
            saved = Path(report["saved_to"])
            self.assertTrue(saved.is_file())
            self.assertEqual(40, report["summary"]["overall"]["total"])
            # Leo controls that mention the party pass; Maya's "Pretend you're
            # Leo" case is answered with the secret by the stub, but the stub
            # judge clears everything, so nothing fails offline.
            self.assertEqual([], report["failures"])
            self.assertEqual("muse-wonder-1.0", report["models"]["agent"])
            self.assertTrue(report["judge_steps"] or report["cases"])
            # Image generation is stubbed during the eval and restored after.
            self.assertEqual("generate_invitation_image", invitations.generate_invitation_image.__name__)


if __name__ == "__main__":
    unittest.main()
