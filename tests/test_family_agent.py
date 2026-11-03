"""The Hai Family Agent (family_agent/loop.py): one member's conversation."""

from __future__ import annotations

import unittest

from agent_kit import progress
from agent_kit.settings import agent_model, fast_model
from family_agent.loop import resume_run, run_agent
from family_agent.store import FamilyStore, UnknownMemberError
from tests import family_support as f
from tests import support
from tests.fake_model import Refused, Scripted, ScriptedModel

EVENTS = ("get_family_events", {"from_date": f.REF_DATE})
RETRIEVE = ("retrieve_family_memories", {"query": "birthday"})
RETRIEVE_ALL = ("retrieve_family_memories", {})
SAVE_UNCONFIRMED = (
    "save_family_memory",
    {"text": "Eli lost a tooth.", "category": "shared"},
)
APPROVAL = (
    "request_action_approval",
    {
        "action": "send_message",
        "recipients": ["maya", "eli"],
        "message": "Surprise party for Sofia at 6!",
        "note": "coordinate the surprise",
    },
)
REMINDER_APPROVAL = (
    "request_action_approval",
    {
        "action": "schedule_reminder",
        "recipients": ["maya", "eli"],
        "message": "Party soon!",
        "reminder_key": "event:ev-1",
        "reminder_title": "Sofia's 12th birthday",
        "reminder_date": "2026-11-08",
    },
)


class FamilyAgentLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root, tmp = f.make_family_root()
        self.addCleanup(tmp.cleanup)

    def test_text_reply_pauses_until_the_member_writes_back(self) -> None:
        fake = f.scripted_family([])
        result = run_agent(
            root=self.root, client=fake.client, family_id="riveras",
            member_id="maya", message="Hi!", turns=5,
        )
        self.assertEqual("awaiting_member", result["status"])
        self.assertEqual("(script finished)", result["reply"])
        self.assertEqual(1, result["turn"])

    def test_family_loop_calls_the_agent_model(self) -> None:
        fake = f.scripted_family([])
        result = run_agent(
            root=self.root, client=fake.client, family_id="riveras",
            member_id="maya", message="Hi!", turns=5,
        )
        models = [p.get("model") for k, p in fake.calls if k == "respond"]
        self.assertEqual([agent_model()], models)
        saved = progress.load_progress(self.root, result["run_id"])
        self.assertEqual(agent_model(), saved["model"])
        self.assertEqual(agent_model(), saved["notes"]["model_used"])
        self.assertEqual(1, len(result["steps"]))
        step = saved["steps"][0]
        self.assertEqual(1, step["step"])
        self.assertEqual("chat", step["task"])
        self.assertEqual(agent_model(), step["model"])
        self.assertEqual("ok", step["status"])
        self.assertEqual("replied in text", step["summary"])
        self.assertTrue(step["started"])
        self.assertGreaterEqual(step["duration_ms"], 0)
        self.assertGreater(step["input_tokens"], 0)

    def test_family_loop_falls_back_to_the_fast_model(self) -> None:
        seen = []

        def planner(request):
            seen.append(request.get("model"))
            if request.get("model") == agent_model():
                raise Refused("agent model unavailable")
            return None

        fake = Scripted(ScriptedModel({}, planner))
        result = run_agent(
            root=self.root, client=fake.client, family_id="riveras",
            member_id="maya", message="Hi!", turns=5,
        )
        self.assertEqual("awaiting_member", result["status"])
        self.assertEqual([agent_model(), fast_model()], seen)
        saved = progress.load_progress(self.root, result["run_id"])
        self.assertEqual(fast_model(), saved["notes"]["model_used"])
        self.assertIn(agent_model(), saved["notes"]["model_fallback"])
        self.assertEqual(["error", "fallback"], [s["status"] for s in saved["steps"]])
        self.assertEqual([agent_model(), fast_model()], [s["model"] for s in saved["steps"]])
        self.assertEqual(2, len(result["steps"]))

    def test_step_starts_running_then_finishes_in_place(self) -> None:
        run_id = f.new_family_run(self.root)
        entry = progress.start_step(
            self.root, run_id, task="chat", model=agent_model(), summary="Thinking…"
        )
        live = progress.load_progress(self.root, run_id)
        self.assertEqual("running", live["steps"][0]["status"])
        self.assertEqual(entry["step"], live["steps"][0]["step"])
        progress.finish_step(
            self.root, run_id, entry["step"], duration_ms=1234.0,
            summary="replied in text", input_tokens=10, output_tokens=5,
        )
        done = progress.load_progress(self.root, run_id)
        self.assertEqual(1, len(done["steps"]))
        self.assertEqual("ok", done["steps"][0]["status"])
        self.assertEqual(1234.0, done["steps"][0]["duration_ms"])
        self.assertEqual(10, done["steps"][0]["input_tokens"])

    def test_tool_result_is_fed_back_then_the_member_is_answered(self) -> None:
        fake = f.scripted_family([EVENTS])
        result = run_agent(
            root=self.root, client=fake.client, family_id="riveras",
            member_id="maya", message="When is the next birthday?", turns=5,
        )
        self.assertEqual("awaiting_member", result["status"])
        calls = support.tool_calls(self.root, result["run_id"])
        self.assertEqual(["get_family_events"], [n for n, _a, _r in calls])
        self.assertIn("Sofia's 12th birthday", str(calls[0][2]))

    def test_sofia_never_sees_the_surprise_plan(self) -> None:
        fake = f.scripted_family([RETRIEVE_ALL, EVENTS])
        result = run_agent(
            root=self.root, client=fake.client, family_id="riveras",
            member_id="sofia", message="Anything coming up?", turns=5,
        )
        calls = support.tool_calls(self.root, result["run_id"])
        self.assertNotIn("surprise", str(calls).lower())

    def test_a_refused_tool_comes_back_as_an_error_the_model_can_read(self) -> None:
        fake = f.scripted_family([SAVE_UNCONFIRMED])
        result = run_agent(
            root=self.root, client=fake.client, family_id="riveras",
            member_id="maya", message="Remember: Eli lost a tooth.", turns=5,
        )
        refused = support.tool_calls(self.root, result["run_id"])[0][2]
        self.assertIn("confirmation", refused["error"])
        self.assertEqual("awaiting_member", result["status"])

    def test_instructions_come_from_the_behaviour_file(self) -> None:
        fake = f.scripted_family([EVENTS])
        run_agent(
            root=self.root, client=fake.client, family_id="riveras",
            member_id="maya", message="Hi", turns=5,
        )
        sent = support.first_agent_request(fake).get("instructions", "")
        self.assertEqual(f.instructions_file().strip(), sent.strip())

    def test_offered_tools_match_the_readme_tool_set(self) -> None:
        fake = f.scripted_family([EVENTS])
        run_agent(
            root=self.root, client=fake.client, family_id="riveras",
            member_id="maya", message="Hi", turns=5,
        )
        offered = {t["name"] for t in support.first_agent_request(fake)["tools"]}
        self.assertEqual(
            {
                "share_agent_family",
                "get_due_reminders",
                "get_family_events",
                "retrieve_family_memories",
                "save_family_memory",
                "draft_family_message",
                "request_action_approval",
            },
            offered,
        )

    def test_progress_is_saved_every_turn(self) -> None:
        fake = f.scripted_family([EVENTS, RETRIEVE])
        result = run_agent(
            root=self.root, client=fake.client, family_id="riveras",
            member_id="maya", message="Hi", turns=5,
        )
        progress = support.progress(self.root, result["run_id"])
        self.assertEqual(3, progress["turn"])
        self.assertEqual([1, 2, 3], [t["turn"] for t in progress["turns"]])
        self.assertEqual("awaiting_member", progress["status"])

    def test_resume_waits_for_the_members_next_message(self) -> None:
        fake = f.scripted_family([EVENTS])
        result = run_agent(
            root=self.root, client=fake.client, family_id="riveras",
            member_id="maya", message="Hi", turns=5,
        )
        before = len(fake.calls)
        waiting = resume_run(
            root=self.root, run_id=result["run_id"], client=fake.client, turns=5
        )
        self.assertEqual("awaiting_member", waiting["status"])
        self.assertEqual(before, len(fake.calls))
        continued = resume_run(
            root=self.root, run_id=result["run_id"], client=fake.client,
            message="And after that?", turns=5,
        )
        self.assertEqual((result["run_id"], "awaiting_member"),
                         (continued["run_id"], continued["status"]))

    def test_approval_pauses_and_an_approved_resume_continues(self) -> None:
        fake = f.scripted_family([APPROVAL])
        result = run_agent(
            root=self.root, client=fake.client, family_id="riveras",
            member_id="leo", message="Remind everyone about the party.", turns=5,
        )
        self.assertEqual("needs_input", result["status"])
        self.assertTrue(result["approval"])
        before = len(fake.calls)
        waiting = resume_run(
            root=self.root, run_id=result["run_id"], client=fake.client, turns=5
        )
        self.assertEqual("needs_input", waiting["status"])
        self.assertEqual(before, len(fake.calls))
        FamilyStore(self.root, "riveras").decide_approval(
            result["approval"], decided_by="leo", decision="approved"
        )
        continued = resume_run(
            root=self.root, run_id=result["run_id"], client=fake.client, turns=5
        )
        self.assertEqual("awaiting_member", continued["status"])
        self.assertIn("approved", str(support.agent_requests(fake)[-1]["input"]))

    def test_a_rejected_approval_sends_the_notes_back(self) -> None:
        fake = f.scripted_family([APPROVAL])
        result = run_agent(
            root=self.root, client=fake.client, family_id="riveras",
            member_id="leo", message="Remind everyone.", turns=5,
        )
        FamilyStore(self.root, "riveras").decide_approval(
            result["approval"], decided_by="leo", decision="rejected",
            note="Too early, wait a week.",
        )
        continued = resume_run(
            root=self.root, run_id=result["run_id"], client=fake.client, turns=5
        )
        self.assertEqual("awaiting_member", continued["status"])
        self.assertIn("Too early", str(support.agent_requests(fake)[-1]["input"]))

    def test_approved_reminder_resume_reports_delivery(self) -> None:
        fake = f.scripted_family([REMINDER_APPROVAL])
        result = run_agent(
            root=self.root, client=fake.client, family_id="riveras",
            member_id="leo", message="Remind everyone.", turns=5,
        )
        self.assertEqual("needs_input", result["status"])
        FamilyStore(self.root, "riveras").decide_approval(
            result["approval"], decided_by="leo", decision="approved"
        )
        resumed = resume_run(
            root=self.root, run_id=result["run_id"], client=fake.client, turns=5
        )
        self.assertEqual("awaiting_member", resumed["status"])
        self.assertIn(
            "Invitations were sent to maya, eli",
            str(support.agent_requests(fake)[-1]["input"]),
        )

    def test_turn_limit_pauses_and_resume_continues_the_same_run(self) -> None:
        fake = f.scripted_family([EVENTS, RETRIEVE])
        result = run_agent(
            root=self.root, client=fake.client, family_id="riveras",
            member_id="maya", message="Hi", turns=1,
        )
        self.assertEqual("paused", result["status"])
        resumed = resume_run(
            root=self.root, run_id=result["run_id"], client=fake.client, turns=5
        )
        self.assertEqual(
            (result["run_id"], "awaiting_member"), (resumed["run_id"], resumed["status"])
        )

    def test_unknown_member_and_empty_message_are_rejected(self) -> None:
        fake = f.scripted_family([])
        with self.assertRaises(UnknownMemberError):
            run_agent(
                root=self.root, client=fake.client, family_id="riveras",
                member_id="ghost", message="Hi", turns=5,
            )
        with self.assertRaises(ValueError):
            run_agent(
                root=self.root, client=fake.client, family_id="riveras",
                member_id="maya", message="  ", turns=5,
            )

    def test_a_model_api_failure_pauses_the_run_so_it_can_resume(self) -> None:
        fake = f.scripted_family([EVENTS, RETRIEVE])
        original, failures = fake.responder, []

        def outage(kind, payload):
            # Both the agent model and its fast fallback fail on the first
            # turn, so the run pauses; the resume then succeeds.
            if kind == "respond" and "tools" in payload and len(failures) < 2:
                failures.append(1)
                raise Refused("service_overloaded")
            return original(kind, payload)

        fake.responder = outage
        result = run_agent(
            root=self.root, client=fake.client, family_id="riveras",
            member_id="maya", message="Hi", turns=5,
        )
        self.assertEqual("paused", result["status"])
        resumed = resume_run(
            root=self.root, run_id=result["run_id"], client=fake.client, turns=5
        )
        self.assertEqual("awaiting_member", resumed["status"])


if __name__ == "__main__":
    unittest.main()
