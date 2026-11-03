"""The family agent's tools, through the registry: definitions and enforcement."""

from __future__ import annotations

import json
import unittest

from agent_kit import progress
from family_agent import tools
from tests import family_support as f


def call(root, run_id, name, args, client=None):
    return tools.run(
        client,
        root,
        run_id,
        {"type": "function_call", "call_id": "call-1", "name": name,
         "arguments": json.dumps(args)},
    )


class FamilyToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root, tmp = f.make_family_root()
        self.addCleanup(tmp.cleanup)
        self.fake = f.scripted_family([])
        self.maya = f.new_family_run(self.root, member_id="maya")
        self.sofia = f.new_family_run(self.root, member_id="sofia")
        self.leo = f.new_family_run(self.root, member_id="leo")

    def test_roster_lists_and_grows(self) -> None:
        listed = call(self.root, self.maya, "share_agent_family", {})
        self.assertEqual("riveras", listed["family_id"])
        self.assertEqual(8, len(listed["members"]))
        grown = call(
            self.root,
            self.maya,
            "share_agent_family",
            {"members": [{"id": "ava", "name": "Ava Rivera"}]},
        )
        self.assertEqual(9, len(grown["members"]))
        duplicate = call(
            self.root,
            self.maya,
            "share_agent_family",
            {"members": [{"id": "ava", "name": "Ava Again"}]},
        )
        self.assertIn("already a member", duplicate["error"])

    def test_events_exclude_what_the_member_may_not_see(self) -> None:
        sofia = call(
            self.root, self.sofia, "get_family_events", {"from_date": f.REF_DATE}
        )
        titles = [e["title"] for e in sofia["events"]]
        self.assertIn("Sofia's 12th birthday", titles)
        self.assertFalse(any("Surprise" in t for t in titles))
        leo = call(self.root, self.leo, "get_family_events", {"from_date": f.REF_DATE})
        self.assertTrue(any("Surprise" in e["title"] for e in leo["events"]))

    def test_memories_are_scoped_to_the_requesting_member(self) -> None:
        found = call(
            self.root, self.sofia, "retrieve_family_memories", {"query": "birthday"}
        )
        self.assertTrue(found["memories"])
        self.assertNotIn("surprise", json.dumps(found).lower())
        everything = call(self.root, self.sofia, "retrieve_family_memories", {})
        owners = {(m["category"], m["owner"]) for m in everything["memories"]}
        self.assertTrue(owners)
        for category, owner in owners:
            self.assertTrue(category == "shared" or owner == "sofia")

    def test_shared_save_needs_confirmation(self) -> None:
        refused = call(
            self.root,
            self.maya,
            "save_family_memory",
            {"text": "Eli lost a tooth.", "category": "shared"},
        )
        self.assertIn("confirmation", refused["error"])
        saved = call(
            self.root,
            self.maya,
            "save_family_memory",
            {"text": "Eli lost a tooth.", "category": "shared", "confirmed": True},
        )
        self.assertEqual("shared", saved["saved"]["category"])
        private = call(
            self.root,
            self.maya,
            "save_family_memory",
            {"text": "Maya's dentist appointment.", "category": "private"},
        )
        self.assertEqual("private", private["saved"]["category"])

    def test_save_can_authorize_specific_members(self) -> None:
        saved = call(
            self.root,
            self.leo,
            "save_family_memory",
            {
                "text": "Leo's gift idea for Maya.",
                "category": "restricted",
                "shared_with": ["maya"],
            },
        )
        stored = next(
            m
            for m in f.family_store(self.root)["memories"]
            if m["id"] == saved["saved"]["id"]
        )
        self.assertEqual(["maya"], stored["shared_with"])

    def test_draft_refuses_unauthorized_recipients(self) -> None:
        secret = next(
            m
            for m in f.family_store(self.root)["memories"]
            if "surprise" in m["text"].lower()
        )
        bad = call(
            self.root,
            self.leo,
            "draft_family_message",
            {
                "recipients": ["sofia"],
                "subject": "Party",
                "body": "Reminder!",
                "memory_ids": [secret["id"]],
            },
        )
        self.assertIn("not authorized", bad["error"])
        self.assertNotIn("surprise", bad["error"].lower())
        birthday = next(
            m
            for m in f.family_store(self.root)["memories"]
            if "Sofia turns 12" in m["text"]
        )
        good = call(
            self.root,
            self.leo,
            "draft_family_message",
            {
                "recipients": ["maya", "eli"],
                "subject": "Party prep",
                "body": "Reminder!",
                "memory_ids": [birthday["id"]],
            },
        )
        self.assertTrue(good["authorized"])
        own = call(
            self.root,
            self.leo,
            "draft_family_message",
            {
                "recipients": ["leo"],
                "subject": "Party",
                "body": "Reminder!",
                "memory_ids": [secret["id"]],
            },
        )
        self.assertTrue(own["authorized"])
        unknown = call(
            self.root,
            self.leo,
            "draft_family_message",
            {"recipients": ["ghost"], "body": "hi"},
        )
        self.assertIn("not a member", unknown["error"])

    def test_approval_pauses_the_run_for_a_person(self) -> None:
        result = call(
            self.root,
            self.leo,
            "request_action_approval",
            {
                "action": "send_message",
                "recipients": ["maya", "eli"],
                "message": "Party at 6!",
                "note": "coordinate the surprise",
            },
        )
        self.assertEqual("needs_input", result["status"])
        self.assertEqual("pending", result["approval"]["status"])
        pending = f.family_store(self.root)["approvals"]
        self.assertEqual(1, len(pending))
        self.assertEqual(self.leo, pending[0]["run_id"])

    def test_verification_rechecks_recipients_at_approval_time(self) -> None:
        verified = f.new_family_run(self.root, member_id="leo", verification=True)
        secret = next(
            m
            for m in f.family_store(self.root)["memories"]
            if "surprise" in m["text"].lower()
        )
        refused = call(
            self.root,
            verified,
            "request_action_approval",
            {
                "action": "send_message",
                "recipients": ["sofia"],
                "message": "Party at 6!",
                "memory_ids": [secret["id"]],
            },
        )
        self.assertIn("not authorized", refused["error"])
        self.assertEqual([], f.family_store(self.root)["approvals"])

    def test_due_reminders_are_scoped_to_the_member(self) -> None:
        import datetime

        root, tmp = f.make_family_root(today=datetime.date.today())
        self.addCleanup(tmp.cleanup)
        leo = f.new_family_run(root, member_id="leo")
        seen = call(root, leo, "get_due_reminders", {"days_ahead": 30})
        titles = [r["title"] for r in seen["reminders"]]
        self.assertIn("Sofia's 12th birthday", titles)
        self.assertIn("Surprise party for Sofia (secret)", titles)
        sofia = f.new_family_run(root, member_id="sofia")
        unseen = call(root, sofia, "get_due_reminders", {"days_ahead": 30})
        hidden = [r["title"] for r in unseen["reminders"]]
        self.assertNotIn("Sofia's 12th birthday", hidden)
        self.assertNotIn("Surprise party for Sofia (secret)", hidden)

    def test_schedule_reminder_with_key_records_a_reminder(self) -> None:
        args = {
            "action": "schedule_reminder",
            "recipients": ["maya", "eli"],
            "message": "Party soon!",
            "reminder_key": "event:ev-1",
            "reminder_title": "Sofia's 12th birthday",
            "reminder_date": "2026-11-08",
        }
        result = call(self.root, self.leo, "request_action_approval", args)
        self.assertEqual("needs_input", result["status"])
        self.assertTrue(result["reminder"])
        self.assertNotIn("image", f.family_store(self.root)["reminders"][0])
        repeat = call(self.root, self.leo, "request_action_approval", args)
        self.assertIn("already proposed", repeat["error"])

    def test_roster_adds_relations(self) -> None:
        result = call(
            self.root,
            self.maya,
            "share_agent_family",
            {
                "members": [
                    {
                        "id": "ava",
                        "name": "Ava Rivera",
                        "partner": "eli",
                        "parents": ["maya"],
                    }
                ]
            },
        )
        ava = next(m for m in result["members"] if m["id"] == "ava")
        self.assertEqual("eli", ava["partner"])
        self.assertEqual(["maya"], ava["parents"])

    def test_unknown_tool_bad_arguments_and_foreign_runs(self) -> None:
        self.assertIn(
            "unknown tool",
            call(self.root, self.maya, "launch_fireworks", {})["error"],
        )
        broken = tools.run(
            self.fake.client,
            self.root,
            self.maya,
            {"type": "function_call", "call_id": "c", "name": "share_agent_family",
             "arguments": "{oops"},
        )
        self.assertIn("not valid JSON", broken["error"])
        plain = progress.new_run(self.root, instructions="x", objective="y")
        self.assertIn(
            "not a family run", call(self.root, plain, "share_agent_family", {})["error"]
        )
        ghost = f.new_family_run(self.root, member_id="ghost")
        self.assertIn(
            "not a member", call(self.root, ghost, "share_agent_family", {})["error"]
        )


if __name__ == "__main__":
    unittest.main()
