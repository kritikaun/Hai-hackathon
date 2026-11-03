"""The demo console API: seed, chat, approvals, run inspection (offline)."""

from __future__ import annotations

import tempfile
import unittest

from family_agent import console
from tests import family_support as f

EVENTS = ("get_family_events", {"from_date": f.REF_DATE})
APPROVAL = (
    "request_action_approval",
    {
        "action": "send_message",
        "recipients": ["maya", "eli"],
        "message": "Surprise party for Sofia at 6!",
    },
)


class FamilyConsoleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root, tmp = f.make_family_root()
        self.addCleanup(tmp.cleanup)

    def test_seed_chat_approvals_and_run_flow(self) -> None:
        with tempfile.TemporaryDirectory() as fresh:
            from pathlib import Path

            seeded = console.api_seed(Path(fresh), {"family": "riveras"})
            self.assertEqual(8, seeded["members"])
        fake = f.scripted_family([EVENTS])
        started = console.api_start(
            self.root, fake.client,
            {"family": "riveras", "member": "sofia", "message": "Hi"},
        )
        self.assertEqual("awaiting_member", started["status"])
        run = console.api_run(
            self.root,
            {"family": "riveras", "member": "sofia", "run_id": started["run_id"]},
        )
        self.assertEqual(
            ("riveras", "sofia", ["get_family_events"]),
            (run["family_id"], run["member_id"], [c["name"] for c in run["calls"]]),
        )
        self.assertTrue(started["steps"])
        self.assertEqual([s["step"] for s in run["steps"]], list(range(1, len(run["steps"]) + 1)))
        self.assertTrue(all(s["task"] == "chat" for s in run["steps"]))
        self.assertTrue(all(s["model"] for s in run["steps"]))
        history = console.api_chat_history(
            self.root,
            {"family": "riveras", "member": "sofia", "run_id": started["run_id"]},
        )
        self.assertEqual(run["steps"], history["steps"])
        live = console.api_run_steps(
            self.root,
            {"family": "riveras", "member": "sofia", "run_id": started["run_id"]},
        )
        self.assertEqual(run["steps"], live["steps"])
        self.assertIn("run_steps", console.HTML)
        self.assertIn("Hai is working", console.HTML)
        self.assertIn("Checked the family calendar", console.HTML)
        self.assertIn("Asked for your OK", console.HTML)
        continued = console.api_message(
            self.root, fake.client,
            {
                "family": "riveras",
                "member": "sofia",
                "run_id": started["run_id"],
                "message": "And after that?",
            },
        )
        self.assertEqual("awaiting_member", continued["status"])
        fake2 = f.scripted_family([APPROVAL])
        proposed = console.api_start(
            self.root, fake2.client,
            {"family": "riveras", "member": "leo", "message": "Remind everyone."},
        )
        self.assertEqual("needs_input", proposed["status"])
        pending = console.api_approvals(
            self.root, {"family": "riveras", "member": "leo"}
        )
        self.assertEqual(1, len(pending["approvals"]))
        with self.assertRaises(console.FamilyError):
            console.api_decide(
                self.root,
                {
                    "family": "riveras",
                    "approval_id": proposed["approval"],
                    "by": "maya",
                    "decision": "approved",
                },
            )
        decided = console.api_decide(
            self.root,
            {
                "family": "riveras",
                "approval_id": proposed["approval"],
                "by": "leo",
                "decision": "approved",
            },
        )
        self.assertEqual("approved", decided["decision"])
        resumed = console.api_message(
            self.root,
            fake2.client,
            {
                "family": "riveras",
                "member": "leo",
                "run_id": proposed["run_id"],
            },
        )
        self.assertEqual("awaiting_member", resumed["status"])

    def test_browse_and_direct_message_apis(self) -> None:
        families = console.api_families(self.root, {})
        self.assertIn("riveras", families["families"])
        members = console.api_members(self.root, {"family": "riveras"})
        self.assertEqual(8, len(members["members"]))
        as_sofia = console.api_memories(
            self.root, {"family": "riveras", "member": "sofia"}
        )
        self.assertTrue(as_sofia["memories"])
        self.assertFalse(
            any("surprise" in m["text"].lower() for m in as_sofia["memories"])
        )
        shared = console.api_memories(
            self.root, {"family": "riveras", "member": "sofia", "category": "shared"}
        )
        self.assertTrue(shared["memories"])
        self.assertTrue(all(m["category"] == "shared" for m in shared["memories"]))
        sent = console.api_dm_send(
            self.root,
            {
                "family": "riveras",
                "sender": "maya",
                "recipient": "sofia",
                "text": "Hi!",
            },
        )
        self.assertEqual("maya", sent["sent"]["sender"])
        thread = console.api_dm_thread(
            self.root, {"family": "riveras", "member": "sofia", "other": "maya"}
        )
        self.assertEqual(["Hi!"], [m["text"] for m in thread["messages"]])

    def test_approved_reminder_delivers_invitation_cards(self) -> None:
        due = console.api_due_reminders(
            self.root,
            {
                "family": "riveras",
                "member": "leo",
                "ref_date": f.REF_DATE,
                "days_ahead": 30,
            },
        )
        item = next(i for i in due["items"] if i["title"] == "Sofia's 12th birthday")
        proposed = console.api_propose_reminder(
            self.root,
            {
                "family": "riveras",
                "by": "leo",
                "key": item["key"],
                "title": item["title"],
                "date": item["date"],
                "recipients": ["maya", "eli"],
                "message": item["suggested_message"],
            },
        )
        with self.assertRaises(console.FamilyError):
            console.api_decide(
                self.root,
                {
                    "family": "riveras",
                    "approval_id": proposed["approval_id"],
                    "by": "maya",
                    "decision": "approved",
                },
            )
        console.api_decide(
            self.root,
            {
                "family": "riveras",
                "approval_id": proposed["approval_id"],
                "by": "leo",
                "decision": "approved",
            },
        )
        thread = console.api_dm_thread(
            self.root, {"family": "riveras", "member": "eli", "other": "leo"}
        )
        cards = [m for m in thread["messages"] if m.get("kind") == "card"]
        self.assertEqual(1, len(cards))
        self.assertEqual("Sofia's 12th birthday", cards[0]["card"]["event_title"])
        self.assertEqual("Sofia Rivera", cards[0]["card"]["for"])
        self.assertEqual("Leo Rivera", cards[0]["card"]["from"])

    def test_due_and_propose_reminder_apis(self) -> None:
        due = console.api_due_reminders(
            self.root,
            {
                "family": "riveras",
                "member": "leo",
                "ref_date": f.REF_DATE,
                "days_ahead": 30,
            },
        )
        item = next(i for i in due["items"] if i["title"] == "Sofia's 12th birthday")
        self.assertEqual(
            ["maya", "leo", "eli", "elena", "mateo", "carlos", "lucia"],
            item["recipients"],
        )
        thanks = next(i for i in due["items"] if i["title"] == "Thanksgiving Day")
        self.assertEqual(
            ["maya", "leo", "sofia", "eli", "elena", "mateo", "carlos", "lucia"],
            thanks["recipients"],
        )
        proposed = console.api_propose_reminder(
            self.root,
            {
                "family": "riveras",
                "by": "leo",
                "key": item["key"],
                "title": item["title"],
                "date": item["date"],
                "recipients": item["recipients"],
                "message": item["suggested_message"],
            },
        )
        self.assertTrue(proposed["reminder_id"])
        self.assertTrue(proposed["approval_id"])
        again = console.api_due_reminders(
            self.root,
            {
                "family": "riveras",
                "member": "leo",
                "ref_date": f.REF_DATE,
                "days_ahead": 30,
            },
        )
        self.assertNotIn(
            "Sofia's 12th birthday", [i["title"] for i in again["items"]]
        )

    def test_conversations_and_chat_history(self) -> None:
        fake = f.scripted_family([("get_family_events", {"from_date": f.REF_DATE})])
        started = console.api_start(
            self.root,
            fake.client,
            {"family": "riveras", "member": "maya", "message": "When is the next birthday?"},
        )
        convos = console.api_conversations(
            self.root, {"family": "riveras", "member": "maya"}
        )
        self.assertEqual(1, len(convos["conversations"]))
        self.assertEqual(started["run_id"], convos["conversations"][0]["run_id"])
        empty = console.api_conversations(
            self.root, {"family": "riveras", "member": "eli"}
        )
        self.assertEqual([], empty["conversations"])
        history = console.api_chat_history(
            self.root,
            {"family": "riveras", "member": "maya", "run_id": started["run_id"]},
        )
        texts = [(t["from"], t["text"]) for t in history["turns"]]
        self.assertIn(("member", "When is the next birthday?"), texts)
        self.assertIn(("system", "called get_family_events"), texts)
        self.assertTrue(any(kind == "agent" for kind, _ in texts))

    def test_member_switch_clears_previous_chat_in_page_js(self) -> None:
        # Regression: selecting a different member must not keep showing (or
        # sending into) the previous member's run/chat in the agent pane.
        self.assertIn("const changed = S.me !== id;", console.HTML)
        self.assertIn("if (changed) newChat();", console.HTML)
        self.assertIn("if (S.chatKey !== key) { S.chatKey = key; newChat(); }", console.HTML)
        self.assertIn("if (S.chatKey !== key) return;", console.HTML)
        self.assertIn("if (cur && (r.conversations || []).some(c => c.run_id === cur)) sel.value = cur;", console.HTML)

    def test_add_member_api(self) -> None:
        added = console.api_add_member(
            self.root,
            {
                "family": "riveras",
                "name": "Ava Rivera",
                "role": "child",
                "partner": "",
                "parents": ["maya", "leo"],
            },
        )
        self.assertEqual("ava_rivera", added["member"]["id"])
        self.assertEqual(["maya", "leo"], added["member"]["parents"])


if __name__ == "__main__":
    unittest.main()
