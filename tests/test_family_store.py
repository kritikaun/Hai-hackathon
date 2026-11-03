"""The family-memory store: members, visibility, events, approvals."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from family_agent.seed import seed_all, seed_family
from family_agent.store import (
    FamilyError,
    FamilyStore,
    PermissionDeniedError,
    UnknownMemberError,
)
from tests import family_support as f


class FamilyStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root, tmp = f.make_family_root()
        self.addCleanup(tmp.cleanup)
        self.store = FamilyStore(self.root, "riveras")

    def test_seed_builds_the_demo_family(self) -> None:
        data = f.family_store(self.root)
        self.assertEqual(
            {"maya", "leo", "sofia", "eli", "elena", "mateo", "carlos", "lucia"},
            set(data["members"]),
        )
        self.assertEqual("leo", data["members"]["maya"]["partner"])
        self.assertEqual(["elena", "mateo"], data["members"]["maya"]["parents"])
        self.assertEqual("mother", data["members"]["maya"]["role"])
        self.assertEqual("father", data["members"]["leo"]["role"])
        self.assertEqual("sister", data["members"]["sofia"]["role"])
        self.assertEqual("brother", data["members"]["eli"]["role"])
        self.assertEqual("uncle", data["members"]["carlos"]["role"])
        self.assertEqual("aunt", data["members"]["lucia"]["role"])
        self.assertEqual(["elena", "mateo"], data["members"]["carlos"]["parents"])
        self.assertEqual("lucia", data["members"]["carlos"]["partner"])
        self.assertEqual(6, len(data["memories"]))
        self.assertEqual(3, len(data["events"]))
        self.assertTrue(data["audit"])

    def test_seed_refuses_to_overwrite(self) -> None:
        with self.assertRaises(FamilyError):
            seed_family(self.root, "riveras")

    def test_members_are_validated(self) -> None:
        with self.assertRaises(FamilyError):
            self.store.add_member("maya", "Someone Else")
        with self.assertRaises(FamilyError):
            self.store.add_member("Not Valid", "Nope")
        with self.assertRaises(FamilyError):
            self.store.add_member("ava", "  ")
        ava = self.store.add_member("ava", "Ava Rivera", email="ava@example.com")
        self.assertEqual("ava@example.com", ava["email"])

    def test_unknown_member_is_rejected_everywhere(self) -> None:
        with self.assertRaises(UnknownMemberError):
            self.store.require_member("ghost")
        with self.assertRaises(UnknownMemberError):
            self.store.visible_memories("ghost")
        with self.assertRaises(UnknownMemberError):
            self.store.save_memory(owner="ghost", category="private", text="hi")

    def test_shared_memory_needs_confirmation(self) -> None:
        with self.assertRaises(FamilyError) as raised:
            self.store.save_memory(
                owner="maya", category="shared", text="Eli lost a tooth."
            )
        self.assertIn("confirmation", str(raised.exception))
        memory = self.store.save_memory(
            owner="maya",
            category="shared",
            text="Eli lost a tooth.",
            confirmed=True,
        )
        self.assertEqual("shared", memory["category"])

    def test_private_and_restricted_stay_with_their_owner(self) -> None:
        maya = {m["text"] for m in self.store.visible_memories("maya")}
        self.assertIn("Maya's book club meets on Thursdays.", maya)
        self.assertNotIn("Sofia is learning guitar and practices after school.", maya)
        self.assertFalse(any("surprise" in t.lower() for t in maya))
        sofia = {m["text"] for m in self.store.visible_memories("sofia")}
        self.assertFalse(any("surprise" in t.lower() for t in sofia))
        leo = {m["text"] for m in self.store.visible_memories("leo")}
        self.assertTrue(any("surprise" in t.lower() for t in leo))

    def test_shared_memories_are_visible_to_every_member(self) -> None:
        for member in ("maya", "leo", "sofia", "eli"):
            texts = [m["text"] for m in self.store.visible_memories(member)]
            self.assertTrue(any("Sofia turns 12" in t for t in texts), member)

    def test_search_matches_text_and_people(self) -> None:
        self.assertEqual(
            1, len(self.store.visible_memories("maya", query="book club"))
        )
        self.assertEqual(
            3, len(self.store.visible_memories("leo", query="leo maya"))
        )
        self.assertEqual([], self.store.visible_memories("sofia", query="surprise"))

    def test_get_memory_enforces_authorization(self) -> None:
        secret = next(
            m
            for m in self.store.data["memories"]
            if "surprise" in m["text"].lower()
        )
        with self.assertRaises(PermissionDeniedError):
            self.store.get_memory("sofia", secret["id"])
        self.assertEqual(secret["id"], self.store.get_memory("leo", secret["id"])["id"])
        with self.assertRaises(FamilyError):
            self.store.get_memory("maya", "mem-missing")

    def test_upcoming_events_are_sorted_and_authorized(self) -> None:
        events = self.store.upcoming_events("sofia", ref_date=f.REF_DATE)
        self.assertEqual(
            ["Sofia's 12th birthday", "Leo and Maya's anniversary"],
            [e["title"] for e in events],
        )
        leo_titles = [
            e["title"]
            for e in self.store.upcoming_events("leo", ref_date=f.REF_DATE)
        ]
        self.assertIn("Surprise party for Sofia (secret)", leo_titles)

    def test_restricted_events_reach_shared_with_but_not_honoree(self) -> None:
        self.store.save_memory(
            owner="leo",
            category="restricted",
            text="Planning a surprise for Maya with Sofia and Eli.",
            people=["maya"],
            event_date="2026-11-04",
            event_title="Maya's birthday (surprise)",
            event_kind="birthday",
            shared_with=["sofia", "eli"],
        )
        titles = lambda m: [  # noqa: E731
            e["title"]
            for e in self.store.upcoming_events(m, ref_date=f.REF_DATE)
        ]
        self.assertIn("Maya's birthday (surprise)", titles("sofia"))
        self.assertIn("Maya's birthday (surprise)", titles("eli"))
        self.assertIn("Maya's birthday (surprise)", titles("leo"))
        self.assertNotIn("Maya's birthday (surprise)", titles("maya"))
        self.assertNotIn("Maya's birthday (surprise)", titles("elena"))
        self.assertEqual(
            1, len(self.store.upcoming_events("sofia", ref_date=f.REF_DATE, limit=1))
        )
        self.assertEqual(
            [], self.store.upcoming_events("sofia", ref_date="2027-01-01")
        )

    def test_dates_are_validated(self) -> None:
        with self.assertRaises(FamilyError):
            self.store.save_memory(
                owner="maya", category="private", text="x", event_date="soon"
            )
        with self.assertRaises(FamilyError):
            self.store.add_event(owner="maya", title="x", date="2026-13-01")

    def test_approvals_lifecycle(self) -> None:
        approval = self.store.request_approval(
            run_id="run-1",
            requested_by="leo",
            action="send_message",
            recipients=["maya", "eli"],
            message="Party at 6!",
            note="coordinate the surprise",
        )
        self.assertEqual("pending", approval["status"])
        self.assertEqual(1, len(self.store.pending_approvals("run-1")))
        with self.assertRaises(FamilyError):
            self.store.decide_approval(
                approval["id"], decided_by="maya", decision="approved"
            )
        decided = self.store.decide_approval(
            approval["id"], decided_by="leo", decision="approved"
        )
        self.assertEqual("leo", decided["decided_by"])
        with self.assertRaises(FamilyError):
            self.store.decide_approval(
                approval["id"], decided_by="leo", decision="approved"
            )
        with self.assertRaises(FamilyError):
            self.store.decide_approval(
                "ap-missing", decided_by="maya", decision="approved"
            )

    def test_approvals_validate_recipients_and_action(self) -> None:
        with self.assertRaises(FamilyError):
            self.store.request_approval(
                run_id="r", requested_by="leo", action="send_message", recipients=[]
            )
        with self.assertRaises(UnknownMemberError):
            self.store.request_approval(
                run_id="r",
                requested_by="leo",
                action="send_message",
                recipients=["ghost"],
            )
        with self.assertRaises(FamilyError):
            self.store.request_approval(
                run_id="r", requested_by="leo", action="launch_fireworks"
            )

    def test_owner_can_authorize_specific_members(self) -> None:
        memory = self.store.save_memory(
            owner="leo",
            category="restricted",
            text="Leo's gift idea for Maya.",
            shared_with=["eli"],
        )
        self.assertEqual(["eli"], memory["shared_with"])
        self.assertTrue(
            any(m["id"] == memory["id"] for m in self.store.visible_memories("eli"))
        )
        self.assertFalse(
            any(
                m["id"] == memory["id"]
                for m in self.store.visible_memories("sofia")
            )
        )
        with self.assertRaises(UnknownMemberError):
            self.store.save_memory(
                owner="leo", category="private", text="x", shared_with=["ghost"]
            )

    def test_share_memory_approval_grants_access(self) -> None:
        secret = next(
            m for m in self.store.data["memories"] if "surprise" in m["text"].lower()
        )
        approval = self.store.request_approval(
            run_id="run-1",
            requested_by="leo",
            action="share_memory",
            recipients=["maya", "eli"],
            memory_ids=[secret["id"]],
        )
        self.assertEqual("pending", approval["status"])
        decided = self.store.decide_approval(
            approval["id"], decided_by="leo", decision="approved"
        )
        self.assertEqual(["maya", "eli"], decided["granted"][0]["members"])
        granted = self.store.get_memory("maya", secret["id"])
        self.assertEqual(["maya", "eli"], granted["shared_with"])
        with self.assertRaises(PermissionDeniedError):
            self.store.get_memory("sofia", secret["id"])

    def test_share_memory_is_owner_only(self) -> None:
        secret = next(
            m for m in self.store.data["memories"] if "surprise" in m["text"].lower()
        )
        with self.assertRaises(PermissionDeniedError):
            self.store.request_approval(
                run_id="r",
                requested_by="maya",
                action="share_memory",
                recipients=["eli"],
                memory_ids=[secret["id"]],
            )
        with self.assertRaises(FamilyError):
            self.store.request_approval(
                run_id="r",
                requested_by="leo",
                action="share_memory",
                recipients=["eli"],
                memory_ids=[],
            )
        with self.assertRaises(FamilyError):
            self.store.request_approval(
                run_id="r",
                requested_by="leo",
                action="share_memory",
                recipients=["eli"],
                memory_ids=["mem-missing"],
            )

    def test_direct_messages_between_members(self) -> None:
        sent = self.store.send_direct(
            sender="maya", recipient="sofia", text="Happy early birthday!"
        )
        self.assertEqual("maya", sent["sender"])
        self.assertEqual("sofia", sent["recipient"])
        self.store.send_direct(sender="sofia", recipient="maya", text="Thank you!")
        thread = self.store.thread("maya", "sofia")
        self.assertEqual(["maya", "sofia"], [m["sender"] for m in thread])
        self.assertEqual([], self.store.thread("maya", "eli"))
        with self.assertRaises(FamilyError):
            self.store.send_direct(sender="maya", recipient="sofia", text="  ")
        with self.assertRaises(UnknownMemberError):
            self.store.send_direct(sender="maya", recipient="ghost", text="hi")

    def test_propose_reminder_links_reminder_and_approval(self) -> None:
        proposed = self.store.propose_reminder(
            run_id="run-1",
            requested_by="leo",
            key="event:ev-1",
            title="Sofia's 12th birthday",
            date="2026-11-08",
            recipients=["maya", "eli"],
            message="Party soon!",
        )
        reminder, approval = proposed["reminder"], proposed["approval"]
        self.assertEqual("proposed", reminder["status"])
        self.assertEqual("schedule_reminder", approval["action"])
        self.assertEqual("pending", approval["status"])
        self.assertEqual(reminder["id"], approval["reminder_id"])
        with self.assertRaises(FamilyError):
            self.store.propose_reminder(
                run_id="run-2",
                requested_by="leo",
                key="event:ev-1",
                title="x",
                date="2026-11-08",
                recipients=["maya"],
            )
        decided = self.store.decide_approval(
            approval["id"], decided_by="leo", decision="approved"
        )
        self.assertEqual("approved", decided["status"])
        self.assertEqual("approved", self.store.reminders()[0]["status"])

    def test_propose_reminder_validates(self) -> None:
        with self.assertRaises(FamilyError):
            self.store.propose_reminder(
                run_id="r",
                requested_by="leo",
                key="",
                title="t",
                date="2026-11-08",
                recipients=["maya"],
            )
        with self.assertRaises(UnknownMemberError):
            self.store.propose_reminder(
                run_id="r",
                requested_by="leo",
                key="k",
                title="t",
                date="2026-11-08",
                recipients=["ghost"],
            )
        with self.assertRaises(FamilyError):
            self.store.propose_reminder(
                run_id="r",
                requested_by="leo",
                key="k",
                title="t",
                date="soon",
                recipients=["maya"],
            )

    def test_seed_all_plants_one_demo_family(self) -> None:
        with tempfile.TemporaryDirectory() as fresh:
            root = Path(fresh)
            first = seed_all(root)
            self.assertEqual(1, len(first["seeded"]))
            self.assertEqual([], first["skipped"])
            second = seed_all(root)
            self.assertEqual([], second["seeded"])
            self.assertEqual(["riveras"], second["skipped"])

    def test_approving_reminder_delivers_cards_rejecting_delivers_none(
        self,
    ) -> None:
        proposed = self.store.propose_reminder(
            run_id="r",
            requested_by="leo",
            key="event:ev-missing",
            title="Sofia's 12th birthday",
            date="2026-11-08",
            recipients=["maya", "eli"],
            message="Party!",
        )
        decided = self.store.decide_approval(
            proposed["approval"]["id"], decided_by="leo", decision="approved"
        )
        self.assertEqual(2, len(decided["delivered"]))
        thread = self.store.thread("eli", "leo")
        self.assertEqual(1, len(thread))
        self.assertEqual("card", thread[0]["kind"])
        self.assertEqual("Sofia's 12th birthday", thread[0]["card"]["event_title"])
        second = self.store.propose_reminder(
            run_id="r",
            requested_by="leo",
            key="event:ev-other",
            title="Other",
            date="2026-11-08",
            recipients=["sofia"],
            message="x",
        )
        self.store.decide_approval(
            second["approval"]["id"], decided_by="leo", decision="rejected", note="no"
        )
        self.assertEqual([], self.store.thread("sofia", "leo"))

    def test_set_relations_validates(self) -> None:
        maya = self.store.set_relations("maya", partner="leo", parents=["elena"])
        self.assertEqual("leo", maya["partner"])
        self.assertEqual(["elena"], maya["parents"])
        with self.assertRaises(FamilyError):
            self.store.set_relations("maya", partner="maya")
        with self.assertRaises(FamilyError):
            self.store.set_relations("maya", parents=["maya"])
        with self.assertRaises(UnknownMemberError):
            self.store.set_relations("maya", partner="ghost")
        with self.assertRaises(UnknownMemberError):
            self.store.set_relations("maya", parents=["ghost"])

    def test_card_messages_round_trip(self) -> None:
        sent = self.store.send_direct(
            sender="maya",
            recipient="eli",
            text="Join us!",
            kind="card",
            card={
                "event_title": "Sofia's 12th birthday",
                "event_date": "2026-11-08",
            },
        )
        self.assertEqual("card", sent["kind"])
        self.assertEqual("Sofia's 12th birthday", sent["card"]["event_title"])
        with self.assertRaises(FamilyError):
            self.store.send_direct(
                sender="maya", recipient="eli", text="x", kind="card", card={}
            )
        with self.assertRaises(FamilyError):
            self.store.send_direct(
                sender="maya", recipient="eli", text="x", kind="pigeon"
            )

    def test_families_are_isolated(self) -> None:
        seed_family(self.root, "garcias")
        garcias = FamilyStore(self.root, "garcias")
        garcias.add_member("ana", "Ana Garcia")
        with self.assertRaises(UnknownMemberError):
            garcias.require_member("maya")
        with self.assertRaises(UnknownMemberError):
            self.store.require_member("ana")
        self.assertEqual([], garcias.visible_memories("ana"))
        garcias.save_memory(owner="ana", category="private", text="Ana's note.")
        self.assertFalse(
            any("Ana's note" in m["text"] for m in self.store.visible_memories("maya"))
        )


if __name__ == "__main__":
    unittest.main()
