"""Proactive reminders: recipient rules, holiday phases, no repeats."""

from __future__ import annotations

import unittest

from family_agent.reminders import due_reminders
from family_agent.store import FamilyStore, UnknownMemberError
from tests import family_support as f

REF = f.REF_DATE  # the seed day


def titles(result: dict) -> list:
    return [i["title"] for i in result["items"]]


def by_title(result: dict, title: str) -> dict:
    return next(i for i in result["items"] if i["title"] == title)


class ReminderEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root, tmp = f.make_family_root()
        self.addCleanup(tmp.cleanup)
        self.store = FamilyStore(self.root, "riveras")

    def test_birthday_skips_the_birthday_person(self) -> None:
        maya = due_reminders(self.store, "maya", REF, 30)
        item = by_title(maya, "Sofia's 12th birthday")
        self.assertEqual(
            ["maya", "leo", "eli", "elena", "mateo", "carlos", "lucia"],
            item["recipients"],
        )
        sofia = due_reminders(self.store, "sofia", REF, 30)
        self.assertNotIn("Sofia's 12th birthday", titles(sofia))

    def test_anniversary_skips_the_couple(self) -> None:
        eli = due_reminders(self.store, "eli", REF, 30)
        item = by_title(eli, "Leo and Maya's anniversary")
        self.assertEqual(
            ["sofia", "eli", "elena", "mateo", "carlos", "lucia"],
            item["recipients"],
        )
        leo = due_reminders(self.store, "leo", REF, 30)
        self.assertNotIn("Leo and Maya's anniversary", titles(leo))
        maya = due_reminders(self.store, "maya", REF, 30)
        self.assertNotIn("Leo and Maya's anniversary", titles(maya))

    def test_secret_stays_among_those_in_the_know(self) -> None:
        leo = due_reminders(self.store, "leo", REF, 30)
        item = by_title(leo, "Surprise party for Sofia (secret)")
        self.assertEqual(["leo"], item["recipients"])
        maya = due_reminders(self.store, "maya", REF, 30)
        self.assertNotIn("Surprise party for Sofia (secret)", titles(maya))

    def test_shared_celebration_goes_to_everyone(self) -> None:
        self.store.save_memory(
            owner="maya",
            category="shared",
            text="Hosting a thanksgiving party at home.",
            people=["maya", "leo", "sofia", "eli"],
            event_date="2026-11-26",
            event_title="Thanksgiving party",
            event_kind="party",
            confirmed=True,
        )
        eli = due_reminders(self.store, "eli", REF, 30)
        item = by_title(eli, "Thanksgiving party")
        self.assertEqual(
            ["maya", "leo", "sofia", "eli", "elena", "mateo", "carlos", "lucia"],
            item["recipients"],
        )

    def test_due_vs_upcoming_by_distance(self) -> None:
        near = due_reminders(self.store, "maya", "2026-11-05", 30)
        self.assertEqual("due", by_title(near, "Sofia's 12th birthday")["status"])
        eli = due_reminders(self.store, "eli", "2026-11-05", 30)
        self.assertEqual("upcoming", by_title(eli, "Leo and Maya's anniversary")["status"])

    def test_public_holiday_propose_then_remind_phases(self) -> None:
        early = due_reminders(self.store, "eli", REF, 60)
        thanks = by_title(early, "Thanksgiving Day")
        self.assertEqual("upcoming", thanks["status"])
        self.assertTrue(thanks["key"].endswith(":propose"))
        self.assertEqual(
            ["maya", "leo", "sofia", "eli", "elena", "mateo", "carlos", "lucia"],
            thanks["recipients"],
        )
        late = due_reminders(self.store, "eli", "2026-11-24", 60)
        thanks2 = by_title(late, "Thanksgiving Day")
        self.assertEqual("due", thanks2["status"])
        self.assertTrue(thanks2["key"].endswith(":remind"))

    def test_proposed_reminders_are_not_repeated(self) -> None:
        first = due_reminders(self.store, "maya", REF, 30)
        item = by_title(first, "Sofia's 12th birthday")
        self.store.propose_reminder(
            run_id="r",
            requested_by="maya",
            key=item["key"],
            title=item["title"],
            date=item["date"],
            recipients=item["recipients"],
            message=item["suggested_message"],
        )
        second = due_reminders(self.store, "maya", REF, 30)
        self.assertNotIn("Sofia's 12th birthday", titles(second))

    def test_unknown_member_rejected(self) -> None:
        with self.assertRaises(UnknownMemberError):
            due_reminders(self.store, "ghost", REF, 30)


if __name__ == "__main__":
    unittest.main()
