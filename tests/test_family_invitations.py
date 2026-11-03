"""Invitation cards: building, delivery, and image fallback."""

from __future__ import annotations

import base64
import unittest

from family_agent import invitations
from family_agent.store import FamilyStore
from tests import family_support as f


class _Images:
    def __init__(self, payload):
        self.payload = payload

    def generate(self, **kwargs):
        return self.payload


class _StubClient:
    def __init__(self, payload):
        self.images = _Images(payload)


class _Datum:
    def __init__(self, b64_json=None, url=None):
        self.b64_json = b64_json
        self.url = url


class _ImagesResponse:
    def __init__(self, data):
        self.data = data


class InvitationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root, tmp = f.make_family_root()
        self.addCleanup(tmp.cleanup)
        self.store = FamilyStore(self.root, "riveras")

    def test_build_card_names_honoree_and_sender(self) -> None:
        event = next(
            e for e in self.store.data["events"] if "birthday" in e["title"].lower()
        )
        reminder = {
            "id": "rm-1",
            "key": f"event:{event['id']}",
            "title": event["title"],
            "date": event["date"],
            "requested_by": "leo",
            "recipients": ["maya"],
        }
        card = invitations.build_card(self.store, reminder, "maya")
        self.assertEqual("Sofia's 12th birthday", card["event_title"])
        self.assertEqual("Sofia Rivera", card["for"])
        self.assertEqual("Leo Rivera", card["from"])
        self.assertEqual("Maya Rivera", card["to"])
        self.assertEqual("birthday", card["event_kind"])
        self.assertEqual("", card["image"])

    def test_build_card_without_event(self) -> None:
        reminder = {
            "id": "rm-2",
            "key": "holiday:2026-11-26:propose",
            "title": "Thanksgiving Day",
            "date": "2026-11-26",
            "requested_by": "maya",
            "recipients": ["eli"],
        }
        card = invitations.build_card(self.store, reminder, "eli")
        self.assertEqual("public_holiday", card["event_kind"])
        self.assertEqual("", card["for"])
        self.assertIn("Thanksgiving", card["message"])

    def test_deliver_sends_one_card_per_recipient(self) -> None:
        proposed = self.store.propose_reminder(
            run_id="r",
            requested_by="leo",
            key="event:ev-missing",
            title="Sofia's 12th birthday",
            date="2026-11-08",
            recipients=["maya", "eli"],
            message="Party!",
        )
        sent = invitations.deliver_approved_reminder(self.store, proposed["reminder"])
        self.assertEqual(2, len(sent))
        self.assertTrue(all(m["kind"] == "card" for m in sent))
        thread = self.store.thread("eli", "leo")
        self.assertEqual("card", thread[0]["kind"])

    def test_image_generation_falls_back_offline(self) -> None:
        fake = f.scripted_family([])
        self.assertIsNone(
            invitations.generate_invitation_image(fake.client, "Party", "birthday")
        )

    def test_image_generation_reads_b64(self) -> None:
        raw = b"fake-png-bytes"
        payload = _ImagesResponse([_Datum(b64_json=base64.b64encode(raw).decode())])
        self.assertEqual(
            raw,
            invitations.generate_invitation_image(_StubClient(payload), "Party", "birthday"),
        )

    def test_attach_saves_file_and_links_reminder(self) -> None:
        proposed = self.store.propose_reminder(
            run_id="r",
            requested_by="leo",
            key="event:ev-missing",
            title="Sofia's 12th birthday",
            date="2026-11-08",
            recipients=["maya"],
            message="Party!",
        )
        reminder = proposed["reminder"]
        payload = _ImagesResponse([_Datum(b64_json=base64.b64encode(b"img").decode())])
        name = invitations.attach_generated_image(
            self.store, _StubClient(payload), reminder
        )
        self.assertRegex(name, r"^[0-9a-f]{32}\.png$")
        self.assertNotEqual(f"{reminder['id']}.png", name)
        self.assertTrue(
            (self.root / "workspace/families/riveras/images" / name).is_file()
        )
        card = invitations.build_card(self.store, self.store.reminders()[0], "maya")
        self.assertEqual(f"/images/riveras/{name}", card["image"])
        webp = b"RIFF" + b"\x00" * 4 + b"WEBP" + b"data"
        payload = _ImagesResponse([_Datum(b64_json=base64.b64encode(webp).decode())])
        self.assertEqual(
            "webp", invitations.attach_generated_image(self.store, _StubClient(payload), reminder).rsplit(".", 1)[1]
        )


if __name__ == "__main__":
    unittest.main()
