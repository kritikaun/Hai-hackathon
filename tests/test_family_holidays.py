"""US public holidays: fixed and floating dates."""

from __future__ import annotations

import unittest

from family_agent.holidays import upcoming_holidays, us_holidays


class HolidaysTest(unittest.TestCase):
    def test_known_2026_dates(self) -> None:
        days = {h["title"]: h["date"] for h in us_holidays(2026)}
        self.assertEqual("2026-11-26", days["Thanksgiving Day"])
        self.assertEqual("2026-12-25", days["Christmas Day"])
        self.assertEqual("2026-01-19", days["Martin Luther King Jr. Day"])
        self.assertEqual("2026-05-25", days["Memorial Day"])
        self.assertEqual("2026-09-07", days["Labor Day"])

    def test_upcoming_window_crosses_new_year(self) -> None:
        days = [h["date"] for h in upcoming_holidays("2026-12-20", 30)]
        self.assertIn("2026-12-25", days)
        self.assertIn("2027-01-01", days)
        self.assertNotIn("2026-11-26", days)


if __name__ == "__main__":
    unittest.main()
