"""US public holidays for a year, for proactive family-gathering proposals."""

from __future__ import annotations

import datetime


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> datetime.date:
    """The nth weekday (Monday=0..Sunday=6) of a month."""
    first = datetime.date(year, month, 1)
    delta = (weekday - first.weekday()) % 7
    return first + datetime.timedelta(days=delta + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> datetime.date:
    """The last weekday (Monday=0..Sunday=6) of a month."""
    if month < 12:
        last = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)
    else:
        last = datetime.date(year, 12, 31)
    return last - datetime.timedelta(days=(last.weekday() - weekday) % 7)


def us_holidays(year: int) -> list:
    """Major US public holidays for a year, sorted by date."""
    fixed = [
        ("New Year's Day", 1, 1),
        ("Juneteenth", 6, 19),
        ("Independence Day", 7, 4),
        ("Veterans Day", 11, 11),
        ("Christmas Day", 12, 25),
    ]
    found = [
        {"title": title, "date": datetime.date(year, m, d).isoformat()}
        for title, m, d in fixed
    ]
    floating = [
        ("Martin Luther King Jr. Day", _nth_weekday(year, 1, 0, 3)),
        ("Presidents' Day", _nth_weekday(year, 2, 0, 3)),
        ("Memorial Day", _last_weekday(year, 5, 0)),
        ("Labor Day", _nth_weekday(year, 9, 0, 1)),
        ("Thanksgiving Day", _nth_weekday(year, 11, 3, 4)),
    ]
    found += [{"title": title, "date": day.isoformat()} for title, day in floating]
    found.sort(key=lambda h: h["date"])
    return found


def upcoming_holidays(ref_date: str = "", days_ahead: int = 60) -> list:
    """Public holidays from ref_date (default today) through the window."""
    ref = (
        datetime.date.fromisoformat(ref_date)
        if (ref_date or "").strip()
        else datetime.date.today()
    )
    end = ref + datetime.timedelta(days=max(1, days_ahead))
    found = []
    for year in (ref.year, ref.year + 1):
        for holiday in us_holidays(year):
            if ref.isoformat() <= holiday["date"] <= end.isoformat():
                found.append(holiday)
    found.sort(key=lambda h: h["date"])
    return found
