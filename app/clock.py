"""The single source of "now".

Every time comparison in the system (cut-off, "is this trip in the past") goes
through now(). That makes rules testable: tests freeze the clock instead of
waiting for real time to pass.
"""
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from app import config

IST = ZoneInfo("Asia/Kolkata")

_frozen: datetime | None = None


def freeze(at: datetime | None) -> None:
    """Freeze the clock (tests). Pass None to unfreeze."""
    global _frozen
    _frozen = at


def now() -> datetime:
    if _frozen is not None:
        return _frozen
    if config.DEMO_TIME:  # "17:00" -> today at 17:00 IST
        hh, mm = map(int, config.DEMO_TIME.split(":"))
        return datetime.combine(datetime.now(IST).date(), time(hh, mm), tzinfo=IST)
    return datetime.now(IST)


def today() -> date:
    return now().date()


def day_to_date(day: str) -> date:
    """'today' / 'tomorrow' -> a calendar date (the LLM only gives us the word)."""
    return today() + timedelta(days=1 if day == "tomorrow" else 0)


def pretty(hhmm: str) -> str:
    """'20:30' -> '8:30 PM' (what employees see)."""
    hh, mm = map(int, hhmm.split(":"))
    return f"{(hh - 1) % 12 + 1}:{mm:02d} {'AM' if hh < 12 else 'PM'}"


def at(d: date, hhmm: str) -> datetime:
    """Combine a date and 'HH:MM' into an IST datetime."""
    hh, mm = map(int, hhmm.split(":"))
    return datetime.combine(d, time(hh, mm), tzinfo=IST)
