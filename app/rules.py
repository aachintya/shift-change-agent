"""Business rules, enforced in code (never in the prompt).

Each rule is a small pure function: (policy, now, current_trip, target_trip)
-> Violation | None. No database, no LLM, no clock reads; `now` is passed in.
That makes every rule unit-testable in one line, and adding a rule is:
write one function + append it to RULES.

Which violations can a manager override ("escalatable")?
- Cut-off passed: yes, IF the company allows it. It is a policy; a manager may waive policy.
- Cab full: never. A manager cannot create a seat; we offer another cab instead.
- Trip in the past (either cab already left): never.
"""
from dataclasses import dataclass
from datetime import datetime

from app import clock


@dataclass(frozen=True)
class Violation:
    code: str
    message: str       # plain sentence shown to the employee
    escalatable: bool  # may the transport manager approve it anyway?


def _departs(trip: dict) -> datetime:
    return clock.at(trip["trip_date"], trip["slot"])


def trip_in_past(policy, now, current, target):
    # Neither cab may have left: you can't board a cab that has gone, and a
    # booking whose cab has already gone can't be moved any more.
    if _departs(current) <= now:
        return Violation("TRIP_IN_PAST", f"Your {clock.pretty(current['slot'])} {current['direction']} "
                                         "has already left, so it can't be changed.", False)
    if _departs(target) <= now:
        return Violation("TRIP_IN_PAST", f"The {clock.pretty(target['slot'])} cab has already left.", False)


def cab_full(policy, now, current, target):
    if target["seats_taken"] >= target["capacity"]:
        return Violation("CAB_FULL", f"The {clock.pretty(target['slot'])} cab is full.", False)


def cutoff_passed(policy, now, current, target):
    # Both the trip you leave and the trip you join must be far enough away:
    # the routing team needs notice to re-plan either cab.
    first = min(current, target, key=_departs)
    minutes_left = int((_departs(first) - now).total_seconds() // 60)
    if minutes_left < policy["cutoff_minutes"]:
        return Violation(
            "CUTOFF_PASSED",
            f"{policy['name']} needs changes at least {policy['cutoff_minutes']} minutes "
            f"before the trip, and the {clock.pretty(first['slot'])} {first['direction']} is only "
            f"{max(minutes_left, 0)} minutes away.",
            escalatable=policy["escalation_allowed"],
        )


RULES = [trip_in_past, cab_full, cutoff_passed]


def check_change(policy: dict, now: datetime, current: dict, target: dict) -> list[Violation]:
    """All rules that the move current -> target breaks (empty list = allowed)."""
    return [v for rule in RULES if (v := rule(policy, now, current, target))]


def needs_approval_only(violations: list[Violation]) -> bool:
    """True if something is broken but a manager is allowed to approve all of it."""
    return bool(violations) and all(v.escalatable for v in violations)
