"""The agent's tools: what the assistant can actually DO.

Every tool takes `who` (the Principal from the JWT) as its first argument. The
LLM never supplies identity: it only picks WHICH tool to run and with which
trip/time; the agent code passes `who` in. So "I am employee G201" in a chat
message has nowhere to go.

Failures: any database problem or timeout becomes a ToolError, which the agent
turns into a plain "nothing was changed, try again" message.
FAULTS lets tests (and the demo) simulate a failing backend, e.g.
FAULTS=check_cab_options makes that tool time out.
"""
import logging
import os
from functools import wraps

import psycopg

from app import clock, repo, rules

log = logging.getLogger(__name__)

FAULTS: set[str] = {f for f in os.getenv("FAULTS", "").split(",") if f}


class ToolError(Exception):
    """A backend call failed. The employee is told; nothing was changed."""


def tool(fn):
    """Decorator: fault injection + turn backend errors into ToolError."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if fn.__name__ in FAULTS:
            raise ToolError(f"{fn.__name__} timed out (simulated)")
        try:
            return fn(*args, **kwargs)
        except (psycopg.Error, TimeoutError) as e:
            log.exception("tool %s failed", fn.__name__)
            raise ToolError(f"{fn.__name__} failed") from e
    return wrapper


@tool
def get_roster(who) -> list[dict]:
    """The employee's upcoming bookings (trips that have not left yet)."""
    now = clock.now()
    return [e for e in repo.roster(who.company_id, who.employee_id, clock.today())
            if clock.at(e["trip_date"], e["slot"]) > now]


@tool
def check_cab_options(who, entry: dict) -> list[dict]:
    """Other cabs on the same day and direction as `entry`, each with the rules
    it would break for this employee's company (empty list = bookable now)."""
    policy = repo.get_company(who.company_id)
    now = clock.now()
    return [
        {**trip, "violations": rules.check_change(policy, now, entry, trip)}
        for trip in repo.trips_on(who.company_id, entry["trip_date"], entry["direction"])
        if trip["id"] != entry["id"]
    ]


@tool
def raise_change_request(who, roster_id: int, to_trip_id: int) -> dict:
    """Try to move the employee's booking to another cab.

    Re-reads everything from the database and re-checks every rule: the option
    the employee saw a minute ago may no longer be valid. Returns a dict whose
    "status" is one of:
      APPLIED       moved now
      PENDING       needs manager approval; request saved (created=False if it already existed)
      REJECTED      breaks a rule nobody can override ("reasons" says which)
      CAB_FULL      the last seat went to someone else just now
      ALREADY_DONE  the booking is already on that cab (e.g. double submit)
      CHANGED       the booking moved to another cab meanwhile ("trip" is where it is now)
      NOT_FOUND     not this employee's booking / not this company's cab
    """
    entry = repo.roster_entry(who.company_id, who.employee_id, roster_id)
    target = repo.trip(who.company_id, to_trip_id)
    if (entry is None or target is None or target["trip_date"] != entry["trip_date"]
            or target["direction"] != entry["direction"]):
        return {"status": "NOT_FOUND"}
    if entry["id"] == target["id"]:
        return {"status": "ALREADY_DONE", "trip": target}

    policy = repo.get_company(who.company_id)
    violations = rules.check_change(policy, clock.now(), entry, target)

    if not violations:
        try:
            request_id = repo.apply_change(who.company_id, who.employee_id, roster_id, entry["id"], target["id"])
        except repo.SeatTaken:
            return {"status": "CAB_FULL", "trip": target}
        except repo.AlreadyChanged:
            # Rolled back: the booking moved under us. Say only what is true NOW.
            now_on = repo.roster_entry(who.company_id, who.employee_id, roster_id)
            if now_on is None:
                return {"status": "NOT_FOUND"}  # cancelled meanwhile
            if now_on["id"] == target["id"]:
                return {"status": "ALREADY_DONE", "trip": target}  # double submit
            return {"status": "CHANGED", "trip": now_on}  # moved somewhere else
        return {"status": "APPLIED", "request_id": request_id, "from": entry, "trip": target}

    if any(v.code == "CAB_FULL" for v in violations):  # filled since it was offered
        return {"status": "CAB_FULL", "trip": target}
    reasons = [v.message for v in violations]
    if rules.needs_approval_only(violations):
        request_id, created = repo.create_pending(
            who.company_id, who.employee_id, roster_id, entry["id"], target["id"], " ".join(reasons))
        return {"status": "PENDING", "request_id": request_id, "created": created,
                "from": entry, "trip": target, "reasons": reasons}
    return {"status": "REJECTED", "reasons": reasons, "trip": target}


@tool
def get_request_status(who) -> dict | None:
    """The employee's most recent change request, with the from/to cab details."""
    req = repo.latest_request(who.company_id, who.employee_id)
    if req is None:
        return None
    to = req["to_trip_id"]  # None for a cancellation
    return {**req,
            "from": repo.trip(who.company_id, req["from_trip_id"]),
            "to": repo.trip(who.company_id, to) if to else None}


@tool
def cancel_trips(who, roster_ids: list[int]) -> dict:
    """Cancel one or more of the employee's bookings (leave, work from home...).

    Allowed any time before the cab leaves: a cancellation only frees a seat,
    so it never needs a manager. Trips whose cab already left are skipped and
    reported; the rest are cancelled together, all or nothing. Returns:
      {"status": "CANCELLED", "cancelled": [...], "too_late": [...]}
      {"status": "TOO_LATE", "too_late": [...]}   every cab has already left
      {"status": "ALREADY_DONE"}                  already cancelled (double submit)
      {"status": "CHANGED"}                       bookings changed meanwhile; nothing cancelled
    """
    entries = [e for rid in roster_ids if (e := repo.roster_entry(who.company_id, who.employee_id, rid))]
    if not entries:
        return {"status": "ALREADY_DONE"}
    now = clock.now()
    too_late = [e for e in entries if clock.at(e["trip_date"], e["slot"]) <= now]
    todo = [e for e in entries if e not in too_late]
    if not todo:
        return {"status": "TOO_LATE", "too_late": too_late}
    try:
        repo.cancel_many(who.company_id, who.employee_id, [(e["roster_id"], e["id"]) for e in todo])
    except repo.AlreadyChanged:  # rolled back: nothing was cancelled by this call
        still_there = any(repo.roster_entry(who.company_id, who.employee_id, e["roster_id"]) for e in todo)
        return {"status": "CHANGED" if still_there else "ALREADY_DONE"}
    return {"status": "CANCELLED", "cancelled": todo, "too_late": too_late}


@tool
def get_policy(who) -> dict:
    """The employee's OWN company's rules (cut-off, seats per cab, override)."""
    return repo.get_company(who.company_id)
