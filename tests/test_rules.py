"""Pure rule tests: no database, no LLM. Same request, different company -> different answer."""
from app import clock, rules

ACME = {"name": "Acme Corp", "cutoff_minutes": 120, "escalation_allowed": True}
GLOBEX = {"name": "Globex Ltd", "cutoff_minutes": 60, "escalation_allowed": False}


def trip(slot, taken=0, capacity=4, direction="drop"):
    return {"trip_date": clock.today(), "slot": slot, "direction": direction,
            "capacity": capacity, "seats_taken": taken}


def codes(policy, now_hhmm, current, target):
    return [v.code for v in rules.check_change(policy, clock.at(clock.today(), now_hhmm), current, target)]


def test_change_well_before_cutoff_is_allowed():
    assert codes(ACME, "15:00", trip("18:30"), trip("20:30")) == []


def test_same_request_differs_by_company():
    # 17:00, drop at 18:30 = 90 minutes away: inside Acme's 120, outside Globex's 60.
    assert codes(ACME, "17:00", trip("18:30"), trip("20:30")) == ["CUTOFF_PASSED"]
    assert codes(GLOBEX, "17:00", trip("18:30"), trip("20:30")) == []


def test_cutoff_is_escalatable_only_where_company_allows():
    now = clock.at(clock.today(), "18:00")
    [acme] = rules.check_change(ACME, now, trip("18:30"), trip("20:30"))
    [globex] = rules.check_change(GLOBEX, now, trip("18:30"), trip("20:30"))
    assert acme.escalatable and not globex.escalatable
    assert "Acme Corp needs changes at least 120 minutes" in acme.message


def test_cutoff_uses_the_earlier_trip_when_moving_earlier():
    # Moving 20:30 -> 18:30 at 17:00: the NEW cab is only 90 min away.
    assert codes(ACME, "17:00", trip("20:30"), trip("18:30")) == ["CUTOFF_PASSED"]


def test_full_cab_is_never_escalatable():
    [v] = rules.check_change(GLOBEX, clock.at(clock.today(), "12:00"),
                             trip("18:30"), trip("21:30", taken=6, capacity=6))
    assert v.code == "CAB_FULL" and not v.escalatable


def test_trip_in_past():
    assert "TRIP_IN_PAST" in codes(ACME, "19:00", trip("20:30"), trip("18:30"))


def test_needs_approval_only():
    cut = rules.Violation("CUTOFF_PASSED", "", True)
    full = rules.Violation("CAB_FULL", "", False)
    assert rules.needs_approval_only([cut])
    assert not rules.needs_approval_only([cut, full])
    assert not rules.needs_approval_only([])


def test_a_booking_whose_cab_left_cannot_be_moved_even_by_a_manager():
    v = rules.check_change(ACME, clock.at(clock.today(), "18:45"), trip("18:30"), trip("20:30"))
    assert "TRIP_IN_PAST" in [x.code for x in v]
    assert not rules.needs_approval_only(v)          # no manager override possible
