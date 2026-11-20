"""Bonus: two people race for the last seat, and one person double-submits.
Exactly one seat is given and exactly one request is created.

Threads call the tool directly (the layer that owns the guarantee), each on its
own pooled DB connection, released together by a Barrier to maximise overlap.
"""
import threading
from datetime import timedelta

from app import clock, tools
from app.auth import Principal
from tests.conftest import sql

ACME = "acme"


def tomorrow_trip(slot):
    return sql("SELECT id, seats_taken, capacity FROM trips WHERE company_id = %s AND direction = 'drop'"
               " AND slot = %s AND trip_date = %s", ACME, slot, clock.today() + timedelta(days=1))[0]


def tomorrow_drop(employee):
    return sql("SELECT r.id FROM roster r JOIN trips t ON t.id = r.trip_id WHERE r.employee_id = %s"
               " AND r.direction = 'drop' AND t.trip_date > %s", employee, clock.today())[0]["id"]


def race(calls):
    """Run all calls at the same instant; return their results."""
    barrier, results = threading.Barrier(len(calls)), [None] * len(calls)

    def run(i, fn):
        barrier.wait()
        results[i] = fn()

    threads = [threading.Thread(target=run, args=(i, fn)) for i, fn in enumerate(calls)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def test_last_seat_goes_to_exactly_one(client):
    last = tomorrow_trip("20:30")                   # 3 of 4 taken: one seat left
    assert last["capacity"] - last["seats_taken"] == 1
    priya, rahul = Principal("E101", ACME, "employee"), Principal("E102", ACME, "employee")
    results = race([
        lambda: tools.raise_change_request(priya, tomorrow_drop("E101"), last["id"]),
        lambda: tools.raise_change_request(rahul, tomorrow_drop("E102"), last["id"]),
    ])
    assert sorted(r["status"] for r in results) == ["APPLIED", "CAB_FULL"]
    assert tomorrow_trip("20:30")["seats_taken"] == 4                       # full, not 5
    assert len(sql("SELECT id FROM change_requests WHERE status = 'APPLIED'")) == 1


def test_double_submit_applies_once(client):
    target = tomorrow_trip("19:30")
    priya = Principal("E101", ACME, "employee")
    roster_id = tomorrow_drop("E101")
    before = {s: tomorrow_trip(s)["seats_taken"] for s in ("18:30", "19:30")}
    results = race([lambda: tools.raise_change_request(priya, roster_id, target["id"])] * 2)
    assert sorted(r["status"] for r in results) == ["ALREADY_DONE", "APPLIED"]
    assert len(sql("SELECT id FROM change_requests")) == 1
    assert tomorrow_trip("19:30")["seats_taken"] == before["19:30"] + 1     # one seat, not two
    assert tomorrow_trip("18:30")["seats_taken"] == before["18:30"] - 1


def test_double_submit_of_approval_request_creates_one(client):
    # Today at 17:00 Acme needs approval, so both submits try to create a PENDING request.
    target = sql("SELECT id FROM trips WHERE company_id = %s AND direction = 'drop' AND slot = '19:30'"
                 " AND trip_date = %s", ACME, clock.today())[0]["id"]
    roster_id = sql("SELECT id FROM roster WHERE employee_id = 'E101' AND direction = 'drop'"
                    " AND trip_date = %s", clock.today())[0]["id"]
    priya = Principal("E101", ACME, "employee")
    results = race([lambda: tools.raise_change_request(priya, roster_id, target)] * 2)
    assert {r["status"] for r in results} == {"PENDING"}
    assert sorted(r["created"] for r in results) == [False, True]
    assert len(sql("SELECT id FROM change_requests")) == 1


def test_double_cancel_frees_exactly_one_seat(client):
    from app import repo
    who = Principal("G201", "globex", "employee")
    entry = next(e for e in tools.get_roster(who) if e["direction"] == "drop")
    before = repo.trip("globex", entry["id"])["seats_taken"]
    results = race([lambda: tools.cancel_trips(who, [entry["roster_id"]])] * 2)
    assert sorted(r["status"] for r in results) == ["ALREADY_DONE", "CANCELLED"]
    assert repo.trip("globex", entry["id"])["seats_taken"] == before - 1


def move_first(monkeypatch, who, roster_id, elsewhere_id):
    """Simulate another request winning the race: just before our write, the same
    booking is moved to `elsewhere`. Our write then finds it no longer where it was."""
    from app import repo
    real = repo.apply_change

    def racing(company_id, employee_id, rid, from_trip, to_trip):
        monkeypatch.setattr(repo, "apply_change", real)  # the other request runs once
        real(company_id, employee_id, rid, from_trip, elsewhere_id)
        return real(company_id, employee_id, rid, from_trip, to_trip)

    monkeypatch.setattr(repo, "apply_change", racing)


def test_conflict_never_confirms_a_cab_the_booking_is_not_on(client, monkeypatch):
    # Regression: the loser of a race used to report ALREADY_DONE for the cab it
    # asked for (20:30) while the booking was really on 19:30.
    priya = Principal("E101", ACME, "employee")
    roster_id = tomorrow_drop("E101")
    asked, actual = tomorrow_trip("20:30"), tomorrow_trip("19:30")
    move_first(monkeypatch, priya, roster_id, actual["id"])

    out = tools.raise_change_request(priya, roster_id, asked["id"])

    assert out["status"] == "CHANGED"
    assert out["trip"]["slot"] == "19:30"                                   # the truth, not the request
    assert tomorrow_trip("20:30")["seats_taken"] == asked["seats_taken"]    # no seat taken on 20:30
    assert len(sql("SELECT id FROM change_requests")) == 1                  # only the winner's


def test_conflict_reply_states_the_real_booking_and_offers_again(client, llm, monkeypatch):
    from tests.conftest import Chat, booking
    from app import repo
    chat = Chat(client, llm, "G201")
    r = chat.say("can I leave later today?", {"tool": "check_cab_options", "direction": "drop", "day": "today"})
    assert "8:30 PM" in r["reply"]
    roster_id = sql("SELECT id FROM roster WHERE employee_id = 'G201' AND direction = 'drop'"
                    " AND trip_date = %s", clock.today())[0]["id"]
    seven_thirty = sql("SELECT id FROM trips WHERE company_id = 'globex' AND direction = 'drop'"
                       " AND slot = '19:30' AND trip_date = %s", clock.today())[0]["id"]
    move_first(monkeypatch, Principal("G201", "globex", "employee"), roster_id, seven_thirty)

    r = chat.say("2")                                                       # picks 8:30 PM

    assert booking("G201", "drop") == "19:30"
    assert "Done" not in r["reply"] and "already on the 8:30" not in r["reply"]
    assert "now on the 7:30 PM cab" in r["reply"]
    assert r["llm_calls"] == 0
