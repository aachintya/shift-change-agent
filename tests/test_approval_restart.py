"""Manager approval: the pause survives a restart and the decision reaches the right employee."""
from app import clock, tools
from app.auth import Principal
from tests.conftest import Chat, booking, headers, sql, stop


def escalate(client, llm, employee="E101", time="19:30"):
    """Acme employee at 17:00 asks for a change inside the cut-off and says yes."""
    chat = Chat(client, llm, employee)
    chat.say(f"move my drop to {time}", {"tool": "check_cab_options", "direction": "drop", "time": time})
    chat.say("yes", {"tool": "raise_change_request"})
    return sql("SELECT id FROM change_requests WHERE employee_id = %s", employee)[0]["id"]


def decide(client, request_id, approve, note=None, manager="M100"):
    return client.post(f"/manager/requests/{request_id}/decision",
                       json={"approve": approve, "note": note}, headers=headers(client, manager))


def notifications(client, employee):
    return client.get("/notifications", headers=headers(client, employee)).json()["notifications"]


def test_approval_resumes_after_restart(client, llm, start_app):
    request_id = escalate(client, llm)

    stop(client)            # service goes down: pool closed, graphs gone from memory
    client = start_app()    # service comes back (same database, no reseed)

    assert [r["id"] for r in client.get("/manager/requests", headers=headers(client, "M100")).json()["requests"]] == [request_id]
    r = decide(client, request_id, approve=True)
    assert r.json()["status"] == "APPROVED"
    assert booking("E101", "drop") == "19:30"
    [note] = notifications(client, "E101")
    assert "approved" in note and "7:30 PM" in note
    assert notifications(client, "E101") == []   # delivered exactly once
    assert notifications(client, "E102") == []   # and only to the right person
    seats = {r["slot"]: r["seats_taken"] for r in sql(
        "SELECT slot, seats_taken FROM trips WHERE company_id='acme' AND direction='drop' AND trip_date=%s",
        clock.today())}
    assert seats["18:30"] == 2 and seats["19:30"] == 3   # seat moved, counts consistent


def test_rejection_keeps_booking_and_tells_employee(client, llm):
    request_id = escalate(client, llm)
    assert decide(client, request_id, approve=False, note="Team dinner is optional").json()["status"] == "REJECTED"
    assert booking("E101", "drop") == "18:30"
    [note] = notifications(client, "E101")
    assert "declined" in note and "Team dinner is optional" in note


def test_deciding_twice_is_refused(client, llm):
    request_id = escalate(client, llm)
    decide(client, request_id, approve=True)
    r = decide(client, request_id, approve=False)
    assert r.status_code == 409 and "already approved" in r.json()["reply"]


def test_cab_fills_while_waiting_for_manager(client, llm):
    request_id = escalate(client, llm, time="20:30")   # Acme 20:30 has exactly 1 free seat
    sql("UPDATE trips SET seats_taken = capacity WHERE company_id='acme' AND slot='20:30' AND trip_date=%s",
        clock.today())                                  # someone else takes it meanwhile
    assert decide(client, request_id, approve=True).json()["status"] == "FAILED"
    assert booking("E101", "drop") == "18:30"
    [note] = notifications(client, "E101")
    assert "filled up" in note and "unchanged" in note


def test_double_submit_creates_one_pending_request(client, llm):
    escalate(client, llm)
    priya = Principal("E101", "acme", "employee")
    first = sql("SELECT roster_id, to_trip_id FROM change_requests")[0]
    again = tools.raise_change_request(priya, first["roster_id"], first["to_trip_id"])
    assert again["status"] == "PENDING" and again["created"] is False
    assert len(sql("SELECT id FROM change_requests")) == 1


def test_decision_and_notification_are_saved_together(client, llm):
    # Review finding: if the notification can't be written, the approval must
    # not be saved either, so the manager can simply retry.
    request_id = escalate(client, llm)
    sql("ALTER TABLE notifications ADD CONSTRAINT test_block CHECK (false) NOT VALID")
    try:
        r = decide(client, request_id, approve=True)
    finally:
        sql("ALTER TABLE notifications DROP CONSTRAINT test_block")
    assert r.status_code == 500                         # plain message, ref in the logs
    assert sql("SELECT status FROM change_requests WHERE id = %s", request_id) == [{"status": "PENDING"}]
    assert booking("E101", "drop") == "18:30"
    r = decide(client, request_id, approve=True)        # the manager retries
    assert r.json()["status"] == "APPROVED" and booking("E101", "drop") == "19:30"
    assert len(notifications(client, "E101")) == 1


def test_two_decisions_at_once_give_one_outcome_and_one_notification(client, llm):
    # Review finding: the losing decision must not announce its own result.
    from app import approval
    from tests.test_concurrency import race
    request_id = escalate(client, llm)
    approve = {"approve": True, "manager_id": "M100", "note": None}
    reject = {"approve": False, "manager_id": "M100", "note": "no"}
    results = race([lambda: approval.finalize("acme", request_id, approve),
                    lambda: approval.finalize("acme", request_id, reject)])
    saved = sql("SELECT status FROM change_requests WHERE id = %s", request_id)[0]["status"]
    assert results == [saved, saved]                    # both callers report what was saved
    assert len(notifications(client, "E101")) == 1


def test_approval_after_the_original_cab_left_fails(client, llm):
    # Review finding: 18:30 -> 20:30 approved at 19:00 must not move anything.
    request_id = escalate(client, llm, time="20:30")
    clock.freeze(clock.at(clock.today(), "19:00"))
    r = decide(client, request_id, approve=True)
    assert r.json()["status"] == "FAILED" and booking("E101", "drop") == "18:30"
    [note] = notifications(client, "E101")
    assert "has already left" in note
