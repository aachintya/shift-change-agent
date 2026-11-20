"""When things fail: the employee always gets a plain message, never a stack trace,
nothing is half-changed, and the LLM budget is never exceeded."""
import pytest

from app import clock, repo, tools
from app.llm import LLMError
from tests.conftest import Chat, booking, sql

DESK = "transport desk"


def test_llm_outage_is_explained(client, llm):
    r = Chat(client, llm, "G201").say("move my drop", LLMError("429 quota exceeded"))
    assert "couldn't process your message" in r["reply"] and DESK in r["reply"]
    assert r["llm_calls"] == 1


def test_garbage_llm_output_retried_once_then_gives_up(client, llm):
    r = Chat(client, llm, "G201").say("???", "Sure! I'd be happy to help", '{"tool": "delete_everything"}')
    assert r["reply"].startswith("Sorry, I didn't get that")
    assert r["llm_calls"] == 2 <= 4


def test_garbage_then_valid_output_recovers(client, llm):
    r = Chat(client, llm, "G201").say("show cabs", "not json", {"tool": "get_roster"})
    assert "Your upcoming cabs" in r["reply"] and r["llm_calls"] == 2


def test_invalid_time_from_model_is_rejected(client, llm):
    r = Chat(client, llm, "G201").say("drop at 25:99", {"tool": "check_cab_options", "time": "25:99"},
                                      {"tool": "check_cab_options", "time": "25:99"})
    assert r["reply"].startswith("Sorry, I didn't get that")


@pytest.mark.parametrize("fault", ["get_roster", "check_cab_options"])
def test_tool_timeout_is_explained_and_nothing_changes(client, llm, fault):
    tools.FAULTS.add(fault)
    try:
        r = Chat(client, llm, "G201").say("8:30 drop", {"tool": "check_cab_options", "direction": "drop", "time": "20:30"})
    finally:
        tools.FAULTS.discard(fault)
    assert "couldn't reach the roster system" in r["reply"] and "nothing has been changed" in r["reply"]
    assert booking("G201", "drop") == "18:30"


def test_cab_fills_up_between_offer_and_yes(client, llm):
    chat = Chat(client, llm, "G201")
    chat.say("8:30 drop", {"tool": "check_cab_options", "direction": "drop", "time": "20:30"})
    sql("UPDATE trips SET seats_taken = capacity WHERE company_id='globex' AND slot='20:30' AND trip_date=%s",
        clock.today())
    r = chat.say("yes", {"tool": "raise_change_request"})
    assert "filled up while you were deciding" in r["reply"] and "Nothing has changed" in r["reply"]
    assert "7:30 PM" in r["reply"]                      # fresh alternative offered
    assert booking("G201", "drop") == "18:30"


def test_empty_message_gets_plain_error(client, llm):
    chat = Chat(client, llm, "G201")
    r = client.post("/chat", json={"message": ""}, headers=chat.h)
    assert r.status_code == 422 and "between 1 and 500 characters" in r.json()["reply"]


def test_unexpected_crash_shows_no_stack_trace(client, llm, monkeypatch):
    def boom(*a):
        raise RuntimeError("secret internal detail")
    chat = Chat(client, llm, "G201")
    monkeypatch.setattr(repo, "take_notifications", boom)
    resp = client.post("/chat", json={"message": "hi"}, headers=chat.h)
    body = resp.json()
    assert resp.status_code == 500
    assert "Something went wrong on our side (ref" in body["reply"]
    assert "secret internal detail" not in str(body) and "Traceback" not in str(body)


def test_cancelling_several_trips_is_all_or_nothing(client, llm):
    # Review finding: on leave, pickup + drop. If the 2nd cancellation fails,
    # the 1st must not stay cancelled, so "nothing has been changed" is true.
    chat = Chat(client, llm, "E101")
    chat.say("I'm on leave tomorrow", {"tool": "cancel_trip", "day": "tomorrow"})
    drop = sql("SELECT id FROM roster WHERE employee_id = 'E101' AND direction = 'drop'"
               " AND trip_date > %s", clock.today())[0]["id"]
    sql(f"ALTER TABLE change_requests ADD CONSTRAINT test_block CHECK (roster_id <> {drop}) NOT VALID")
    try:
        r = chat.say("yes")
    finally:
        sql("ALTER TABLE change_requests DROP CONSTRAINT test_block")
    assert "nothing has been changed" in r["reply"]
    assert booking("E101", "pickup", 1) == "09:00" and booking("E101", "drop", 1) == "18:30"
    assert sql("SELECT count(*) AS n FROM roster WHERE cancelled_at IS NOT NULL") == [{"n": 0}]


def test_cancel_many_rolls_back_if_any_trip_changed(client):
    rows = sql("SELECT id, trip_id FROM roster WHERE employee_id = 'E101' AND trip_date > %s ORDER BY id",
               clock.today())
    with pytest.raises(repo.AlreadyChanged):
        repo.cancel_many("acme", "E101", [(rows[0]["id"], rows[0]["trip_id"]), (rows[1]["id"], -1)])
    assert sql("SELECT count(*) AS n FROM roster WHERE cancelled_at IS NOT NULL") == [{"n": 0}]
