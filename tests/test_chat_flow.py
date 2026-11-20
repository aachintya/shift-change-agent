"""Conversation behaviour: clarifying questions, follow-ups, cut-off rules, confirmation."""
from app import clock
from tests.conftest import Chat, booking, sql

OPTIONS = {"tool": "check_cab_options"}
YES = {"tool": "raise_change_request"}


def requests():
    return sql("SELECT status FROM change_requests")


def test_vague_request_gets_options_not_a_guess(client, llm):
    r = Chat(client, llm, "G201").say("can I go a bit late?", {**OPTIONS, "direction": "drop"})
    assert "7:30 PM" in r["reply"] and "8:30 PM" in r["reply"]
    assert "9:30 PM" not in r["reply"]           # full cab is never offered
    assert booking("G201", "drop") == "18:30"    # nothing changed
    assert requests() == []


def test_follow_up_uses_earlier_context_then_applies(client, llm):
    chat = Chat(client, llm, "G201")
    chat.say("can I go a bit late?", {**OPTIONS, "direction": "drop"})
    # "make it 8:30 instead" names no trip: the model gives only a time, and
    # the agent's remembered focus fills in "today's drop".
    r = chat.say("make it 8:30 instead", {**OPTIONS, "time": "20:30"})
    assert "Shall I move your drop today at 6:30 PM to 8:30 PM" in r["reply"]
    assert "Employee: can I go a bit late?" in llm.prompts[-1]  # history reached the LLM
    r = chat.say("yes", YES)
    assert r["reply"].startswith("Done.") and "8:30 PM" in r["reply"]
    assert booking("G201", "drop") == "20:30"
    assert requests() == [{"status": "APPLIED"}]


def test_time_without_a_cab_offers_nearest(client, llm):
    r = Chat(client, llm, "G201").say("make it 8", {**OPTIONS, "direction": "drop", "time": "20:00"})
    assert "There is no cab at 8:00 PM" in r["reply"] and "7:30 PM" in r["reply"]


def test_full_cab_explained_and_alternatives_offered(client, llm):
    r = Chat(client, llm, "G201").say("9:30 drop", {**OPTIONS, "direction": "drop", "time": "21:30"})
    assert "The 9:30 PM cab is full" in r["reply"] and "8:30 PM" in r["reply"]


def test_cutoff_rejection_globex_cannot_escalate(client, llm):
    clock.freeze(clock.at(clock.today(), "17:45"))  # Globex 18:30 drop now 45 min away (< 60)
    chat = Chat(client, llm, "G201")
    r = chat.say("move my drop to 8:30", {**OPTIONS, "direction": "drop", "time": "20:30"})
    assert "Globex Ltd needs changes at least 60 minutes" in r["reply"]
    assert "manager" not in r["reply"].lower()   # Globex does not allow an override
    assert booking("G201", "drop") == "18:30" and requests() == []


def test_cutoff_acme_offers_escalation_and_creates_pending(client, llm):
    chat = Chat(client, llm, "E101")  # 17:00, Acme 18:30 drop is 90 min away (< 120)
    r = chat.say("move my drop to 7:30 pm", {**OPTIONS, "direction": "drop", "time": "19:30"})
    assert "Acme Corp needs changes at least 120 minutes" in r["reply"]
    assert "Send it? (yes/no)" in r["reply"]
    r = chat.say("yes", YES)
    assert "Sent to your transport manager" in r["reply"]
    assert booking("E101", "drop") == "18:30"    # not applied until approved
    assert requests() == [{"status": "PENDING"}]


def test_ambiguous_trip_gets_a_question(client, llm):
    r = Chat(client, llm, "E101").say("change my cab tomorrow", {**OPTIONS, "day": "tomorrow"})
    assert r["reply"].startswith("Which trip do you mean")
    assert "pickup" in r["reply"] and "drop" in r["reply"]


def test_model_asks_for_clarification(client, llm):
    r = Chat(client, llm, "E101").say("cab thing", {"tool": "ask_clarification"})
    assert "Which trip do you mean" in r["reply"]


def test_never_books_without_showing_first(client, llm):
    # Model jumps straight to "book it" on the first message: agent proposes instead.
    r = Chat(client, llm, "G201").say("book 8:30 drop", {**YES, "direction": "drop", "time": "20:30"})
    assert "Shall I move" in r["reply"]
    assert booking("G201", "drop") == "18:30"


def test_saying_no_changes_nothing(client, llm):
    chat = Chat(client, llm, "G201")
    chat.say("move drop to 8:30", {**OPTIONS, "direction": "drop", "time": "20:30"})
    r = chat.say("no thanks", {"tool": "none"})
    assert "haven't changed anything" in r["reply"]
    assert booking("G201", "drop") == "18:30"


def test_status_while_pending(client, llm):
    chat = Chat(client, llm, "E101")
    chat.say("drop at 7:30 pm", {**OPTIONS, "direction": "drop", "time": "19:30"})
    chat.say("yes", YES)
    r = chat.say("any update?", {"tool": "get_request_status"})
    assert "with your transport manager since" in r["reply"]


def test_plain_yes_is_read_by_code_not_the_llm(client, llm):
    chat = Chat(client, llm, "G201")
    r = chat.say("move my drop to 8:30", {**OPTIONS, "direction": "drop", "time": "20:30"})
    assert r["suggestions"] == ["Yes", "No"]      # buttons for the UI
    r = chat.say("yes")                            # no scripted model output at all
    assert r["reply"].startswith("Done.") and r["llm_calls"] == 0
    assert booking("G201", "drop") == "20:30"


def test_option_number_is_read_by_code(client, llm):
    chat = Chat(client, llm, "G201")
    r = chat.say("can I go a bit late?", {**OPTIONS, "direction": "drop"})
    assert r["suggestions"] == ["1. 7:30 PM", "2. 8:30 PM"]
    r = chat.say("2. 8:30 PM")                     # what the button sends
    assert r["llm_calls"] == 0 and booking("G201", "drop") == "20:30"


def test_cancel_asks_first_then_frees_the_seat(client, llm):
    seats = lambda: sql("SELECT t.seats_taken FROM roster r JOIN trips t ON t.id = r.trip_id"
                        " WHERE r.employee_id = 'E101' AND r.direction = 'drop' AND r.trip_date = %s",
                        clock.today())[0]["seats_taken"]
    before = seats()
    chat = Chat(client, llm, "E101")
    r = chat.say("cancel my drop today", {"tool": "cancel_trip", "direction": "drop", "day": "today"})
    assert "Shall I cancel your drop today at 6:30 PM" in r["reply"]
    assert requests() == []                        # nothing before "yes"
    r = chat.say("yes")
    assert "is cancelled" in r["reply"]
    assert seats() == before - 1
    assert requests() == [{"status": "CANCELLED"}]
    r = chat.say("show my cabs", {"tool": "get_roster"})
    assert "Drop today" not in r["reply"]


def test_cancel_closes_a_request_waiting_for_the_manager(client, llm):
    chat = Chat(client, llm, "E101")
    chat.say("drop at 7:30 pm", {**OPTIONS, "direction": "drop", "time": "19:30"})
    chat.say("yes")                                # Acme: needs approval -> PENDING
    chat.say("actually cancel my drop today", {"tool": "cancel_trip", "direction": "drop", "day": "today"})
    chat.say("yes")
    assert sql("SELECT status FROM change_requests ORDER BY id") == [{"status": "FAILED"},
                                                                     {"status": "CANCELLED"}]


def test_greeting_uses_first_name(client, llm):
    r = Chat(client, llm, "E101").say("hello!", {"tool": "small_talk"})
    assert r["reply"].startswith("Hi Priya!")


def test_on_leave_cancels_every_trip_that_day_after_one_yes(client, llm):
    chat = Chat(client, llm, "E101")
    r = chat.say("I'm on leave tomorrow", {"tool": "cancel_trip", "day": "tomorrow"})
    assert "pickup tomorrow at 9:00 AM" in r["reply"] and "drop tomorrow at 6:30 PM" in r["reply"]
    chat.say("yes")
    assert requests() == [{"status": "CANCELLED"}, {"status": "CANCELLED"}]
    assert "Drop today" in chat.say("show my cabs", {"tool": "get_roster"})["reply"]  # today untouched


def test_conflicting_follow_up_never_confirms_the_old_proposal(client, llm):
    # Review finding: we asked "cancel today's drop?", the reply is about
    # tomorrow's drop at 8:30. That must become a new proposal, not a yes.
    chat = Chat(client, llm, "G201")
    chat.say("cancel my drop today", {"tool": "cancel_trip", "direction": "drop", "day": "today"})
    r = chat.say("actually move tomorrow's drop to 8:30",
                 {"tool": "raise_change_request", "direction": "drop", "day": "tomorrow", "time": "20:30"})
    assert requests() == []                          # nothing cancelled, nothing moved
    assert booking("G201", "drop") == "18:30"
    assert "Shall I move your drop tomorrow" in r["reply"]


def test_narrowing_a_cancellation_asks_again(client, llm):
    chat = Chat(client, llm, "E101")
    chat.say("I'm on leave tomorrow", {"tool": "cancel_trip", "day": "tomorrow"})
    r = chat.say("only the drop", {"tool": "raise_change_request", "direction": "drop"})
    assert requests() == [] and r["reply"].startswith("Shall I cancel your drop tomorrow")
    assert "pickup" not in r["reply"]
    chat.say("yes")
    assert requests() == [{"status": "CANCELLED"}]
    assert booking("E101", "pickup", 1) == "09:00"   # pickup kept


def test_later_with_no_later_cab_says_so_before_offering_nearest(client, llm):
    # Asha's drop is 8:30 PM; the only later cab (9:30 PM) is full. Regression: the
    # earlier cabs used to be listed as if they answered "later".
    r = Chat(client, llm, "E103").say("can I leave a bit later today?",
                                      {**OPTIONS, "direction": "drop", "day": "today", "shift": "later"})
    assert r["reply"].startswith("There's no later drop cab I can offer today (the 9:30 PM cab can't be booked)")
    assert "nearest I have instead" in r["reply"] and "7:30 PM" in r["reply"]
    assert booking("E103", "drop") == "20:30" and requests() == []


def test_earlier_shows_only_earlier_cabs(client, llm):
    r = Chat(client, llm, "G202").say("can I leave a bit early today?",
                                      {**OPTIONS, "direction": "drop", "day": "today", "shift": "earlier"})
    assert "6:30 PM" in r["reply"] and "8:30 PM" not in r["reply"]
    assert "There's no" not in r["reply"]
