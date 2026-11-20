"""The chat agent, as a LangGraph graph.

    START -> understand --(action.tool)--> one handler node -> finish -> END
                 |                                               ^
                 +------ (LLM failed: reply already set) --------+

How one message is handled:
1. `understand` decides WHAT the employee wants, as an Action:
   - Fast path (no LLM): a plain "yes" / "no" / "2" answering the question we
     just asked. These are the most common replies and the easiest to get
     wrong, so code reads them. 0 LLM calls.
   - Otherwise the LLM reads the message (1 call, +1 repair retry if the output
     is malformed; never more than MAX_LLM_ATTEMPTS = 2, below the limit of 4).
2. One handler (plain Python below) does it: calls tools.py, which applies the
   rules, and builds the reply from replies.py templates.
3. `finish` saves a short history, and the quick-reply buttons for the UI.

Memory: state is checkpointed in Postgres per employee (thread
"chat:<company>:<employee>"), so "make it 8 instead" works across messages and
restarts. `options` + `awaiting` remember the question we asked last.

Identity (`who`) comes from the graph config, which main.py fills from the
JWT. It is not in the state, and never shown to or produced by the LLM.
"""
import logging
import re
import time
from datetime import date
from typing import Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app import clock, replies, repo, rules, tools
from app.config import LLM_BUDGET_SECONDS, LLM_TIMEOUT_SECONDS, MAX_LLM_ATTEMPTS
from app.llm import SYSTEM_PROMPT, Action, LLMError, parse_action

log = logging.getLogger(__name__)

HISTORY_LINES = 6  # the last 3 exchanges are enough to resolve follow-ups


class ChatState(TypedDict, total=False):
    # per message (main.py resets these on every new message)
    message: str
    action: dict | None
    reply: str
    llm_calls: int
    suggestions: list[str]  # quick-reply buttons for the UI
    # remembered across messages (checkpointed)
    history: list[str]
    focus: int | None    # roster_id of the trip being discussed
    options: list[dict]  # what we offered last, numbered from 1
    awaiting: Literal["confirm", "choice"] | None  # the question we asked last


CLEAR = {"options": [], "awaiting": None}  # "no open question any more"


# ----------------------------------------------------------- 1. understanding

YES = re.compile(r"(yes|yeah|yep|yup|y|ok|okay|sure|confirm|go ahead|do it|please do)"
                 r"( please| send it| book it| do it| go ahead| thanks)*[.! ]*", re.I)
NO = re.compile(r"(no|nope|nah|no thanks|not now|leave it)[.! ]*", re.I)
PICK = re.compile(r"(?:option\s*)?(\d)(?:\.\s.*)?", re.I)  # "2", "option 2", "2. 8:30 PM" (button)


def quick_action(state: ChatState) -> dict | None:
    """Short answers to the question we just asked, read without the LLM.
    Anything longer ("no, make it 9 instead") goes to the LLM."""
    msg, awaiting = state["message"].strip(), state.get("awaiting")
    if awaiting and NO.fullmatch(msg):
        return Action(tool="none").model_dump()
    if awaiting == "confirm" and YES.fullmatch(msg):
        return Action(tool="raise_change_request", option=1).model_dump()
    if awaiting == "choice" and (m := PICK.fullmatch(msg)) and 1 <= int(m[1]) <= len(state["options"]):
        return Action(tool="raise_change_request", option=int(m[1])).model_dump()
    return None


def llm_context(state: ChatState, entries: list[dict]) -> str:
    """What the LLM sees: time, THIS employee's bookings, open options, recent chat."""
    bookings = "\n".join(f"- {replies.day(e['trip_date'])} {e['direction']} at {e['slot']}"
                         for e in entries) or "- none"
    offered = "\n".join(f"{i}. {o['slot']}" for i, o in enumerate(state.get("options") or [], 1)) or "none"
    history = "\n".join(state.get("history") or []) or "(start of conversation)"
    return (f"Now: {clock.now():%A %d %b %Y, %H:%M} (IST)\n"
            f"Employee's upcoming bookings:\n{bookings}\n"
            f"Options currently offered to the employee:\n{offered}\n"
            f"Recent conversation:\n{history}\n\n"
            f"Latest message from the employee:\n\"\"\"{state['message']}\"\"\"")


def ask_llm(llm, context: str) -> dict:
    """At most MAX_LLM_ATTEMPTS calls, all inside one LLM_BUDGET_SECONDS budget
    (so a slow first call plus a repair call can't exceed the budget; database
    work and checkpointing come on top, so this bounds LLM time, not the request).
    Returns {"action": ...} or {"reply": ...}."""
    prompt, deadline = context, time.monotonic() + LLM_BUDGET_SECONDS
    for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
        remaining = deadline - time.monotonic()
        if remaining < 0.5:  # no time left for a useful call: answer now
            return {"reply": replies.llm_down(), "llm_calls": attempt - 1}
        try:
            raw = llm.complete(SYSTEM_PROMPT, prompt, timeout=min(LLM_TIMEOUT_SECONDS, remaining))
        except LLMError as e:  # quota / timeout / network: do not retry, answer fast
            log.warning("LLM unavailable: %s", e)
            return {"reply": replies.llm_down(), "llm_calls": attempt}
        try:
            return {"action": parse_action(raw).model_dump(), "llm_calls": attempt}
        except ValueError as e:  # malformed / out-of-schema output: one repair try
            log.info("bad LLM output %r: %s", raw[:200], e)
            prompt = f"{context}\n\nYour previous answer was invalid ({e}). Reply with ONLY the JSON object."
    return {"reply": replies.not_understood(), "llm_calls": MAX_LLM_ATTEMPTS}


# ----------------------------------------------------------- helpers

def matching_trips(state: ChatState, who, action: dict) -> list[dict]:
    """The employee's upcoming bookings that fit what they said (direction/day)."""
    entries = tools.get_roster(who)
    direction, day = action.get("direction"), action.get("day")
    # Follow-up with no new details ("make it 8 instead"): the same trip as before.
    if direction is None and day is None and state.get("focus"):
        same = [e for e in entries if e["roster_id"] == state["focus"]]
        if same:
            return same
    return [e for e in entries
            if (direction is None or e["direction"] == direction)
            and (day is None or e["trip_date"] == clock.day_to_date(day))]


def find_trip(state: ChatState, who, action: dict):
    """Which ONE booking is the employee talking about?
    Returns (entry, None), or (None, reply) when we must ask instead of guessing."""
    matches = matching_trips(state, who, action)
    if not matches:
        return None, replies.no_booking(action.get("direction"), action.get("day"))
    if len({e["direction"] for e in matches}) > 1:  # pickup or drop? we must ask
        return None, replies.which_trip(matches)
    # Same direction on several days: take the next one. Safe, because the
    # proposal names the day and the employee must confirm it.
    return matches[0], None


def offer(who, entry: dict, wanted: str | None, prefix: str = "", shift: str | None = None) -> dict:
    """Propose one cab (if they named a time we can do) or list the best few.

    Options that break a rule nobody may override are never offered. Options
    that need manager approval are offered, clearly marked. `shift` ("later" /
    "earlier") keeps only cabs on that side of the current one; if none is left
    we say so instead of offering cabs the other way.
    """
    if wanted == entry["slot"]:
        return {"reply": prefix + replies.already_on(entry), "focus": entry["roster_id"], **CLEAR}

    candidates = tools.check_cab_options(who, entry)
    offerable = [
        {"kind": "move", **about(entry), "trip_id": c["id"], "slot": c["slot"],
         "cab_no": c["cab_no"], "seats_free": c["capacity"] - c["seats_taken"],
         "approval_reason": " ".join(v.message for v in c["violations"]) or None}
        for c in candidates
        if not c["violations"] or rules.needs_approval_only(c["violations"])
    ]
    if shift:  # they asked for later/earlier: show only that side if we can...
        on_side = (lambda s: s > entry["slot"]) if shift == "later" else (lambda s: s < entry["slot"])
        that_way = [o for o in offerable if on_side(o["slot"])]
        if that_way:
            offerable = that_way
        elif offerable:  # ...otherwise say so plainly, then offer the nearest we have (desk step 3)
            unavailable = [c["slot"] for c in candidates if on_side(c["slot"])]
            prefix += replies.nothing_that_way(entry, shift, unavailable)

    if wanted:
        exact = next((o for o in offerable if o["slot"] == wanted), None)
        if exact:  # exactly what they asked for: propose it and ask to confirm
            return {"reply": prefix + replies.propose(entry, exact), "focus": entry["roster_id"],
                    "options": [exact], "awaiting": "confirm"}
        blocked = next((c for c in candidates if c["slot"] == wanted), None)
        prefix += (replies.blocked(wanted, [v.message for v in blocked["violations"]])
                   if blocked else replies.no_cab_at(wanted))

    # The 3 nearest times (to what they asked for, else to their current cab), in time order.
    anchor = minutes(wanted or entry["slot"])
    best = sorted(sorted(offerable, key=lambda o: abs(minutes(o["slot"]) - anchor))[:3],
                  key=lambda o: o["slot"])
    if not best:
        return {"reply": prefix + replies.no_options(entry), "focus": entry["roster_id"], **CLEAR}
    return {"reply": prefix + replies.options(entry, best), "focus": entry["roster_id"],
            "options": best, "awaiting": "choice"}


def about(entry: dict) -> dict:
    """Which booking an offered option belongs to (stored with the option, so a
    later "yes" can be checked against it)."""
    return {"roster_id": entry["roster_id"], "direction": entry["direction"],
            "trip_date": entry["trip_date"].isoformat()}


def contradicts(options: list[dict], action: dict) -> bool:
    """Is this "acceptance" really about something else? E.g. we asked
    "cancel today's drop?" and the reply is "move tomorrow's drop to 8:30".
    That is a new request, and must never confirm the old proposal."""
    o, direction = options[0], action.get("direction")
    # Every proposed trip must be in that direction: "only the drop", after we
    # offered to cancel pickup AND drop, narrows the request, so we ask again.
    if direction and any(x.get("direction", direction) != direction for x in options):
        return True
    if action.get("day") and o.get("trip_date") and clock.day_to_date(action["day"]).isoformat() != o["trip_date"]:
        return True
    return o.get("kind") == "cancel" and bool(action.get("time"))  # a time means a move, not a cancel


def minutes(hhmm: str) -> int:
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m


def picked(options: list[dict], action: dict) -> dict | None:
    """Which offered option did they accept? By number, by time, or the only one."""
    n = action.get("option")
    if n and 1 <= n <= len(options):
        return options[n - 1]
    if action.get("time"):
        return next((o for o in options if o["slot"] == action["time"]), None)
    return options[0] if len(options) == 1 else None


# ----------------------------------------------------------- 2. handlers
# One per tool the LLM can choose. Each takes (state, who) and returns the
# state update; tools.ToolError is turned into a polite reply by as_node().

def show_roster(state, who):
    return {"reply": replies.roster(tools.get_roster(who))}


def check_cab_options(state, who):
    entry, question = find_trip(state, who, state["action"])
    if question:
        return {"reply": question}
    action = state["action"]
    # A named time wins; "later"/"earlier" only narrows the list when no time is given.
    return offer(who, entry, action.get("time"), shift=None if action.get("time") else action.get("shift"))


def cancel_trip(state, who):
    """Propose cancelling. "I'm on leave tomorrow" (no pickup/drop given) means
    every trip that day; we still ask once before cancelling anything."""
    action = state["action"]
    matches = matching_trips(state, who, action)
    if not matches:
        return {"reply": replies.no_booking(action.get("direction"), action.get("day"))}
    trips = [e for e in matches if e["trip_date"] == matches[0]["trip_date"]]  # the next day only
    return {"reply": replies.propose_cancel(trips), "focus": trips[0]["roster_id"],
            "options": [{"kind": "cancel", **about(e), "slot": e["slot"]} for e in trips],
            "awaiting": "confirm"}


def cancel_all(who, options: list[dict]) -> dict:
    """Cancel every trip we offered to cancel (all or nothing, see tools.cancel_trips)."""
    out = tools.cancel_trips(who, [o["roster_id"] for o in options])
    if out["status"] == "ALREADY_DONE":
        return {"reply": replies.already_done(), **CLEAR}
    if out["status"] == "CHANGED":
        return {"reply": replies.changed_meanwhile(), **CLEAR}
    lines = ([replies.cancelled(e) for e in out.get("cancelled", [])]
             + [replies.too_late_to_cancel(e) for e in out["too_late"]])
    return {"reply": "\n".join(lines), **CLEAR}


def confirm(state, who, start_approval):
    """The employee accepted something we offered: do it (rules re-checked by tools.py)."""
    action, options = state["action"], state.get("options") or []
    if not options:  # never book anything the employee has not been shown
        return check_cab_options(state, who)
    if contradicts(options, action):  # a new request, not a yes: make a fresh proposal
        if not action.get("day") and options[0].get("trip_date"):  # "only the drop": same day as before
            day = replies.day(date.fromisoformat(options[0]["trip_date"]))
            state = {**state, "action": {**action, "day": day}}
        if options[0].get("kind") == "cancel" and not action.get("time"):
            return cancel_trip(state, who)
        return check_cab_options(state, who)
    if options[0].get("kind") == "cancel":  # "yes" to "Shall I cancel ...?"
        return cancel_all(who, options)
    chosen = picked(options, action)
    if chosen is None:
        if action.get("time"):  # a new time, not one of the offered ones
            return check_cab_options(state, who)
        return {"reply": replies.pick_option()}

    out = tools.raise_change_request(who, chosen["roster_id"], chosen["trip_id"])
    status = out["status"]
    if status == "APPLIED":
        return {"reply": replies.applied(out["from"], out["trip"]), **CLEAR}
    if status == "PENDING":
        if not out["created"]:
            return {"reply": replies.already_pending(), **CLEAR}
        start_approval(who.company_id, who.employee_id, out["request_id"])
        return {"reply": replies.sent_for_approval(out["from"], out["trip"]), **CLEAR}
    if status == "ALREADY_DONE":
        return {"reply": replies.already_done(out["trip"]), **CLEAR}
    if status in ("CAB_FULL", "REJECTED", "CHANGED"):  # explain, then offer fresh options
        why = (replies.cab_filled(chosen["slot"]) if status == "CAB_FULL"
               else replies.booking_changed(out["trip"]) if status == "CHANGED"
               else replies.rejected(out["reasons"]))
        entry = next((e for e in tools.get_roster(who) if e["roster_id"] == chosen["roster_id"]), None)
        return offer(who, entry, None, why) if entry else {"reply": why, **CLEAR}
    return {"reply": replies.gone(), **CLEAR}  # NOT_FOUND


def request_status(state, who):
    return {"reply": replies.status(tools.get_request_status(who))}


def policy(state, who):
    return {"reply": replies.policy(tools.get_policy(who))}


def clarify(state, who):
    entries = tools.get_roster(who)
    return {"reply": replies.which_trip(entries) if entries else replies.roster(entries)}


def small_talk(state, who):
    name = repo.get_employee(who.employee_id)["name"].split()[0]  # first name
    thanked = re.search(r"thank|thx|cheers", state["message"], re.I)
    return {"reply": replies.thanks(name) if thanked else replies.greeting(name)}


def nothing(state, who):
    if state.get("awaiting"):  # "no" to our question
        return {"reply": replies.nothing_changed(), **CLEAR}
    if YES.fullmatch(state["message"].strip()) or NO.fullmatch(state["message"].strip()):
        return {"reply": replies.nothing_to_confirm()}  # a stray "yes" after it's already done
    return {"reply": replies.off_topic()}


def suggestions(state: ChatState) -> list[str]:
    """Quick-reply buttons matching the question we just asked."""
    if state.get("awaiting") == "confirm":
        return ["Yes", "No"]
    if state.get("awaiting") == "choice":
        return [f"{i}. {clock.pretty(o['slot'])}" for i, o in enumerate(state["options"], 1)]
    return []


# ----------------------------------------------------------- 3. the graph

def as_node(handler):
    """Adapt a handler to a graph node: inject `who` from the config, and turn a
    failing backend into a plain "nothing changed, try again" reply."""
    def node(state: ChatState, config: RunnableConfig):
        try:
            return handler(state, config["configurable"]["principal"])
        except tools.ToolError:
            log.warning("tool failure in %s", handler.__name__, exc_info=True)
            return {"reply": replies.tool_failed()}
    return node


def build_chat_graph(llm, checkpointer, start_approval):
    """`start_approval(company_id, employee_id, request_id)` launches the
    approval graph (approval.py) when a request needs a manager."""

    def understand(state: ChatState, config: RunnableConfig):
        if action := quick_action(state):
            return {"action": action}
        try:
            entries = tools.get_roster(config["configurable"]["principal"])
        except tools.ToolError:  # fail before spending an LLM call
            return {"reply": replies.tool_failed()}
        return ask_llm(llm, llm_context(state, entries))

    def finish(state: ChatState):
        history = (state.get("history") or []) + [f"Employee: {state['message']}",
                                                  f"Assistant: {state['reply']}"]
        return {"history": history[-HISTORY_LINES:], "suggestions": suggestions(state)}

    def do_confirm(state, who):
        return confirm(state, who, start_approval)
    do_confirm.__name__ = "confirm"

    # Action.tool -> handler. The keys are exactly the tools listed in llm.py.
    handlers = {
        "get_roster": show_roster,
        "check_cab_options": check_cab_options,
        "raise_change_request": do_confirm,
        "cancel_trip": cancel_trip,
        "get_request_status": request_status,
        "get_policy": policy,
        "ask_clarification": clarify,
        "small_talk": small_talk,
        "none": nothing,
    }

    g = StateGraph(ChatState)
    g.add_node("understand", understand)
    g.add_node("finish", finish)
    for name, handler in handlers.items():
        g.add_node(name, as_node(handler))
        g.add_edge(name, "finish")
    g.add_edge(START, "understand")
    g.add_conditional_edges("understand",
                            lambda s: "finish" if s.get("reply") else s["action"]["tool"],
                            ["finish", *handlers])
    g.add_edge("finish", END)
    return g.compile(checkpointer=checkpointer)
