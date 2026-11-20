"""Every sentence the employee sees, in one file.

Why templates and not LLM-written replies: replies contain times, cab numbers
and "done". Building them from database rows means the bot can never confirm
something that did not happen, or quote the wrong company's rule. It also
saves an LLM call per message (latency + the 4-call budget).

Tone: warm and short, like a helpful transport-desk colleague. When something
fails it always says (1) what happened, (2) that nothing changed (when true),
and (3) what to do next.
"""
from datetime import date

from app import clock, config

DESK = f"call the transport desk on {config.DESK_PHONE}"


t = clock.pretty  # '20:30' -> '8:30 PM' (friendlier for a tired employee)


def day(d: date) -> str:
    return "today" if d == clock.today() else "tomorrow" if (d - clock.today()).days == 1 else d.isoformat()


def trip_label(trip: dict) -> str:
    """'drop today at 6:30 PM'"""
    return f"{trip['direction']} {day(trip['trip_date'])} at {t(trip['slot'])}"


def seats(n: int) -> str:
    return f"{n} seat{'s' if n != 1 else ''} left"


# --- normal conversation ------------------------------------------------------

CAN_DO = ("I can show your cabs, move a pickup or drop to another time, cancel a trip, "
          "or tell you your company's change rules.")


def help_text() -> str:
    return f"{CAN_DO} Try \"show my cabs\" or \"can I leave at 8:30 tonight?\"."


def greeting(name: str) -> str:
    return f"Hi {name}! 👋 {CAN_DO} What do you need?"


def thanks(name: str) -> str:
    return f"You're welcome, {name}! Anything else about your cabs?"


def off_topic() -> str:
    return f"Sorry, I can only help with your own office cab bookings. {help_text()}"


def nothing_to_confirm() -> str:
    return "There's nothing waiting for a yes or no right now. Anything else I can help with?"


def nothing_changed() -> str:
    return "Okay, I haven't changed anything. Your booking stays as it is. Anything else?"


def roster(entries: list[dict]) -> str:
    if not entries:
        return "You have no upcoming cab bookings."
    lines = [f"• {e['direction'].capitalize()} {day(e['trip_date'])} at {t(e['slot'])}, cab {e['cab_no']}"
             for e in entries]
    return "Your upcoming cabs:\n" + "\n".join(lines)


def policy(p: dict) -> str:
    override = ("If it's later than that, I can ask your transport manager to approve it."
                if p["escalation_allowed"] else
                f"After that, changes aren't possible through me; please {DESK}.")
    return (f"At {p['name']}, a pickup or drop can be changed until {p['cutoff_minutes']} minutes "
            f"before the cab leaves. {override} Each cab has {p['cab_capacity']} seats, "
            "and you can cancel a trip any time before it leaves.")


def which_trip(entries: list[dict]) -> str:
    options = " or ".join(trip_label(e) for e in entries)
    return f"Which trip do you mean: {options}?"


def no_booking(direction: str | None, day_word: str | None) -> str:
    what = direction or "cab"
    when = f" {day_word}" if day_word else ""
    return f"I can't find an upcoming {what} booking for you{when}. Say \"show my cabs\" to see what you have."


def options(entry: dict, opts: list[dict]) -> str:
    lines = []
    for i, o in enumerate(opts, 1):
        extra = " (needs manager approval)" if o["approval_reason"] else ""
        lines.append(f"{i}. {t(o['slot'])}, cab {o['cab_no']}, {seats(o['seats_free'])}{extra}")
    return (f"Your {trip_label(entry)} could move to:\n" + "\n".join(lines)
            + "\nReply with the option number or the time you want.")


def nothing_that_way(entry: dict, shift: str, unavailable: list[str]) -> str:
    """Said before the nearest options, so they are never mistaken for what was asked."""
    why = ""
    if unavailable:
        names = " and ".join(t(s) for s in unavailable)
        why = f" (the {names} cab{'s' if len(unavailable) > 1 else ''} can't be booked)"
    return (f"There's no {shift} {entry['direction']} cab I can offer {day(entry['trip_date'])}{why}. "
            f"These are the nearest I have instead.\n")


def no_options(entry: dict) -> str:
    return (f"Sorry, there's no other {entry['direction']} cab with a free seat that I can offer "
            f"{day(entry['trip_date'])}. Your {t(entry['slot'])} booking is unchanged. "
            f"If you need something else, {DESK}.")


def propose(entry: dict, opt: dict) -> str:
    if opt["approval_reason"]:
        return (f"{opt['approval_reason']} I can ask your transport manager to approve moving your "
                f"{trip_label(entry)} to {t(opt['slot'])} (cab {opt['cab_no']}). Your "
                f"{t(entry['slot'])} seat stays booked until they decide. Send it? (yes/no)")
    return (f"Good news: the {t(opt['slot'])} cab ({opt['cab_no']}) has {seats(opt['seats_free'])}. "
            f"Shall I move your {trip_label(entry)} to {t(opt['slot'])}? (yes/no)")


def propose_cancel(entries: list[dict]) -> str:
    trips = " and ".join(f"{trip_label(e)} (cab {e['cab_no']})" for e in entries)
    return f"Shall I cancel your {trips}? (yes/no)"


def already_on(entry: dict) -> str:
    return f"You're already booked on the {t(entry['slot'])} cab ({entry['cab_no']}). "


def no_cab_at(slot: str) -> str:
    return f"There is no cab at {t(slot)}. "


def blocked(slot: str, reasons: list[str]) -> str:
    return f"I can't book the {t(slot)} cab: {' '.join(reasons)} "


def pick_option() -> str:
    return "Which option would you like? Reply with the number or the time."


def applied(old: dict, new: dict) -> str:
    return (f"Done. Your {new['direction']} {day(new['trip_date'])} is now at {t(new['slot'])} "
            f"in cab {new['cab_no']} (it was {t(old['slot'])}). Have a good trip!")


def cancelled(entry: dict) -> str:
    return (f"Done. Your {trip_label(entry)} (cab {entry['cab_no']}) is cancelled and the seat "
            "has been freed for a colleague.")


def changed_meanwhile() -> str:
    return ("Your bookings changed while I was working on that, so I haven't cancelled anything. "
            "Say \"show my cabs\" to see where things stand.")


def too_late_to_cancel(entry: dict) -> str:
    return f"Your {t(entry['slot'])} cab has already left, so I can't cancel it. If something is wrong, {DESK}."


def sent_for_approval(old: dict, new: dict) -> str:
    return (f"Sent to your transport manager for approval: {old['direction']} {day(new['trip_date'])} "
            f"from {t(old['slot'])} to {t(new['slot'])}. Your {t(old['slot'])} seat stays booked until "
            "they decide. I'll tell you here as soon as they do.")


def already_pending() -> str:
    return ("You already have this request waiting for your manager. "
            "I'll tell you here as soon as they decide.")


def already_done(trip: dict | None = None) -> str:
    if trip is None:
        return "That's already done. Nothing else to do."
    return f"You're already on the {t(trip['slot'])} cab ({trip['cab_no']}). Nothing else to do."


def cab_filled(slot: str) -> str:
    return f"Sorry, the {t(slot)} cab filled up while you were deciding. Nothing has changed. "


def booking_changed(trip: dict) -> str:
    return (f"Your booking changed while I was working on that: you're now on the {t(trip['slot'])} cab "
            f"({trip['cab_no']}). I haven't moved it again. ")


def rejected(reasons: list[str]) -> str:
    return f"I couldn't make that change: {' '.join(reasons)} Nothing has changed. "


def gone() -> str:
    return "That option is no longer available. Say \"show my cabs\" to start again."


def status(req: dict | None) -> str:
    if req is None:
        return "You have no change requests."
    if req["to"] is None:  # a cancellation
        return f"Your last request cancelled your {trip_label(req['from'])}. It is done."
    move = f"{req['from']['direction']} {t(req['from']['slot'])} → {t(req['to']['slot'])}"
    if req["status"] == "PENDING":
        since = req["created_at"].astimezone(clock.IST).strftime("%I:%M %p").lstrip("0")
        return (f"Your request ({move}) has been with your transport manager since {since}. "
                f"Your original seat is still booked. If it is urgent, {DESK}.")
    return {
        "APPLIED": f"Your change ({move}) is done.",
        "APPROVED": f"Your manager approved your change ({move}), and it is done.",
        "REJECTED": f"Your manager declined your change ({move}). Their note: {req['note'] or 'none'}. "
                    "Your original booking is unchanged.",
        "FAILED": f"Your change ({move}) could not be completed: {req['note']}. "
                  "Your original booking is unchanged.",
    }[req["status"]]


# --- failures -----------------------------------------------------------------

def llm_down() -> str:
    return ("Sorry, I couldn't process your message just now. Nothing on your booking has changed. "
            f"Please try again in a minute, or {DESK}.")


def not_understood() -> str:
    return "Sorry, I didn't get that. " + help_text()


def tool_failed() -> str:
    return ("I couldn't reach the roster system just now, so nothing has been changed. "
            f"Please try again in a minute, or {DESK}.")


def unexpected(ref: str) -> str:
    return (f"Something went wrong on our side (ref {ref}). Nothing on your booking has changed. "
            f"Please try again, or {DESK}.")


# --- notifications after a manager decision ------------------------------------

def notify_approved(old: dict, new: dict) -> str:
    return (f"Good news: your manager approved your request. Your {new['direction']} "
            f"{day(new['trip_date'])} is now at {t(new['slot'])} in cab {new['cab_no']} (it was {t(old['slot'])}).")


def notify_rejected(old: dict, note: str | None) -> str:
    why = f" Their note: \"{note}\"." if note else ""
    return (f"Your manager declined the change.{why} Your {trip_label(old)} "
            f"(cab {old['cab_no']}) is unchanged.")


def notify_failed(old: dict, new: dict, reason: str) -> str:
    return (f"Your manager approved moving to {t(new['slot'])}, but I couldn't complete it: {reason} "
            f"Your {trip_label(old)} is unchanged. Ask me for other options, or {DESK}.")
