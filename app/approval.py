"""Manager approval: a tiny LangGraph graph that PAUSES until a manager decides.

    START -> wait_for_manager (interrupt) ... hours later ... -> finalize -> END

One graph run per request, thread id "approval:<request_id>". `interrupt()`
saves the run in Postgres (PostgresSaver) and stops. A separate HTTP call from
the manager resumes it with Command(resume=decision). Because the paused run
lives in Postgres, not in memory, it survives a service restart.

Why a separate thread from the employee's chat: the employee can keep chatting
("what's the status?", "show tomorrow's cab") while the request waits. If the
chat thread itself were paused, every message would be stuck behind the manager.

The change_requests row is the source of truth for the request's status; the
graph is how the pause/resume is delivered.
"""
import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app import clock, replies, repo, rules

log = logging.getLogger(__name__)


class ApprovalState(TypedDict, total=False):
    request_id: int
    company_id: str
    employee_id: str
    decision: dict  # {"approve": bool, "manager_id": str, "note": str | None}
    outcome: str


def finalize(company_id: str, request_id: int, decision: dict) -> str:
    """Apply the manager's decision. Re-checks the rules first: hours may have
    passed, the cab may be full now, or the trip may already have left.
    The manager waives only the overridable rule (cut-off); nothing else.

    Each outcome is saved together with the employee's notification in one
    transaction (repo.close_request). If someone else decided first, nothing is
    written or sent, and the outcome that was actually saved is returned."""
    req = repo.get_request(company_id, request_id)
    if req is None or req["status"] != "PENDING":
        return req["status"] if req else "NOT_FOUND"
    old = repo.trip(company_id, req["from_trip_id"])
    new = repo.trip(company_id, req["to_trip_id"])
    mgr, note = decision["manager_id"], decision.get("note")

    def close(status: str, why: str | None, move: bool, message: str) -> str:
        if repo.close_request(company_id, request_id, status, mgr, why, move, message):
            return status
        return repo.get_request(company_id, request_id)["status"]  # lost the race: report the winner

    def fail(reason: str) -> str:
        return close("FAILED", reason, False, replies.notify_failed(old, new, reason))

    if not decision["approve"]:
        return close("REJECTED", note, False, replies.notify_rejected(old, note))

    filled_up = f"the {replies.t(new['slot'])} cab filled up while the request was waiting."
    policy = repo.get_company(company_id)
    blocking = [v for v in rules.check_change(policy, clock.now(), old, new) if not v.escalatable]
    if any(v.code == "CAB_FULL" for v in blocking):
        return fail(filled_up)
    if blocking:
        return fail(" ".join(v.message for v in blocking))
    try:
        return close("APPROVED", note, True, replies.notify_approved(old, new))
    except repo.SeatTaken:  # filled in the last millisecond, after the check above
        return fail(filled_up)
    except repo.AlreadyChanged:
        return fail("your booking was changed in the meantime.")


def build_approval_graph(checkpointer):
    def wait_for_manager(state: ApprovalState):
        # Execution stops here and is saved. On resume, interrupt() returns the
        # value passed in Command(resume=...). (LangGraph re-runs this node from
        # the top on resume, so nothing with side effects goes before it.)
        decision = interrupt({"request_id": state["request_id"]})
        return {"decision": decision}

    def finalize_node(state: ApprovalState):
        return {"outcome": finalize(state["company_id"], state["request_id"], state["decision"])}

    g = StateGraph(ApprovalState)
    g.add_node("wait_for_manager", wait_for_manager)
    g.add_node("finalize", finalize_node)
    g.add_edge(START, "wait_for_manager")
    g.add_edge("wait_for_manager", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


def _cfg(request_id: int) -> dict:
    return {"configurable": {"thread_id": f"approval:{request_id}"}}


def start(graph, company_id: str, employee_id: str, request_id: int) -> None:
    """Run until the interrupt, i.e. pause and wait for the manager."""
    graph.invoke({"request_id": request_id, "company_id": company_id, "employee_id": employee_id},
                 _cfg(request_id))


def decide(graph, company_id: str, employee_id: str, request_id: int, decision: dict) -> str:
    """Resume the paused run with the manager's decision; returns the final status."""
    cfg = _cfg(request_id)
    if not graph.get_state(cfg).next:
        # Crash between saving the request and starting the run: start it now,
        # so a decision is never lost.
        start(graph, company_id, employee_id, request_id)
    if graph.get_state(cfg).next == ("finalize",):
        # An earlier decision failed half-way (e.g. the database dropped out).
        # Its transaction rolled back, so the request is still PENDING: run
        # finalize again, with THIS decision.
        graph.update_state(cfg, {"decision": decision})
        return graph.invoke(None, cfg)["outcome"]
    return graph.invoke(Command(resume=decision), cfg)["outcome"]
