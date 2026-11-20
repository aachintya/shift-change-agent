"""HTTP API.

    POST /auth/token                     mock identity provider (demo only)
    POST /chat                           employee chat        (employee JWT)
    GET  /notifications                  manager decisions    (employee JWT)
    GET  /me/trips                       my bookings          (employee JWT, for the demo page)
    GET  /manager/requests               pending approvals    (manager JWT)
    POST /manager/requests/{id}/decision approve / reject     (manager JWT)
    GET  /                               demo page
    GET  /demo/people                    who can sign in      (demo page only)

Identity: every route gets its Principal from auth.py (the JWT). Bodies carry
only the message or the decision, never an employee or company id.
Errors: three handlers below guarantee a plain-language JSON reply, never a
stack trace.
"""
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from langgraph.checkpoint.postgres import PostgresSaver
from pydantic import BaseModel, Field

from app import agent, approval, auth, clock, config, db, replies, repo, seed, tools
from app.auth import Principal
from app.llm import OpenAICompatibleLLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")


class TokenIn(BaseModel):
    employee_id: str


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=500)


class DecisionIn(BaseModel):
    approve: bool
    note: str | None = Field(default=None, max_length=300)


def create_app(llm=None) -> FastAPI:
    """App factory. Tests pass a FakeLLM; production uses the real provider.
    Creating a second app against the same database = a service restart."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pool = db.pool(config.DATABASE_URL)
        db.create_schema(pool)
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()  # LangGraph's own tables (idempotent)
        seed.seed_if_empty(pool)

        approvals = approval.build_approval_graph(checkpointer)
        app.state.approvals = approvals
        app.state.chat = agent.build_chat_graph(
            llm or OpenAICompatibleLLM(), checkpointer,
            start_approval=lambda c, e, r: approval.start(approvals, c, e, r))
        yield
        db.close_pool()

    app = FastAPI(title="Shift-Change Request Agent", lifespan=lifespan)

    # ---- errors: always a plain message, never a stack trace ----------------
    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return JSONResponse({"reply": exc.detail}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def bad_input(request: Request, exc: RequestValidationError):
        return JSONResponse({"reply": "Please send a message between 1 and 500 characters."}, status_code=422)

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception):
        ref = uuid.uuid4().hex[:8]
        log.exception("unhandled error ref=%s", ref)  # full trace goes to logs only
        return JSONResponse({"reply": replies.unexpected(ref)}, status_code=500)

    # ---- routes ---------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.post("/auth/token")
    def token(body: TokenIn):
        """DEMO identity provider. Company and role are read from the database,
        never from the caller."""
        emp = repo.get_employee(body.employee_id)
        if emp is None:
            raise HTTPException(404, "Unknown employee id.")
        p = Principal(emp["id"], emp["company_id"], emp["role"])
        return {"token": auth.issue_token(p), "name": emp["name"],
                "role": emp["role"], "company": repo.get_company(emp["company_id"])["name"]}

    @app.post("/chat")
    def chat(body: ChatIn, who: Principal = Depends(auth.employee)):
        result = app.state.chat.invoke(
            # per-message fields are reset; history/focus/options come from the checkpoint
            {"message": body.message.strip(), "action": None, "reply": "", "llm_calls": 0,
             "suggestions": []},
            {"configurable": {"thread_id": f"chat:{who.company_id}:{who.employee_id}",
                              "principal": who}},
        )
        action = result.get("action") or {}
        return {
            "reply": result["reply"],
            "tool": action.get("tool"),          # exposed for the eval / demo
            "llm_calls": result.get("llm_calls", 0),
            "suggestions": result.get("suggestions", []),  # quick-reply buttons
            "notifications": repo.take_notifications(who.company_id, who.employee_id),
        }

    @app.get("/notifications")
    def notifications(who: Principal = Depends(auth.employee)):
        return {"notifications": repo.take_notifications(who.company_id, who.employee_id)}

    @app.get("/me/trips")
    def my_trips(who: Principal = Depends(auth.employee)):
        """The employee's bookings, so the demo page can show changes as they happen."""
        req = tools.get_request_status(who)
        waiting = req if req and req["status"] == "PENDING" else None
        return {"trips": [{"roster_id": e["roster_id"], "direction": e["direction"],
                           "day": replies.day(e["trip_date"]), "time": clock.pretty(e["slot"]),
                           "cab_no": e["cab_no"],
                           "waiting_for": clock.pretty(waiting["to"]["slot"])
                           if waiting and waiting["roster_id"] == e["roster_id"] else None}
                          for e in tools.get_roster(who)]}

    @app.get("/demo/people", include_in_schema=False)
    def people():
        """DEMO page only: the sign-in list and the demo clock."""
        return {"now": f"{clock.now():%a %d %b, %I:%M %p} IST", "people": repo.people()}

    @app.get("/manager/requests")
    def pending(who: Principal = Depends(auth.manager)):
        return {"requests": [{**r, "day": replies.day(r["trip_date"]),  # friendly labels for the UI
                              "from_time": clock.pretty(r["from_slot"]), "to_time": clock.pretty(r["to_slot"])}
                             for r in repo.pending_requests(who.company_id)]}

    @app.post("/manager/requests/{request_id}/decision")
    def decide(request_id: int, body: DecisionIn, who: Principal = Depends(auth.manager)):
        req = repo.get_request(who.company_id, request_id)  # scoped: other company -> None
        if req is None:
            raise HTTPException(404, "No such request.")  # 404, not 403: don't leak existence
        if req["status"] != "PENDING":
            raise HTTPException(409, f"This request is already {req['status'].lower()}.")
        status = approval.decide(app.state.approvals, who.company_id, req["employee_id"], request_id,
                                 {"approve": body.approve, "manager_id": who.employee_id, "note": body.note})
        return {"request_id": request_id, "status": status}

    return app


app = create_app()
