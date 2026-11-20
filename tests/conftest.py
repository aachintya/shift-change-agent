"""Shared test fixtures.

- Real Postgres (DATABASE_URL), reset to the sample data before every test.
- FakeLLM: tests script what the "model" answers, so no key or network is needed
  and a quota error can never fail a test.
- Clock frozen at 17:00 IST today, so cut-off maths is deterministic:
    Acme   (120 min): today's 18:30 drop is 90 min away -> needs approval
    Globex  (60 min): today's 18:30 drop is 90 min away -> allowed
"""
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from app import clock, db, seed
from app.llm import FakeLLM
from app.main import create_app


@pytest.fixture(autouse=True)
def frozen_clock():
    clock.freeze(clock.at(clock.today(), "17:00"))
    yield
    clock.freeze(None)


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def start_app(llm):
    """Start the service. Calling it again after stop() = a service restart."""
    running = []

    def start() -> TestClient:
        client = TestClient(create_app(llm=llm), raise_server_exceptions=False)
        client.__enter__()  # runs startup: pool, schema, checkpointer
        running.append(client)
        return client

    yield start
    for c in running:
        c.__exit__(None, None, None)


@pytest.fixture
def client(start_app):
    c = start_app()
    seed.seed(db.get())
    return c


def stop(client: TestClient) -> None:
    """Shut the service down (closes the DB pool, drops in-memory graphs)."""
    client.__exit__(None, None, None)


def headers(client: TestClient, employee_id: str) -> dict:
    token = client.post("/auth/token", json={"employee_id": employee_id}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


class Chat:
    """One employee talking to the bot; each say() scripts the fake LLM's answer."""

    def __init__(self, client, llm, employee_id):
        self.client, self.llm = client, llm
        self.h = headers(client, employee_id)

    def say(self, text: str, *llm_outputs) -> dict:
        # Replace (not extend): a plain "yes" is read by code and uses no LLM
        # output, so a leftover item must not leak into the next message.
        self.llm.script = list(llm_outputs)
        return self.client.post("/chat", json={"message": text}, headers=self.h).json()


def booking(employee_id: str, direction: str, day_offset: int = 0) -> str:
    """The slot an employee is booked on, read straight from the database."""
    with db.get().connection() as conn:
        return conn.execute(
            "SELECT t.slot FROM roster r JOIN trips t ON t.id = r.trip_id"
            " WHERE r.employee_id = %s AND r.direction = %s AND r.trip_date = %s",
            (employee_id, direction, clock.today() + timedelta(days=day_offset))).fetchone()["slot"]


def sql(query: str, *params):
    with db.get().connection() as conn:
        cur = conn.execute(query, params)
        return cur.fetchall() if cur.description else None
