"""All SQL lives here.

Isolation rule: every function that reads or writes company data takes
`company_id` and puts it in the WHERE clause. There is deliberately no
"get anything by id" function, so a forgotten filter cannot happen by accident.
(The composite foreign key in db.py is the second, database-level guard.)
"""
from datetime import date

from psycopg import errors

from app import db

TRIP_COLS = "t.id, t.trip_date, t.direction, t.slot, t.cab_no, t.capacity, t.seats_taken"


class SeatTaken(Exception):
    """The target cab filled up between showing the option and booking it."""


class AlreadyChanged(Exception):
    """The roster entry is no longer on the trip we expected (double submit / race)."""


def _one(sql: str, params: tuple):
    with db.get().connection() as conn:
        return conn.execute(sql, params).fetchone()


def _all(sql: str, params: tuple):
    with db.get().connection() as conn:
        return conn.execute(sql, params).fetchall()


# --- reads -------------------------------------------------------------------

def get_employee(employee_id: str):
    """Used only by the mock token issuer, to put the TRUE company into the token."""
    return _one("SELECT id, company_id, name, role FROM employees WHERE id = %s", (employee_id,))


def people():
    """Demo page only: who can sign in, with their company's rules."""
    return _all("SELECT e.id, e.name, e.role, c.id AS company_id, c.name AS company,"
                " c.cutoff_minutes, c.cab_capacity, c.escalation_allowed"
                " FROM employees e JOIN companies c ON c.id = e.company_id ORDER BY c.id, e.role, e.id", ())


def get_company(company_id: str):
    return _one("SELECT * FROM companies WHERE id = %s", (company_id,))


def roster(company_id: str, employee_id: str, from_date: date):
    """The employee's bookings from `from_date` on, with trip details."""
    return _all(
        f"SELECT r.id AS roster_id, {TRIP_COLS} FROM roster r"
        " JOIN trips t ON t.id = r.trip_id AND t.company_id = r.company_id"
        " WHERE r.company_id = %s AND r.employee_id = %s AND r.trip_date >= %s"
        "   AND r.cancelled_at IS NULL ORDER BY t.trip_date, t.slot",
        (company_id, employee_id, from_date),
    )


def roster_entry(company_id: str, employee_id: str, roster_id: int):
    return _one(
        f"SELECT r.id AS roster_id, {TRIP_COLS} FROM roster r"
        " JOIN trips t ON t.id = r.trip_id AND t.company_id = r.company_id"
        " WHERE r.company_id = %s AND r.employee_id = %s AND r.id = %s AND r.cancelled_at IS NULL",
        (company_id, employee_id, roster_id),
    )


def trip(company_id: str, trip_id: int):
    return _one(f"SELECT {TRIP_COLS} FROM trips t WHERE t.company_id = %s AND t.id = %s",
                (company_id, trip_id))


def trips_on(company_id: str, trip_date: date, direction: str):
    return _all(
        f"SELECT {TRIP_COLS} FROM trips t"
        " WHERE t.company_id = %s AND t.trip_date = %s AND t.direction = %s ORDER BY t.slot",
        (company_id, trip_date, direction),
    )


def get_request(company_id: str, request_id: int):
    return _one("SELECT * FROM change_requests WHERE company_id = %s AND id = %s",
                (company_id, request_id))


def latest_request(company_id: str, employee_id: str):
    return _one(
        "SELECT * FROM change_requests WHERE company_id = %s AND employee_id = %s"
        " ORDER BY id DESC LIMIT 1",
        (company_id, employee_id),
    )


def pending_requests(company_id: str):
    """For the manager's inbox."""
    return _all(
        "SELECT cr.id, cr.employee_id, e.name, cr.reason, cr.created_at,"
        "       f.trip_date, f.direction, f.slot AS from_slot, t.slot AS to_slot"
        " FROM change_requests cr"
        " JOIN employees e ON e.id = cr.employee_id"
        " JOIN trips f ON f.id = cr.from_trip_id AND f.company_id = cr.company_id"
        " JOIN trips t ON t.id = cr.to_trip_id AND t.company_id = cr.company_id"
        " WHERE cr.company_id = %s AND cr.status = 'PENDING' ORDER BY cr.id",
        (company_id,),
    )


# --- writes ------------------------------------------------------------------

def _move(conn, company_id: str, roster_id: int, from_trip: int, to_trip: int) -> None:
    """Move one rider from one cab to another. MUST run inside a transaction.

    Concurrency, without explicit locks:
    1. The roster update only matches if the rider is still on `from_trip`.
       A second, identical submit waits for the first to commit, then matches
       0 rows -> AlreadyChanged -> whole transaction rolls back.
    2. The seat update only matches while a seat is free. Two people racing
       for the last seat: the second waits on the row lock, re-checks
       `seats_taken < capacity`, matches 0 rows -> SeatTaken -> rollback.
    """
    moved = conn.execute(
        "UPDATE roster SET trip_id = %s"
        " WHERE id = %s AND company_id = %s AND trip_id = %s AND cancelled_at IS NULL",
        (to_trip, roster_id, company_id, from_trip)).rowcount
    if moved == 0:
        raise AlreadyChanged()
    taken = conn.execute(
        "UPDATE trips SET seats_taken = seats_taken + 1"
        " WHERE id = %s AND company_id = %s AND seats_taken < capacity",
        (to_trip, company_id)).rowcount
    if taken == 0:
        raise SeatTaken()
    conn.execute("UPDATE trips SET seats_taken = seats_taken - 1 WHERE id = %s AND company_id = %s",
                 (from_trip, company_id))


def apply_change(company_id: str, employee_id: str, roster_id: int, from_trip: int, to_trip: int) -> int:
    """Move the seat AND record the request in one transaction: both or neither."""
    with db.get().connection() as conn, conn.transaction():
        _move(conn, company_id, roster_id, from_trip, to_trip)
        return conn.execute(
            "INSERT INTO change_requests (company_id, employee_id, roster_id, from_trip_id, to_trip_id, status)"
            " VALUES (%s, %s, %s, %s, %s, 'APPLIED') RETURNING id",
            (company_id, employee_id, roster_id, from_trip, to_trip)).fetchone()["id"]


def cancel_many(company_id: str, employee_id: str, trips: list[tuple[int, int]]) -> None:
    """Cancel several bookings, each given as (roster_id, trip_id): free the
    seats, close any request waiting on them, and record each cancellation.
    ONE transaction: either every trip is cancelled or none is ("I'm on leave
    tomorrow" must never end half-done with a reply saying nothing changed).
    Raises AlreadyChanged if any of them was cancelled or moved meanwhile."""
    with db.get().connection() as conn, conn.transaction():
        for roster_id, trip_id in trips:
            done = conn.execute(
                "UPDATE roster SET cancelled_at = now()"
                " WHERE id = %s AND company_id = %s AND trip_id = %s AND cancelled_at IS NULL",
                (roster_id, company_id, trip_id)).rowcount
            if done == 0:
                raise AlreadyChanged()  # rolls back the trips already done in this call
            conn.execute("UPDATE trips SET seats_taken = seats_taken - 1 WHERE id = %s AND company_id = %s",
                         (trip_id, company_id))
            # A manager must not later "approve" a move of a booking that no longer exists.
            conn.execute("UPDATE change_requests SET status = 'FAILED', note = 'trip cancelled by the employee'"
                         " WHERE roster_id = %s AND company_id = %s AND status = 'PENDING'",
                         (roster_id, company_id))
            conn.execute("INSERT INTO change_requests (company_id, employee_id, roster_id, from_trip_id, status)"
                         " VALUES (%s, %s, %s, %s, 'CANCELLED')", (company_id, employee_id, roster_id, trip_id))


def create_pending(company_id: str, employee_id: str, roster_id: int, from_trip: int,
                   to_trip: int, reason: str) -> tuple[int, bool]:
    """Create a PENDING request. Returns (request_id, created).

    If one is already pending for this roster entry (double submit), the unique
    index refuses the insert and we return the existing one instead.
    """
    try:
        with db.get().connection() as conn:
            row = conn.execute(
                "INSERT INTO change_requests (company_id, employee_id, roster_id, from_trip_id, to_trip_id, status, reason)"
                " VALUES (%s, %s, %s, %s, %s, 'PENDING', %s) RETURNING id",
                (company_id, employee_id, roster_id, from_trip, to_trip, reason)).fetchone()
            return row["id"], True
    except errors.UniqueViolation:
        row = _one("SELECT id FROM change_requests WHERE company_id = %s AND roster_id = %s AND status = 'PENDING'",
                   (company_id, roster_id))
        return row["id"], False


def close_request(company_id: str, request_id: int, status: str, manager_id: str,
                  note: str | None, move: bool, message: str) -> bool:
    """Close a PENDING request as APPROVED / REJECTED / FAILED, and queue
    `message` for the employee.

    The status change, the seat move (move=True, i.e. approval) and the
    notification are ONE transaction: the employee can never miss the news of
    a decision that was saved, nor hear about one that was not.
    Returns False if the request was not PENDING any more (decided twice, or
    two managers at once): then nothing is written and nobody is notified.
    May raise SeatTaken / AlreadyChanged (then nothing is changed).
    """
    with db.get().connection() as conn, conn.transaction():
        req = conn.execute(
            "SELECT * FROM change_requests WHERE id = %s AND company_id = %s AND status = 'PENDING' FOR UPDATE",
            (request_id, company_id)).fetchone()
        if req is None:
            return False
        if move:
            _move(conn, company_id, req["roster_id"], req["from_trip_id"], req["to_trip_id"])
        conn.execute(
            "UPDATE change_requests SET status = %s, decided_by = %s, decided_at = now(), note = %s"
            " WHERE id = %s", (status, manager_id, note, request_id))
        conn.execute("INSERT INTO notifications (company_id, employee_id, text) VALUES (%s, %s, %s)",
                     (company_id, req["employee_id"], message))
        return True


def take_notifications(company_id: str, employee_id: str) -> list[str]:
    """Return undelivered notifications and mark them delivered (each shown once)."""
    rows = _all(
        "UPDATE notifications SET delivered = TRUE"
        " WHERE company_id = %s AND employee_id = %s AND NOT delivered RETURNING id, text",
        (company_id, employee_id))
    return [r["text"] for r in sorted(rows, key=lambda r: r["id"])]
