"""Database connection pool and schema.

Design notes
- Plain SQL with psycopg 3, no ORM: the queries are few and short, and every
  query can be read as-is.
- One pool is shared by our code and by LangGraph's PostgresSaver.
- autocommit=True: reads are simple; writes that must be atomic use an explicit
  `with conn.transaction():` block (see repo.py).
"""
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    cutoff_minutes      INT  NOT NULL,   -- change must arrive this long before the trip
    cab_capacity        INT  NOT NULL,   -- seats per cab for this company
    escalation_allowed  BOOLEAN NOT NULL -- may a manager override a cut-off breach?
);

CREATE TABLE IF NOT EXISTS employees (
    id          TEXT PRIMARY KEY,
    company_id  TEXT NOT NULL REFERENCES companies(id),
    name        TEXT NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('employee', 'manager'))
);

CREATE TABLE IF NOT EXISTS trips (            -- one cab at one time on one day
    id           SERIAL PRIMARY KEY,
    company_id   TEXT NOT NULL REFERENCES companies(id),
    trip_date    DATE NOT NULL,
    direction    TEXT NOT NULL CHECK (direction IN ('pickup', 'drop')),
    slot         TEXT NOT NULL,              -- 'HH:MM' (IST)
    cab_no       TEXT NOT NULL,
    capacity     INT  NOT NULL,
    seats_taken  INT  NOT NULL CHECK (seats_taken >= 0 AND seats_taken <= capacity),
    UNIQUE (company_id, id)                  -- target for the composite FK below
);

CREATE TABLE IF NOT EXISTS roster (           -- which employee rides which trip
    id           SERIAL PRIMARY KEY,
    company_id   TEXT NOT NULL,
    employee_id  TEXT NOT NULL REFERENCES employees(id),
    trip_id      INT  NOT NULL,
    trip_date    DATE NOT NULL,
    direction    TEXT NOT NULL,
    cancelled_at TIMESTAMPTZ,                -- set when the employee cancels (row kept for the audit trail)
    UNIQUE (employee_id, trip_date, direction),
    -- Isolation enforced by the database itself: a roster row can only point
    -- at a trip of the SAME company, even if application code had a bug.
    FOREIGN KEY (company_id, trip_id) REFERENCES trips (company_id, id)
);

CREATE TABLE IF NOT EXISTS change_requests (  -- the audit trail: who asked, who decided
    id            SERIAL PRIMARY KEY,
    company_id    TEXT NOT NULL,
    employee_id   TEXT NOT NULL,
    roster_id     INT  NOT NULL REFERENCES roster(id),
    from_trip_id  INT  NOT NULL,
    to_trip_id    INT,            -- NULL for a cancellation
    status        TEXT NOT NULL,  -- APPLIED | PENDING | APPROVED | REJECTED | FAILED | CANCELLED
    reason        TEXT,           -- why it needed approval / why it failed
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    decided_by    TEXT,
    decided_at    TIMESTAMPTZ,
    note          TEXT
);
-- At most one open (PENDING) request per roster entry: a double-submit cannot
-- create two requests, even if both arrive at the same millisecond.
CREATE UNIQUE INDEX IF NOT EXISTS one_pending_per_roster
    ON change_requests (roster_id) WHERE status = 'PENDING';

CREATE TABLE IF NOT EXISTS notifications (    -- messages delivered on the next /chat
    id           SERIAL PRIMARY KEY,
    company_id   TEXT NOT NULL,
    employee_id  TEXT NOT NULL,
    text         TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    delivered    BOOLEAN NOT NULL DEFAULT FALSE
);

-- Upgrades for a database created by an older version (idempotent).
ALTER TABLE roster ADD COLUMN IF NOT EXISTS cancelled_at TIMESTAMPTZ;
ALTER TABLE change_requests ALTER COLUMN to_trip_id DROP NOT NULL;
"""

_pool: ConnectionPool | None = None


def pool(url: str) -> ConnectionPool:
    """Create (once) and return the shared pool."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            url,
            min_size=1,
            max_size=20,
            # prepare_threshold=0 is required by LangGraph's PostgresSaver.
            # statement_timeout: a stuck query fails in 2s instead of hanging the chat.
            kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": 0,
                    "options": "-c statement_timeout=2000"},
            open=True,
        )
    return _pool


def get() -> ConnectionPool:
    """The pool created at startup (used by repo.py)."""
    assert _pool is not None, "db.pool() must be called at startup"
    return _pool


def close_pool() -> None:
    """Close the pool. Tests use this to simulate a service restart."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def create_schema(p: ConnectionPool) -> None:
    with p.connection() as conn:
        conn.execute(SCHEMA, prepare=False)  # several statements: cannot be prepared
