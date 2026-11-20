"""Sample data: two companies with DIFFERENT rules, so per-company enforcement
is visible in the demo and provable in tests.

                cut-off   seats/cab   manager may override cut-off?
    Acme        120 min       4       yes
    Globex       60 min       6       no

Trips exist for today and tomorrow (dates are relative to clock.today(), so the
demo works on any day). `seats_taken` counts every rider, including people who
are not modelled as employees here; that keeps the sample small.

Run `python -m app.seed` to reset the database to this state.
"""
from datetime import timedelta

from psycopg_pool import ConnectionPool

from app import clock, config, db

COMPANIES = [
    # id, name, cutoff_minutes, cab_capacity, escalation_allowed
    ("acme", "Acme Corp", 120, 4, True),
    ("globex", "Globex Ltd", 60, 6, False),
]

EMPLOYEES = [
    # id, company, name, role
    ("E101", "acme", "Priya Sharma", "employee"),
    ("E102", "acme", "Rahul Verma", "employee"),
    ("E103", "acme", "Asha Nair", "employee"),
    ("M100", "acme", "Kavita Rao", "manager"),
    ("G201", "globex", "Arjun Mehta", "employee"),
    ("G202", "globex", "Meera Iyer", "employee"),
    ("M200", "globex", "Vikram Singh", "manager"),
]

# seats already taken on each slot (same pattern today and tomorrow).
# Acme 20:30 has exactly 1 free seat, 21:30 is full (capacity 4).
# Globex 20:30 has exactly 1 free seat, 21:30 is full (capacity 6).
SEATS_TAKEN = {
    "acme":   {"08:00": 1, "09:00": 3, "10:00": 2, "18:30": 3, "19:30": 2, "20:30": 3, "21:30": 4},
    "globex": {"08:00": 2, "09:00": 4, "10:00": 3, "18:30": 3, "19:30": 5, "20:30": 5, "21:30": 6},
}
PICKUP_SLOTS = ["08:00", "09:00", "10:00"]
DROP_SLOTS = ["18:30", "19:30", "20:30", "21:30"]

# employee -> {(day_offset, direction): slot}
ROSTER = {
    "E101": {(0, "drop"): "18:30", (1, "pickup"): "09:00", (1, "drop"): "18:30"},
    "E102": {(0, "drop"): "18:30", (1, "pickup"): "09:00", (1, "drop"): "19:30"},
    "E103": {(0, "drop"): "20:30", (1, "pickup"): "10:00", (1, "drop"): "20:30"},
    "G201": {(0, "drop"): "18:30", (1, "pickup"): "09:00", (1, "drop"): "18:30"},
    "G202": {(0, "drop"): "19:30", (1, "pickup"): "08:00", (1, "drop"): "19:30"},
}

OUR_TABLES = "notifications, change_requests, roster, trips, employees, companies"
LANGGRAPH_TABLES = ("checkpoints", "checkpoint_blobs", "checkpoint_writes")


def seed(p: ConnectionPool) -> None:
    """Wipe all data (including paused conversations) and insert the sample."""
    with p.connection() as conn, conn.transaction():
        conn.execute(f"TRUNCATE {OUR_TABLES} RESTART IDENTITY CASCADE")
        for table in LANGGRAPH_TABLES:  # exist only after the checkpointer's setup()
            if conn.execute("SELECT to_regclass(%s) AS t", (table,)).fetchone()["t"]:
                conn.execute(f"TRUNCATE {table}")

        conn.cursor().executemany("INSERT INTO companies VALUES (%s, %s, %s, %s, %s)", COMPANIES)
        conn.cursor().executemany("INSERT INTO employees VALUES (%s, %s, %s, %s)", EMPLOYEES)

        caps = {c[0]: c[3] for c in COMPANIES}
        trip_ids = {}  # (company, day_offset, direction, slot) -> trip id
        for company, seats in SEATS_TAKEN.items():
            prefix = company[:2].upper()
            for offset in (0, 1):
                d = clock.today() + timedelta(days=offset)
                for direction, slots in (("pickup", PICKUP_SLOTS), ("drop", DROP_SLOTS)):
                    for slot in slots:
                        row = conn.execute(
                            "INSERT INTO trips (company_id, trip_date, direction, slot, cab_no, capacity, seats_taken)"
                            " VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
                            (company, d, direction, slot, f"{prefix}-{slot.replace(':', '')}",
                             caps[company], seats[slot]),
                        ).fetchone()
                        trip_ids[(company, offset, direction, slot)] = row["id"]

        company_of = {e[0]: e[1] for e in EMPLOYEES}
        for emp, entries in ROSTER.items():
            for (offset, direction), slot in entries.items():
                company = company_of[emp]
                conn.execute(
                    "INSERT INTO roster (company_id, employee_id, trip_id, trip_date, direction)"
                    " VALUES (%s, %s, %s, %s, %s)",
                    (company, emp, trip_ids[(company, offset, direction, slot)],
                     clock.today() + timedelta(days=offset), direction),
                )


def seed_if_empty(p: ConnectionPool) -> None:
    with p.connection() as conn:
        empty = conn.execute("SELECT count(*) AS n FROM companies").fetchone()["n"] == 0
    if empty:
        seed(p)


if __name__ == "__main__":
    pool = db.pool(config.DATABASE_URL)
    db.create_schema(pool)
    seed(pool)
    print("Database reset to sample data.")
