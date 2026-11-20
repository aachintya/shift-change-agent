"""Evaluation runner.

    python -m eval.run           # offline: scripted fake LLM (no key needed)
    python -m eval.run --real    # real LLM from .env: also measures latency (p95)

Reads eval/cases.jsonl, runs every case against a fresh copy of the sample
data, and writes eval/results.jsonl (fake) or eval/results_real.jsonl (real)
with the required fields: id, input, expected, actual, pass, notes.

A case is a list of steps:
    {"as": "G201", "say": "text", "llm": <scripted model output(s)>}
    {"as": "M100", "decide": "approve" | "reject", "note": "..."}
    {"restart": true}                      # stop and start the service
    {"fault": "check_cab_options"}         # that tool fails from now on
    {"fill_seat": {"company": "globex", "slot": "20:30"}}   # someone takes the seat

Judging is deterministic: we compare structured outcomes (chosen tool, final
booking in the DB, request status, phrases in the reply), not LLM-as-judge.
So the result is reproducible and costs nothing to rerun.

In --real mode the "llm" scripts are ignored, EXCEPT for cases marked
"scripted_llm": true (they simulate a provider failure, which a real model
cannot be asked to produce).
"""
import json
import os
import statistics
import sys
import time
from datetime import timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from app import clock, db, repo, seed, tools
from app.llm import FakeLLM, LLMError, OpenAICompatibleLLM
from app.main import create_app

HERE = Path(__file__).parent
REAL_PACE_SECONDS = float(os.getenv("EVAL_PACE_SECONDS", "9"))  # Groq free tier: 8k tokens/min


def _script(llm_field) -> list:
    """Case JSON -> FakeLLM script items ({"raise": msg} becomes an LLMError)."""
    items = llm_field if isinstance(llm_field, list) else [llm_field]
    return [LLMError(i["raise"]) if isinstance(i, dict) and "raise" in i else i for i in items]


def _start(llm) -> TestClient:
    client = TestClient(create_app(llm=llm), raise_server_exceptions=False)
    client.__enter__()
    return client


def _auth(client, who) -> dict:
    if who is None:
        return {}
    token = client.post("/auth/token", json={"employee_id": who}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def run_case(case: dict, real: bool) -> dict:
    inp = case["input"]
    clock.freeze(clock.at(clock.today(), inp.get("now", "17:00")))
    use_fake = not real or inp.get("scripted_llm", False)
    llm = FakeLLM() if use_fake else OpenAICompatibleLLM()
    client = _start(llm)
    seed.seed(db.get())

    actual = {"reply": None, "tool": None, "max_llm_calls": 0, "http_status": None,
              "notifications": [], "latencies_ms": []}
    try:
        for step in inp["steps"]:
            if "say" in step:
                if use_fake:
                    llm.script = _script(step.get("llm", []))
                else:  # free tiers cap tokens/minute; pace the run (not counted in latency)
                    time.sleep(REAL_PACE_SECONDS)
                t0 = time.perf_counter()
                r = client.post("/chat", json={"message": step["say"]}, headers=_auth(client, step.get("as")))
                actual["latencies_ms"].append(round((time.perf_counter() - t0) * 1000))
                body = r.json()
                actual.update(reply=body.get("reply"), tool=body.get("tool"), http_status=r.status_code)
                actual["max_llm_calls"] = max(actual["max_llm_calls"], body.get("llm_calls", 0))
                actual["notifications"] += body.get("notifications", [])
            elif "decide" in step:
                h = _auth(client, step["as"])
                with db.get().connection() as conn:
                    rid = conn.execute("SELECT max(id) AS id FROM change_requests").fetchone()["id"]
                r = client.post(f"/manager/requests/{rid}/decision", headers=h,
                                json={"approve": step["decide"] == "approve", "note": step.get("note")})
                actual.update(reply=r.json().get("reply"), http_status=r.status_code,
                              decision=r.json().get("status"))
            elif "restart" in step:
                client.__exit__(None, None, None)
                client = _start(llm)
            elif "fault" in step:
                tools.FAULTS.add(step["fault"])
            elif "fill_seat" in step:
                f = step["fill_seat"]
                with db.get().connection() as conn:
                    conn.execute("UPDATE trips SET seats_taken = capacity WHERE company_id = %s AND slot = %s"
                                 " AND direction = 'drop' AND trip_date = %s", (f["company"], f["slot"], clock.today()))

        exp = case["expected"]
        if "booking" in exp:
            b = exp["booking"]
            rows = repo.roster(repo.get_employee(b["employee"])["company_id"], b["employee"], clock.today())
            day = clock.today() + timedelta(days=b.get("day_offset", 0))
            actual["booking"] = next((e["slot"] for e in rows
                                      if e["direction"] == b["direction"] and e["trip_date"] == day), None)
        if "request_status" in exp:
            with db.get().connection() as conn:
                row = conn.execute("SELECT status FROM change_requests ORDER BY id DESC LIMIT 1").fetchone()
            actual["request_status"] = row["status"] if row else None
        if "notification_contains" in exp:  # pick up anything not yet delivered
            emp = exp["notification_contains"]["employee"]
            actual["notifications"] += repo.take_notifications(repo.get_employee(emp)["company_id"], emp)
    finally:
        tools.FAULTS.clear()
        client.__exit__(None, None, None)
        clock.freeze(None)
    return actual


def judge(expected: dict, actual: dict) -> list[str]:
    """Return the list of failed checks (empty = pass)."""
    fails = []
    reply = (actual.get("reply") or "").lower()
    for s in expected.get("reply_contains", []):
        if s.lower() not in reply:
            fails.append(f"reply missing {s!r}")
    for s in expected.get("reply_not_contains", []):
        if s.lower() in reply:
            fails.append(f"reply should not contain {s!r}")
    if "tool" in expected and actual.get("tool") != expected["tool"]:
        fails.append(f"tool {actual.get('tool')!r} != {expected['tool']!r}")
    if "booking" in expected and actual.get("booking") != expected["booking"]["slot"]:
        fails.append(f"booking {actual.get('booking')!r} != {expected['booking']['slot']!r}")
    if "request_status" in expected and actual.get("request_status") != expected["request_status"]:
        fails.append(f"request {actual.get('request_status')!r} != {expected['request_status']!r}")
    if "http_status" in expected and actual.get("http_status") != expected["http_status"]:
        fails.append(f"http {actual.get('http_status')} != {expected['http_status']}")
    if "decision" in expected and actual.get("decision") != expected["decision"]:
        fails.append(f"decision {actual.get('decision')!r} != {expected['decision']!r}")
    if actual["max_llm_calls"] > expected.get("max_llm_calls", 4):
        fails.append(f"used {actual['max_llm_calls']} LLM calls")
    if "notification_contains" in expected:
        want = expected["notification_contains"]["text"].lower()
        if not any(want in n.lower() for n in actual["notifications"]):
            fails.append(f"no notification containing {want!r}")
    return fails


def run_all(real: bool = False) -> list[dict]:
    cases = [json.loads(line) for line in (HERE / "cases.jsonl").read_text().splitlines() if line.strip()]
    results = []
    for case in cases:
        try:
            actual = run_case(case, real)
            fails = judge(case["expected"], actual)
        except Exception as e:  # a crashing case is a failed case, not a crashed run
            actual, fails = {"error": repr(e)}, [f"crashed: {e!r}"]
        results.append({"id": case["id"], "input": case["input"], "expected": case["expected"],
                        "actual": actual, "pass": not fails,
                        "notes": "; ".join(filter(None, [case.get("notes", "")] + fails))})
    return results


def main() -> None:
    real = "--real" in sys.argv
    results = run_all(real)
    out = HERE / ("results_real.jsonl" if real else "results.jsonl")
    out.write_text("".join(json.dumps(r, default=str) + "\n" for r in results))

    by_cat: dict[str, list[bool]] = {}
    for r in results:
        by_cat.setdefault(r["id"].split("-")[0], []).append(r["pass"])
    print(f"mode: {'REAL LLM' if real else 'fake LLM (scripted)'}")
    for cat, passes in by_cat.items():
        print(f"  {cat:5} {sum(passes)}/{len(passes)}")
    print(f"  total {sum(r['pass'] for r in results)}/{len(results)}  -> {out.name}")
    for r in results:
        if not r["pass"]:
            print(f"  FAIL {r['id']}: {r['notes']}")
    lat = sorted(ms for r in results for ms in r["actual"].get("latencies_ms", []))
    if lat:
        p95 = lat[max(0, int(len(lat) * 0.95) - 1)]
        print(f"  latency per message: p50 {statistics.median(lat):.0f} ms, p95 {p95} ms, max {lat[-1]} ms (n={len(lat)})")
    sys.exit(0 if all(r["pass"] for r in results) else 1)  # CI / scripts see failures


if __name__ == "__main__":
    main()
