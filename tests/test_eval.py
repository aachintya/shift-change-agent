"""The evaluation set runs as part of the test suite (offline, fake LLM)."""
from eval.run import run_all


def test_eval_set_passes_offline():
    results = run_all(real=False)
    failed = [f"{r['id']}: {r['notes']}" for r in results if not r["pass"]]
    assert len(results) >= 30
    assert not failed, failed
