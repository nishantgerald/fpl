"""The limits that make an internet-facing, subscription-backed LLM route safe.

The property under test throughout is that a limit *actually binds* — a limit
that can be walked past by varying an input, or that multiplies by the worker
count, is worse than none, because it reads as protection while providing none.
"""

import json
import multiprocessing
import time

import pytest

from engine import llm_budget


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_budget, "STATE_DIR", tmp_path / "budget")


def test_the_daily_ceiling_binds(monkeypatch):
    monkeypatch.setattr(llm_budget, "DAILY_CALL_CEILING", 3)

    for _ in range(3):
        with llm_budget.reserve("fcps"):
            pass

    with pytest.raises(llm_budget.BudgetExhausted):
        with llm_budget.reserve("fcps"):
            pass


def test_the_ceiling_is_global_not_per_kind(monkeypatch):
    """FCPS and narration draw on one subscription, so they share one ceiling."""
    monkeypatch.setattr(llm_budget, "DAILY_CALL_CEILING", 2)

    with llm_budget.reserve("fcps"):
        pass
    with llm_budget.reserve("narrative"):
        pass

    with pytest.raises(llm_budget.BudgetExhausted):
        with llm_budget.reserve("fcps"):
            pass


def test_a_failed_call_still_costs_its_budget(monkeypatch):
    """Otherwise an error loop is an unmetered retry loop."""
    monkeypatch.setattr(llm_budget, "DAILY_CALL_CEILING", 1)

    with pytest.raises(RuntimeError):
        with llm_budget.reserve("fcps"):
            raise RuntimeError("the model call blew up")

    assert llm_budget.remaining_today() == 0
    with pytest.raises(llm_budget.BudgetExhausted):
        with llm_budget.reserve("fcps"):
            pass


def test_the_count_resets_on_a_new_day(monkeypatch):
    monkeypatch.setattr(llm_budget, "DAILY_CALL_CEILING", 1)
    with llm_budget.reserve("fcps"):
        pass
    assert llm_budget.remaining_today() == 0

    monkeypatch.setattr(llm_budget, "_today", lambda: "2099-01-01")
    assert llm_budget.remaining_today() == 1
    with llm_budget.reserve("fcps"):  # must not raise
        pass


def test_a_corrupt_counter_file_does_not_wedge_the_route(monkeypatch):
    """A truncated write must fail open to a fresh day, not raise forever."""
    llm_budget.STATE_DIR.mkdir(parents=True, exist_ok=True)
    (llm_budget.STATE_DIR / "calls.json").write_text("{not json", "utf-8")

    with llm_budget.reserve("fcps"):
        pass
    assert llm_budget.spent_today()["total"] == 1


def test_concurrency_slots_bind(monkeypatch):
    monkeypatch.setattr(llm_budget, "MAX_CONCURRENT_CALLS", 1)

    with llm_budget.reserve("fcps"):
        with pytest.raises(llm_budget.TooBusy):
            with llm_budget.reserve("narrative"):
                pass


def test_a_slot_is_released_after_use(monkeypatch):
    monkeypatch.setattr(llm_budget, "MAX_CONCURRENT_CALLS", 1)

    with llm_budget.reserve("fcps"):
        pass
    with llm_budget.reserve("fcps"):  # must not raise
        pass


def test_a_slot_is_released_even_when_the_body_raises(monkeypatch):
    """A crashed call must not leak a slot; that would starve the route."""
    monkeypatch.setattr(llm_budget, "MAX_CONCURRENT_CALLS", 1)

    with pytest.raises(RuntimeError):
        with llm_budget.reserve("fcps"):
            raise RuntimeError("boom")

    with llm_budget.reserve("fcps"):  # slot must be free again
        pass


def test_the_client_limit_binds():
    for _ in range(llm_budget.CLIENT_CALLS_PER_HOUR):
        llm_budget.check_client("198.51.100.7")

    with pytest.raises(llm_budget.ClientThrottled):
        llm_budget.check_client("198.51.100.7")


def test_clients_are_limited_independently():
    for _ in range(llm_budget.CLIENT_CALLS_PER_HOUR):
        llm_budget.check_client("198.51.100.7")

    llm_budget.check_client("203.0.113.9")  # a different caller is unaffected


def test_the_client_window_slides(monkeypatch):
    for _ in range(llm_budget.CLIENT_CALLS_PER_HOUR):
        llm_budget.check_client("198.51.100.7")

    later = time.time() + 3601
    monkeypatch.setattr(llm_budget.time, "time", lambda: later)
    llm_budget.check_client("198.51.100.7")  # the hour has rolled


def test_the_client_table_cannot_grow_without_bound(monkeypatch):
    """A spray of forged addresses must not grow the file indefinitely."""
    monkeypatch.setattr(llm_budget, "_CLIENT_TABLE_MAX", 10)

    for n in range(40):
        llm_budget.check_client(f"198.51.100.{n}")

    table = json.loads((llm_budget.STATE_DIR / "clients.json").read_text("utf-8"))
    assert len(table) <= 10


def _spend(state_dir, ceiling, results):
    """Run in a separate process, as gunicorn workers are."""
    from engine import llm_budget as fresh

    fresh.STATE_DIR = state_dir
    fresh.DAILY_CALL_CEILING = ceiling
    granted = 0
    for _ in range(10):
        try:
            with fresh.reserve("fcps"):
                granted += 1
        except (fresh.BudgetExhausted, fresh.TooBusy):
            pass
    results.put(granted)


def test_the_ceiling_holds_across_processes(tmp_path):
    """The bug this guards: a per-process counter multiplies by worker count.

    gunicorn forks several workers. An in-memory counter would let each of them
    grant the full ceiling independently, so a "250/day" limit would really be
    250 x workers. The counter is on disk under a lock precisely so this test
    can hold.
    """
    ceiling = 5
    state_dir = tmp_path / "shared"
    # "spawn" rather than the default fork: pytest is multi-threaded, and
    # forking from a threaded parent is a deadlock risk. Spawn also models
    # gunicorn's independent workers more honestly, since nothing is inherited.
    context = multiprocessing.get_context("spawn")
    results = context.Queue()

    workers = [
        context.Process(target=_spend, args=(state_dir, ceiling, results))
        for _ in range(3)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)

    total_granted = sum(results.get() for _ in workers)
    assert total_granted == ceiling
