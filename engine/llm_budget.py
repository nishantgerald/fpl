"""Hard limits on language-model calls, because the endpoint faces the internet.

The model is reached through the Claude CLI, which authenticates with the
operator's personal Claude subscription. That inverts the usual threat model.
There is no bill to cap and no per-request charge to notice — the scarce resource
is the subscription's own rate-limit window, and it is *shared with the
operator's other work*. An abuser who exhausts it doesn't run up a charge; they
take out the operator's Claude Code sessions as collateral. That is the damage
this module exists to bound.

Three things had to be true before narration could be turned on, and none of them
were:

*Caching is not rate limiting.* :mod:`engine.fcps_llm` caches per
``(entry, gameweek, model)`` for a day. That stops the same manager re-triggering
a call; it does nothing about someone walking ``user_id=1,2,3,...`` through the
several million entry IDs FPL hands out, each of which is a fresh key and a
fresh call.

*Narration multiplies.* ``narrative.annotate`` was written to narrate every plan
the optimiser returned, up to five, on a route with no LLM cache at all. Turned
on as it stood, one request to ``/api/recommendations`` was five calls, and the
same request repeated was five more.

*Concurrency is unbounded.* Each call forks a Node process that idles around a
couple of hundred megabytes. Nothing stopped a hundred of them existing at once,
which exhausts memory and PIDs long before it exhausts any quota.

So: a global daily ceiling that no caller can raise, a cap on simultaneous
processes, and a fail-closed default. All three are enforced here rather than at
the route, so a new route can't forget them.

Both limits are held on disk, not in memory. The app is served by gunicorn with
more than one worker; a per-process counter would multiply the ceiling by the
worker count, which is precisely the mistake that makes a limit look enforced
while not being one.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import time
from pathlib import Path

from . import db

# The ceiling is deliberately low. Legitimate traffic is one call per manager per
# day for FCPS plus a handful of narrations; a few hundred covers real use with
# room to spare, and anything past it is far likelier to be abuse than a good
# day. Raise it knowingly, not reflexively.
DAILY_CALL_CEILING = int(os.getenv("LLM_DAILY_CEILING", "250"))

# Simultaneous CLI processes. Each is a Node runtime; this is a memory bound as
# much as a fairness one.
MAX_CONCURRENT_CALLS = int(os.getenv("LLM_MAX_CONCURRENT", "2"))

STATE_DIR = Path(
    os.getenv("LLM_BUDGET_DIR", Path.home() / ".cache" / "fpl" / "llm-budget")
)


class BudgetExhausted(Exception):
    """The global daily ceiling is spent. Retrying today will not help."""


class TooBusy(Exception):
    """Every concurrency slot is occupied. Retrying shortly may help."""


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


@contextlib.contextmanager
def reserve(kind: str):
    """Hold a concurrency slot and one unit of the daily ceiling.

    The ceiling is consumed on entry rather than on success. A call that fails
    after the model ran still cost the subscription its tokens, and a failure
    loop is exactly the shape of traffic a ceiling exists to stop — refunding it
    would turn every error into free retries.

    Raises :class:`BudgetExhausted` or :class:`TooBusy`; both are the caller's to
    translate into whatever the route should return.
    """
    _spend_one(kind)
    with _slot():
        yield


# ------------------------------------------------------------------ daily count


def _spend_one(kind: str) -> None:
    """Increment today's counter, or refuse. Shared across every worker."""
    if db.is_postgres():
        return _spend_one_shared(kind)
    _spend_one_on_disk(kind)


def _spend_one_shared(kind: str) -> None:
    """The counter, in the database.

    On a dyno the file version counted nothing: the filesystem is wiped on every
    restart, so a 250-a-day ceiling silently reset several times a day and each
    of the three workers kept its own idea of the total besides.

    The check and the increment are one statement. Read-then-write would let two
    workers both see 249 and both spend, which is exactly the shape of traffic a
    ceiling exists to stop; `WHERE total < ceiling` inside the upsert makes the
    refusal atomic, and returning no row *is* the refusal.
    """
    today = _today()
    with db.connect() as conn:
        row = conn.execute(
            """
            INSERT INTO llm_budget_day (day, total) VALUES (?, 1)
            ON CONFLICT (day) DO UPDATE SET total = llm_budget_day.total + 1
            WHERE llm_budget_day.total < ?
            RETURNING total
            """,
            (today, DAILY_CALL_CEILING),
        ).fetchone()
        if row is None:
            raise BudgetExhausted(
                f"The daily model-call ceiling ({DAILY_CALL_CEILING}) is spent."
            )
        # Reporting only, so a failure here must not undo a spend that already
        # happened — the ceiling is the guarantee, the breakdown is a nicety.
        conn.execute(
            """
            INSERT INTO llm_budget_kind (day, kind, total) VALUES (?, ?, 1)
            ON CONFLICT (day, kind) DO UPDATE
                SET total = llm_budget_kind.total + 1
            """,
            (today, kind),
        )
        # Yesterday's rows are dropped rather than accumulated, so the table
        # cannot grow without bound.
        conn.execute("DELETE FROM llm_budget_day WHERE day <> ?", (today,))
        conn.execute("DELETE FROM llm_budget_kind WHERE day <> ?", (today,))


def _spend_one_on_disk(kind: str) -> None:
    """Increment today's counter under an exclusive lock, or refuse."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = STATE_DIR / "calls.json"

    # Opened "a+" so the file is created if absent without truncating it if not.
    with open(path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            raw = handle.read()
            try:
                state = json.loads(raw) if raw.strip() else {}
            except ValueError:
                state = {}

            today = _today()
            # Yesterday's counts are dropped rather than accumulated, so the
            # file cannot grow without bound.
            if state.get("date") != today:
                state = {"date": today, "total": 0, "by_kind": {}}

            if state["total"] >= DAILY_CALL_CEILING:
                raise BudgetExhausted(
                    f"The daily model-call ceiling ({DAILY_CALL_CEILING}) is spent."
                )

            state["total"] += 1
            state["by_kind"][kind] = state["by_kind"].get(kind, 0) + 1

            handle.seek(0)
            handle.truncate()
            json.dump(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def spent_today() -> dict:
    """Today's usage, for the status endpoint. Never raises."""
    if db.is_postgres():
        try:
            with db.connect() as conn:
                day = conn.execute(
                    "SELECT total FROM llm_budget_day WHERE day = ?", (_today(),)
                ).fetchone()
                kinds = conn.execute(
                    "SELECT kind, total FROM llm_budget_kind WHERE day = ?",
                    (_today(),),
                ).fetchall()
            return {
                "date": _today(),
                "total": int(day["total"]) if day else 0,
                "by_kind": {r["kind"]: int(r["total"]) for r in kinds},
            }
        except Exception:
            # A status endpoint must not 500 because the database blinked.
            return {"date": _today(), "total": 0, "by_kind": {}}
    try:
        raw = (STATE_DIR / "calls.json").read_text("utf-8")
        state = json.loads(raw)
        if state.get("date") != _today():
            return {"date": _today(), "total": 0, "by_kind": {}}
        return state
    except (OSError, ValueError):
        return {"date": _today(), "total": 0, "by_kind": {}}


def remaining_today() -> int:
    return max(0, DAILY_CALL_CEILING - int(spent_today().get("total", 0)))


# ------------------------------------------------------------------ per client


# The global ceiling alone is a blunt instrument: it bounds the damage but one
# abuser can still spend the whole day's allowance in a loop and lock out every
# legitimate manager. A per-client limit is what keeps the ceiling *fairly*
# distributed rather than merely finite.
CLIENT_CALLS_PER_HOUR = int(os.getenv("LLM_CLIENT_HOURLY", "10"))
_CLIENT_WINDOW_SECONDS = 3600

# Bounded so a spray of forged addresses can't grow the file without limit. When
# full the oldest entries are dropped, which fails *open* for new clients rather
# than locking everyone out — the global ceiling is still behind it.
_CLIENT_TABLE_MAX = 4096


class ClientThrottled(Exception):
    """This caller has had its share of the hour."""


def check_client(client_id: str) -> None:
    """Record a call against ``client_id`` and refuse if it's over its share.

    ``client_id`` is normally an IP address, which is spoofable and shared by
    everyone behind a NAT. This is therefore a speed bump for casual abuse, not
    an identity control — the global ceiling is the guarantee. It is enforced
    here, on disk, rather than per-process, because gunicorn runs several
    workers and a per-process table would multiply the limit by the worker
    count.
    """
    if not client_id:
        return

    if db.is_postgres():
        return _check_client_shared(client_id)
    _check_client_on_disk(client_id)


def throttled(scope: str, client_id: str, limit: int, window_seconds: int) -> bool:
    """Generic per-client rate limit, shared across workers. True = refuse.

    Exists because the same mistake was made three times: the import limit and
    the auth limit were plain dicts in process memory, so with three gunicorn
    workers a "5 uploads per 15 minutes" cap was really fifteen, a "10 attempts"
    cap was really thirty, and both forgot everything on every restart. A limit
    multiplied by the worker count and reset several times a day is not a limit.

    Falls back to a process-local dict when there is no database, which is
    correct for a single-process dev server and is what the tests exercise.
    """
    if not client_id:
        return False
    key = f"{scope}:{client_id}"
    if not db.is_postgres():
        return _throttled_in_process(key, limit, window_seconds)

    now = time.time()
    cutoff = now - window_seconds
    try:
        with db.connect() as conn:
            conn.execute("DELETE FROM llm_client_calls WHERE called_at < ?", (cutoff,))
            row = conn.execute(
                """
                INSERT INTO llm_client_calls (client_id, called_at)
                SELECT ?, ?
                WHERE (
                    SELECT COUNT(*) FROM llm_client_calls
                    WHERE client_id = ? AND called_at > ?
                ) < ?
                RETURNING called_at
                """,
                (key, now, key, cutoff, limit),
            ).fetchone()
        return row is None
    except Exception:
        # A throttle that cannot reach its store must not take the route down
        # with it. Failing open is the right direction: the daily ceiling is
        # still behind this, and refusing every request because the database
        # blinked is a worse outage than a brief lapse in rate limiting.
        return False


_local_windows: dict[str, list[float]] = {}


def _throttled_in_process(key: str, limit: int, window_seconds: int) -> bool:
    now = time.time()
    window = [t for t in _local_windows.get(key, []) if now - t < window_seconds]
    if len(window) >= limit:
        _local_windows[key] = window
        return True
    window.append(now)
    _local_windows[key] = window
    return False


def _check_client_shared(client_id: str) -> None:
    """The hourly throttle, in the database.

    One statement again, for the same reason as the daily ceiling: counting and
    then inserting lets two concurrent requests both pass a limit neither should
    have. The insert only happens if the count inside the window is under the
    limit, so no row returned means throttled.
    """
    now = time.time()
    cutoff = now - _CLIENT_WINDOW_SECONDS
    with db.connect() as conn:
        # Prune globally rather than per client, or the table only ever grows.
        conn.execute("DELETE FROM llm_client_calls WHERE called_at < ?", (cutoff,))
        row = conn.execute(
            """
            INSERT INTO llm_client_calls (client_id, called_at)
            SELECT ?, ?
            WHERE (
                SELECT COUNT(*) FROM llm_client_calls
                WHERE client_id = ? AND called_at > ?
            ) < ?
            RETURNING called_at
            """,
            (client_id, now, client_id, cutoff, CLIENT_CALLS_PER_HOUR),
        ).fetchone()
    if row is None:
        raise ClientThrottled(
            f"This client has made {CLIENT_CALLS_PER_HOUR} model calls in the "
            f"last hour; that is the limit."
        )


def _check_client_on_disk(client_id: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = STATE_DIR / "clients.json"
    now = time.time()
    cutoff = now - _CLIENT_WINDOW_SECONDS

    with open(path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            raw = handle.read()
            try:
                table = json.loads(raw) if raw.strip() else {}
            except ValueError:
                table = {}

            # Prune globally, not just for this client, or the file only ever
            # grows and every read gets slower.
            table = {
                key: [t for t in stamps if t > cutoff]
                for key, stamps in table.items()
                if isinstance(stamps, list)
            }
            table = {key: stamps for key, stamps in table.items() if stamps}

            stamps = table.get(client_id, [])
            if len(stamps) >= CLIENT_CALLS_PER_HOUR:
                raise ClientThrottled(
                    f"This client has made {len(stamps)} model calls in the last "
                    f"hour; the limit is {CLIENT_CALLS_PER_HOUR}."
                )

            stamps.append(now)
            table[client_id] = stamps

            if len(table) > _CLIENT_TABLE_MAX:
                # Keep the most recently active. Dropping the rest lets those
                # clients start a fresh window, which is the safe direction.
                ordered = sorted(table.items(), key=lambda kv: max(kv[1]), reverse=True)
                table = dict(ordered[:_CLIENT_TABLE_MAX])

            handle.seek(0)
            handle.truncate()
            json.dump(table, handle)
            handle.flush()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


# ------------------------------------------------------------------ concurrency


@contextlib.contextmanager
def _slot():
    """Occupy one of ``MAX_CONCURRENT_CALLS`` slots for the duration.

    Implemented as N lock files tried non-blockingly rather than as a counter,
    because a counter has to be decremented and a process killed mid-call never
    gets to decrement it. A flock is released by the kernel when the file
    descriptor closes, whatever the reason it closed, so a crash cannot leak a
    slot permanently.
    """
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    handles = []
    try:
        for index in range(MAX_CONCURRENT_CALLS):
            handle = open(STATE_DIR / f"slot-{index}.lock", "w", encoding="utf-8")
            handles.append(handle)
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                continue  # taken; try the next
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return
        raise TooBusy(
            f"All {MAX_CONCURRENT_CALLS} model-call slots are busy. Try again shortly."
        )
    finally:
        for handle in handles:
            with contextlib.suppress(OSError):
                handle.close()
