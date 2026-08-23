"""All upstream I/O, behind one TTL cache.

The old code had a module-level ``cached_data`` global that was set once and
never invalidated, so a long-lived gunicorn worker served the prices and
gameweek number it booted with, forever. It was also inconsistent — some helpers
used the cache and some bypassed it, so a single ``/api/players`` request could
trigger five separate 1.5 MB fetches of the same payload.

This module is the only place in the app that talks to fantasy.premierleague.com.
Everything it exposes is cached with an endpoint-appropriate TTL, has a hard
timeout, and degrades to stale data rather than failing when upstream is down.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import requests

BASE = "https://fantasy.premierleague.com/api"

# Where the manager's FPL session cookie lives, if they've set one up. Outside
# the repo by design: a credential in the working tree is one `git add -A` away
# from being published.
SESSION_FILE = Path(
    os.getenv(
        "FPL_SESSION_FILE",
        Path.home() / ".openclaw" / "credentials" / "fpl-session",
    )
)

# (connect, read). The connect value sits just off a multiple of 3s so it
# doesn't line up with TCP retransmit windows.
TIMEOUT = (3.05, 12)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; fpl.nishantgerald.com)"}

# Prices change nightly, news lands hourly; 10 minutes is well inside both and
# keeps upstream load trivial.
TTL_BOOTSTRAP = 600
TTL_FIXTURES = 3600
TTL_ENTRY = 300
TTL_PICKS = 300
TTL_HISTORY = 1800
# Short: a pre-deadline draft changes as the manager edits it, and showing them
# a squad they've already changed is worse than a slightly slower page.
TTL_MY_TEAM = 60
# Standings only move when a gameweek scores.
TTL_LEAGUE = 900

PHOTO_CACHE_SIZE = 512


class UpstreamUnavailable(Exception):
    """Upstream failed and we have no cached value to fall back on."""


class NotAuthenticated(Exception):
    """No usable session cookie. An ordinary state, not a failure.

    Distinct from :class:`UpstreamUnavailable` because it must never degrade to
    a cached value: upstream answered fine, it just declined us.
    """


class _Entry:
    __slots__ = ("value", "fetched_at")

    def __init__(self, value: Any, fetched_at: float):
        self.value = value
        self.fetched_at = fetched_at


class _Cache:
    """A tiny TTL cache with stale-while-error semantics.

    Deliberately in-process: one gunicorn box with an in-memory cache is the
    right size for this app. The interface is narrow enough that swapping in
    Redis later is a change to this class alone.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.Lock()

    def get(self, key: str, ttl: int, fetch: Callable[[], Any]) -> tuple[Any, float, bool]:
        """Return ``(value, fetched_at, stale)``.

        On a fresh hit, returns immediately. On a miss or expiry, fetches. If the
        fetch fails but we hold a stale value, that value is returned with
        ``stale=True`` — an FPL outage should degrade the app, not take it down.
        """
        with self._lock:
            entry = self._entries.get(key)

        now = time.time()
        if entry is not None and now - entry.fetched_at < ttl:
            return entry.value, entry.fetched_at, False

        try:
            value = fetch()
        except NotAuthenticated:
            # Not an outage: upstream answered, and the answer was "no". Serving
            # the stale entry here would pin the manager's squad to whenever the
            # cookie expired and never tell them, so this propagates untouched.
            raise
        except Exception as exc:
            if entry is not None:
                return entry.value, entry.fetched_at, True
            # One exception type for "upstream failed and we have nothing", so
            # callers don't have to know about requests' exception hierarchy.
            raise UpstreamUnavailable(key) from exc

        if value is None:
            if entry is not None:
                return entry.value, entry.fetched_at, True
            raise UpstreamUnavailable(key)

        with self._lock:
            self._entries[key] = _Entry(value, now)
        return value, now, False

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


_cache = _Cache()

# Freshness of the most recent read, so routes can report it without threading
# the value through every call site.
_last_meta = threading.local()


def _get_json(path: str, cookie: str | None = None) -> Any:
    """One upstream GET. Returns ``None`` for a 404 so callers can distinguish.

    ``cookie`` authenticates as the signed-in manager. It is passed per call
    rather than held on a session object so it can never leak onto a request
    that didn't ask for it.
    """
    headers = dict(HEADERS)
    if cookie:
        headers["Cookie"] = cookie
    response = requests.get(f"{BASE}{path}", headers=headers, timeout=TIMEOUT)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def _session_cookie() -> str | None:
    """The manager's FPL session cookie, if one has been provided.

    Read fresh on each use rather than cached at import, so replacing an expired
    cookie takes effect without a restart. Absent is the normal case and must
    never be an error — the app works without it, just with less.
    """
    from_env = os.getenv("FPL_SESSION_COOKIE", "").strip()
    if from_env:
        return from_env
    try:
        raw = SESSION_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    # Tolerates a whole `Cookie:` header being pasted in, which is what a
    # browser's "copy as cURL" hands you.
    if raw.lower().startswith("cookie:"):
        raw = raw.split(":", 1)[1].strip()
    return raw or None


def has_session() -> bool:
    """Whether an authenticated read is even possible. Never returns the value."""
    return _session_cookie() is not None


def session_status(entry_id: int) -> dict:
    """Whether the stored cookie still authenticates, as a reportable fact.

    A session cookie dies silently: it works until it doesn't, and the first
    sign is usually a screen that stopped filling in. This makes the state
    checkable on demand — and on a schedule — so the answer arrives before the
    manager notices something is wrong rather than after.

    Never returns any part of the cookie, so it is safe to log, cache, expose
    on a route and send over a chat notification.
    """
    if not has_session():
        return {
            "state": "absent",
            "ok": False,
            "detail": "No FPL session cookie configured.",
        }

    # Deliberately bypasses the TTL cache: a cached success would report a
    # cookie healthy for a full minute after it stopped working, which defeats
    # the point of asking.
    try:
        payload = _get_json(f"/my-team/{entry_id}/", cookie=_session_cookie())
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status in (401, 403):
            return {
                "state": "expired",
                "ok": False,
                "detail": "The FPL session cookie is no longer accepted. "
                "Sign in again and replace it.",
            }
        return {
            "state": "unknown",
            "ok": False,
            "detail": f"Upstream returned {status}.",
        }
    except requests.RequestException as exc:
        # An outage says nothing about the cookie; don't cry wolf.
        return {"state": "unknown", "ok": False, "detail": f"Upstream error: {exc}"}

    if payload is None:
        return {
            "state": "unknown",
            "ok": False,
            "detail": f"Entry {entry_id} did not resolve.",
        }
    return {
        "state": "valid",
        "ok": True,
        "detail": f"Authenticated; {len(payload.get('picks') or [])} players visible.",
    }


def my_team(entry_id: int, cookie: str | None = None) -> dict | None:
    """The signed-in manager's squad — including a squad drafted pre-deadline.

    The public picks endpoint 404s until the first deadline passes, because FPL
    publishes nobody's team before then. This is the endpoint the FPL site
    itself uses while you're building a squad, and it carries what the public
    one lacks: selling prices, purchase prices, bank and free transfers.

    ``cookie`` authenticates as a specific manager — the multi-user path, where
    each account brings its own. Omitted, the deployment-wide cookie file/env
    is used, which is the single-user path.

    Raises :class:`NotAuthenticated` when there's no cookie or the cookie no
    longer authenticates — the caller should degrade, not 500. Returns ``None``
    when the id doesn't resolve.
    """
    cookie = cookie or _session_cookie()
    if not cookie:
        raise NotAuthenticated("No FPL session cookie configured.")

    def fetch():
        try:
            return _get_json(f"/my-team/{entry_id}/", cookie=cookie)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in (401, 403):
                # An expired cookie must not be served from cache as though it
                # were a transient blip -- that would show a stale squad
                # indefinitely and never prompt a refresh.
                raise NotAuthenticated("FPL session cookie is expired or invalid.")
            raise

    try:
        value, fetched_at, stale = _cache.get(
            f"my_team:{entry_id}", TTL_MY_TEAM, fetch
        )
    except UpstreamUnavailable:
        return None
    _record(fetched_at, stale)
    return _as_picks_payload(value)


def league_standings(league_id: int, page: int = 1) -> dict | None:
    """One page of a classic league table. Public — no cookie.

    Cached for longer than picks: standings only move when a gameweek scores,
    and a mini-league page that re-fetches per visitor would multiply one
    league's traffic by its membership.
    """
    try:
        value, fetched_at, stale = _cache.get(
            f"league:{league_id}:{page}",
            TTL_LEAGUE,
            lambda: _get_json(
                f"/leagues-classic/{league_id}/standings/?page_standings={page}"
            ),
        )
    except UpstreamUnavailable:
        return None
    _record(fetched_at, stale)
    return value


def transfers(entry_id: int) -> list | None:
    """Every transfer the entry has made this season. Public — no cookie.

    Returns newest-first, as upstream does. ``None`` when the id doesn't
    resolve; an empty list is the legitimate "no transfers yet".
    """
    try:
        value, fetched_at, stale = _cache.get(
            f"transfers:{entry_id}",
            TTL_PICKS,
            lambda: _get_json(f"/entry/{entry_id}/transfers/"),
        )
    except UpstreamUnavailable:
        return None
    _record(fetched_at, stale)
    return value


def _as_picks_payload(payload: dict | None) -> dict | None:
    """Reshape a ``/my-team/`` response to look like a ``/picks/`` one.

    The two endpoints disagree about where the same facts live: picks puts bank
    and squad value under ``entry_history``, my-team under ``transfers``. The
    difference is an upstream quirk, so it's absorbed here rather than made the
    caller's problem.
    """
    if not payload or not payload.get("picks"):
        return None

    transfers = payload.get("transfers") or {}
    active_chip = next(
        (
            chip.get("name")
            for chip in (payload.get("chips") or [])
            if chip.get("status_for_entry") == "active"
        ),
        None,
    )

    picks = []
    for pick in payload["picks"]:
        pick = dict(pick)
        # my-team omits `multiplier` on some responses. Positions 1-11 are the
        # XI and 12-15 the bench in both payloads, so derive it rather than
        # defaulting everyone to the bench.
        if "multiplier" not in pick:
            starting = int(pick.get("position", 99)) <= 11
            pick["multiplier"] = (2 if pick.get("is_captain") else 1) if starting else 0
        picks.append(pick)

    return {
        "picks": picks,
        "entry_history": {
            "bank": transfers.get("bank"),
            "value": transfers.get("value"),
        },
        "active_chip": active_chip,
        # Only my-team knows this, and it saves the optimiser a guess.
        "free_transfers": transfers.get("limit"),
        "source": "my_team",
    }


def _record(fetched_at: float, stale: bool) -> None:
    previous_stale = getattr(_last_meta, "stale", False)
    previous_at = getattr(_last_meta, "fetched_at", None)
    _last_meta.stale = previous_stale or stale
    # Report the *oldest* read in the request, which is the honest answer to
    # "how fresh is this page?".
    if previous_at is None or fetched_at < previous_at:
        _last_meta.fetched_at = fetched_at


def begin_request() -> None:
    """Reset per-request freshness tracking."""
    _last_meta.stale = False
    _last_meta.fetched_at = None


def meta(extra: dict | None = None) -> dict:
    """The ``meta`` block every API response carries."""
    fetched_at = getattr(_last_meta, "fetched_at", None)
    out = {
        "fetched_at": (
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(fetched_at))
            if fetched_at
            else None
        ),
        "stale": bool(getattr(_last_meta, "stale", False)),
    }
    if extra:
        out.update(extra)
    return out


# ---------------------------------------------------------------- endpoints


def bootstrap() -> dict:
    value, fetched_at, stale = _cache.get(
        "bootstrap", TTL_BOOTSTRAP, lambda: _get_json("/bootstrap-static/")
    )
    _record(fetched_at, stale)
    return value


def fixtures() -> list:
    value, fetched_at, stale = _cache.get(
        "fixtures", TTL_FIXTURES, lambda: _get_json("/fixtures/")
    )
    _record(fetched_at, stale)
    return value


def entry(entry_id: int) -> dict | None:
    """Manager entry. ``None`` means the id doesn't resolve this season."""
    try:
        value, fetched_at, stale = _cache.get(
            f"entry:{entry_id}", TTL_ENTRY, lambda: _get_json(f"/entry/{entry_id}/")
        )
    except UpstreamUnavailable:
        return None
    _record(fetched_at, stale)
    return value


def picks(entry_id: int, gameweek: int) -> dict | None:
    try:
        value, fetched_at, stale = _cache.get(
            f"picks:{entry_id}:{gameweek}",
            TTL_PICKS,
            lambda: _get_json(f"/entry/{entry_id}/event/{gameweek}/picks/"),
        )
    except UpstreamUnavailable:
        return None
    _record(fetched_at, stale)
    return value


def history(entry_id: int) -> dict | None:
    try:
        value, fetched_at, stale = _cache.get(
            f"history:{entry_id}",
            TTL_HISTORY,
            lambda: _get_json(f"/entry/{entry_id}/history/"),
        )
    except UpstreamUnavailable:
        return None
    _record(fetched_at, stale)
    return value


def element_summary(player_id: int) -> dict | None:
    """One player's per-gameweek record: ``history``, ``history_past``, fixtures.

    A per-player call, so it is cached like the rest. The bootstrap carries a
    player's season totals and his score for the current gameweek and nothing in
    between — this is the only route to what he did three weeks ago.
    """
    try:
        value, fetched_at, stale = _cache.get(
            f"element-summary:{player_id}",
            TTL_HISTORY,
            lambda: _get_json(f"/element-summary/{player_id}/"),
        )
    except UpstreamUnavailable:
        return None
    _record(fetched_at, stale)
    return value


# ---------------------------------------------------------------- derived


def season_state() -> dict | None:
    """Where we are in the season.

    Before the GW1 deadline no event is current and FPL serves no picks for any
    entry, so callers need to tell "not started yet" apart from a genuine lookup
    failure.
    """
    data = bootstrap()
    if not data:
        return None
    events = data.get("events") or []
    if not events:
        return None
    current = next((e for e in events if e.get("is_current")), None)
    upcoming = current or next((e for e in events if e.get("is_next")), events[0])
    return {
        "started": current is not None,
        "gameweek": int(upcoming["id"]),
        "gameweek_name": upcoming.get("name", ""),
        "deadline": upcoming.get("deadline_time"),
    }


def current_gameweek() -> int | None:
    state = season_state()
    return state["gameweek"] if state else None


# ---------------------------------------------------------------- photos


# Cached in place of the bytes when the CDN has no photo for a player, so a
# miss is remembered rather than re-fetched on every request.
_MISSING = object()


class _PhotoCache:
    """Bounded LRU for player photos. Content is immutable, so no TTL."""

    def __init__(self, size: int = PHOTO_CACHE_SIZE):
        self._data: OrderedDict[int, object] = OrderedDict()
        self._size = size
        self._lock = threading.Lock()

    def get(self, code: int):
        with self._lock:
            if code not in self._data:
                return None
            self._data.move_to_end(code)
            return self._data[code]

    def put(self, code: int, value) -> None:
        with self._lock:
            self._data[code] = value
            self._data.move_to_end(code)
            while len(self._data) > self._size:
                self._data.popitem(last=False)


_photos = _PhotoCache()


# Where a player photo might live, in the order worth trying.
#
# The Premier League moved its player art to a season-stamped path and dropped
# the "p" prefix from the filename: p223094.png under /premierleague became
# 223094.png under /premierleague25. The old path was not retired, and the two
# do not hold the same players — of 581 elements, 410 are only reachable at the
# new path and 67 only at the old one. Asking for one and giving up is how two
# hundred players ended up with no face.
#
# Measured 2026-08-09: old path alone 380/581 (65%), new path alone 410 (71%),
# both in this order 477 (82%). The remaining 104 are absent from the CDN
# altogether — mostly deadline-day signings and academy call-ups — and FPL's own
# site shows a placeholder for them too.
#
# The season segment changes each August. When it does, add the new one at the
# top rather than editing this one: a player who has not been rephotographed is
# still served from the older path.
PHOTO_SOURCES: tuple[str, ...] = (
    "https://resources.premierleague.com/premierleague25/photos/players"
    "/110x140/{code}.png",
    "https://resources.premierleague.com/premierleague/photos/players"
    "/110x140/p{code}.png",
)


def photo(code: int) -> bytes | None:
    """The player's headshot, or None if the CDN has no picture of them.

    Tries each source in :data:`PHOTO_SOURCES` in turn. A miss is cached as
    well as a hit: without that, every player the league has never
    photographed costs two upstream requests on every page that lists them.
    """
    cached = _photos.get(code)
    if cached is not None:
        return None if cached is _MISSING else cached

    answered = True
    for template in PHOTO_SOURCES:
        try:
            response = requests.get(template.format(code=code), timeout=TIMEOUT)
        except Exception:
            # A timeout says nothing about whether the picture exists, so this
            # source gets no vote either way.
            answered = False
            continue
        if response.status_code == 200 and response.content:
            _photos.put(code, response.content)
            return response.content

    # Only remember the miss if every source actually answered. Caching one
    # because the network was down would blank that player until the process
    # restarts.
    if answered:
        _photos.put(code, _MISSING)
    return None


def clear_caches() -> None:
    """Test hook."""
    _cache.clear()
