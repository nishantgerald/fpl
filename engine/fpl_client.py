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

import threading
import time
from collections import OrderedDict
from typing import Any, Callable

import requests

BASE = "https://fantasy.premierleague.com/api"

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

PHOTO_CACHE_SIZE = 512


class UpstreamUnavailable(Exception):
    """Upstream failed and we have no cached value to fall back on."""


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


def _get_json(path: str) -> Any:
    """One upstream GET. Returns ``None`` for a 404 so callers can distinguish."""
    response = requests.get(f"{BASE}{path}", headers=HEADERS, timeout=TIMEOUT)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


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


class _PhotoCache:
    """Bounded LRU for player photos. Content is immutable, so no TTL."""

    def __init__(self, size: int = PHOTO_CACHE_SIZE):
        self._data: OrderedDict[int, bytes] = OrderedDict()
        self._size = size
        self._lock = threading.Lock()

    def get(self, code: int) -> bytes | None:
        with self._lock:
            if code not in self._data:
                return None
            self._data.move_to_end(code)
            return self._data[code]

    def put(self, code: int, value: bytes) -> None:
        with self._lock:
            self._data[code] = value
            self._data.move_to_end(code)
            while len(self._data) > self._size:
                self._data.popitem(last=False)


_photos = _PhotoCache()


def photo(code: int) -> bytes | None:
    cached = _photos.get(code)
    if cached is not None:
        return cached
    url = (
        "https://resources.premierleague.com/premierleague/photos/players"
        f"/110x140/p{code}.png"
    )
    try:
        response = requests.get(url, timeout=TIMEOUT)
    except Exception:
        return None
    if response.status_code != 200:
        return None
    _photos.put(code, response.content)
    return response.content


def clear_caches() -> None:
    """Test hook."""
    _cache.clear()
