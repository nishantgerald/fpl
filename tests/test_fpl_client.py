"""The cache layer.

The old code cached bootstrap once and never invalidated it, so a long-lived
worker served the prices it booted with forever. These tests pin the TTL, the
stale-on-failure behaviour, and the photo cache bound.
"""

import pytest

from engine import fpl_client


@pytest.fixture(autouse=True)
def clear():
    fpl_client.clear_caches()
    fpl_client.begin_request()
    yield
    fpl_client.clear_caches()


def test_fresh_hits_do_not_refetch():
    calls = []

    def fetch():
        calls.append(1)
        return {"n": len(calls)}

    cache = fpl_client._Cache()
    first, _at, stale = cache.get("k", ttl=600, fetch=fetch)
    second, _at, _stale = cache.get("k", ttl=600, fetch=fetch)

    assert first == second == {"n": 1}
    assert len(calls) == 1
    assert stale is False


def test_expiry_refetches():
    calls = []

    def fetch():
        calls.append(1)
        return {"n": len(calls)}

    cache = fpl_client._Cache()
    cache.get("k", ttl=0, fetch=fetch)
    cache.get("k", ttl=0, fetch=fetch)
    assert len(calls) == 2


def test_upstream_failure_serves_stale_rather_than_raising():
    cache = fpl_client._Cache()
    cache.get("k", ttl=600, fetch=lambda: {"good": True})

    def boom():
        raise RuntimeError("FPL is down")

    value, _at, stale = cache.get("k", ttl=0, fetch=boom)
    assert value == {"good": True}
    assert stale is True


def test_upstream_failure_with_no_cache_raises_one_known_type():
    """Callers shouldn't have to know about requests' exception hierarchy."""
    cache = fpl_client._Cache()

    def boom():
        raise RuntimeError("FPL is down")

    with pytest.raises(fpl_client.UpstreamUnavailable):
        cache.get("k", ttl=600, fetch=boom)


def test_none_from_upstream_is_treated_as_unavailable():
    cache = fpl_client._Cache()
    with pytest.raises(fpl_client.UpstreamUnavailable):
        cache.get("k", ttl=600, fetch=lambda: None)


def test_photo_cache_is_bounded():
    cache = fpl_client._PhotoCache(size=3)
    for code in range(10):
        cache.put(code, b"x")
    assert len(cache._data) == 3
    assert cache.get(0) is None
    assert cache.get(9) == b"x"


def test_photo_cache_is_lru():
    cache = fpl_client._PhotoCache(size=2)
    cache.put(1, b"a")
    cache.put(2, b"b")
    cache.get(1)             # touch 1 so 2 becomes the eviction candidate
    cache.put(3, b"c")
    assert cache.get(1) == b"a"
    assert cache.get(2) is None


def test_meta_reports_staleness():
    fpl_client.begin_request()
    fpl_client._record(1_700_000_000.0, stale=True)
    meta = fpl_client.meta({"gameweek": 14})
    assert meta["stale"] is True
    assert meta["gameweek"] == 14
    assert meta["fetched_at"].endswith("Z")


def test_meta_reports_the_oldest_read_in_the_request():
    fpl_client.begin_request()
    fpl_client._record(1_700_000_500.0, stale=False)
    fpl_client._record(1_700_000_000.0, stale=False)
    assert fpl_client.meta()["fetched_at"] == "2023-11-14T22:13:20Z"


def test_every_upstream_call_has_a_timeout():
    assert isinstance(fpl_client.TIMEOUT, tuple)
    assert all(t > 0 for t in fpl_client.TIMEOUT)


# --------------------------------------------------------------- photo sources
#
# The league moved player art to a season-stamped path and dropped the "p"
# prefix from the filename, without retiring the old path or copying everyone
# across. Asking only the old one left roughly a third of the game faceless —
# Isidor among them — so `photo` tries both. These tests pin that it does, and
# that a player the CDN has never photographed costs one round of requests
# rather than one per view.


class _Response:
    def __init__(self, status_code, content=b""):
        self.status_code = status_code
        self.content = content


def _serve(available: dict, calls: list):
    """A fake requests.get where only `available` URLs return an image."""

    def get(url, **kwargs):
        calls.append(url)
        if url in available:
            return _Response(200, available[url])
        return _Response(403)

    return get


def _url(template: str, code: int) -> str:
    return template.format(code=code)


def test_photo_tries_the_season_path_first(monkeypatch):
    fpl_client._photos = fpl_client._PhotoCache()
    calls: list[str] = []
    new = _url(fpl_client.PHOTO_SOURCES[0], 437505)
    monkeypatch.setattr(fpl_client.requests, "get", _serve({new: b"png"}, calls))

    assert fpl_client.photo(437505) == b"png"
    assert calls == [new], "the newer path should answer without a second call"


def test_photo_falls_back_to_the_legacy_path(monkeypatch):
    """The case that is *not* obvious: 67 of 581 players live only here."""
    fpl_client._photos = fpl_client._PhotoCache()
    calls: list[str] = []
    old = _url(fpl_client.PHOTO_SOURCES[1], 232905)
    monkeypatch.setattr(fpl_client.requests, "get", _serve({old: b"png"}, calls))

    assert fpl_client.photo(232905) == b"png"
    assert len(calls) == 2, "should have tried the season path before this one"
    assert calls[-1] == old


def test_a_player_with_no_photo_anywhere_is_none(monkeypatch):
    fpl_client._photos = fpl_client._PhotoCache()
    calls: list[str] = []
    monkeypatch.setattr(fpl_client.requests, "get", _serve({}, calls))

    assert fpl_client.photo(620109) is None
    assert len(calls) == len(fpl_client.PHOTO_SOURCES)


def test_a_missing_photo_is_only_looked_up_once(monkeypatch):
    fpl_client._photos = fpl_client._PhotoCache()
    calls: list[str] = []
    monkeypatch.setattr(fpl_client.requests, "get", _serve({}, calls))

    fpl_client.photo(620109)
    before = len(calls)
    assert fpl_client.photo(620109) is None
    assert len(calls) == before, "a known-missing photo must not be re-fetched"


def test_a_network_failure_is_not_recorded_as_missing(monkeypatch):
    """A timeout says nothing about whether the picture exists."""
    fpl_client._photos = fpl_client._PhotoCache()
    attempts: list[str] = []

    def boom(url, **kwargs):
        attempts.append(url)
        raise RuntimeError("network down")

    monkeypatch.setattr(fpl_client.requests, "get", boom)
    assert fpl_client.photo(1) is None

    # Recovered: the next call tries again rather than trusting a cached miss.
    calls: list[str] = []
    good = _url(fpl_client.PHOTO_SOURCES[0], 1)
    monkeypatch.setattr(fpl_client.requests, "get", _serve({good: b"png"}, calls))
    assert fpl_client.photo(1) == b"png"


def test_an_empty_body_is_not_a_photo(monkeypatch):
    """A 200 with nothing in it would otherwise cache as a valid image."""
    fpl_client._photos = fpl_client._PhotoCache()
    calls: list[str] = []
    new = _url(fpl_client.PHOTO_SOURCES[0], 5)
    old = _url(fpl_client.PHOTO_SOURCES[1], 5)
    monkeypatch.setattr(
        fpl_client.requests, "get", _serve({new: b"", old: b"png"}, calls)
    )

    assert fpl_client.photo(5) == b"png"
