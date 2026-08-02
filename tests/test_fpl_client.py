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
