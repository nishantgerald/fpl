"""The actionable layer: advice joined to the fifteen players someone owns.

A fixture ticker is trivia until it is tied to a squad, so the risk here is not
that the arithmetic is wrong — it is that the squad silently goes missing and
the endpoint keeps answering anyway. Before the first deadline no squad is
public, which is exactly when advice is worth most, and the original version
returned nothing but generic buy tips in that window while still looking like
a full answer.
"""

import pytest

from engine import accounts, chips, service


BOOTSTRAP_CHIPS = [
    {"name": "wildcard", "start_event": 2, "stop_event": 19, "chip_type": "transfer"},
    {"name": "wildcard", "start_event": 20, "stop_event": 38, "chip_type": "transfer"},
    {"name": "bboost", "start_event": 1, "stop_event": 19, "chip_type": "team"},
    {"name": "bboost", "start_event": 20, "stop_event": 38, "chip_type": "team"},
]


# ------------------------------------------------------------- squad sourcing


def _elements(ids):
    return {"elements": [{"id": i, "web_name": f"P{i}", "team": 1} for i in ids]}


def test_pre_season_reads_the_squad_through_the_session_cookie(monkeypatch):
    """The window where advice matters most is the one with no public data.

    Everything is still changeable before the first deadline. Refusing to look
    then is refusing to help at the only moment nothing is yet committed.
    """
    monkeypatch.setattr(service.fpl_client, "bootstrap", lambda: _elements([1, 2, 3]))
    monkeypatch.setattr(
        service.fpl_client,
        "my_team",
        lambda entry, cookie=None: {"picks": [{"element": 1}, {"element": 2}]},
    )

    squad, picks, reason = service._squad_for_advice(
        1, gameweek=1, started=False, cookie="pl_profile=abc"
    )

    assert reason is None
    assert [p["id"] for p in squad] == [1, 2]
    assert len(picks) == 2


def test_pre_season_without_a_cookie_says_so_rather_than_guessing(monkeypatch):
    monkeypatch.setattr(service.fpl_client, "bootstrap", lambda: _elements([1]))

    squad, _, reason = service._squad_for_advice(
        1, gameweek=1, started=False, cookie=None
    )

    assert squad == []
    assert reason == "no_cookie"


def test_a_saved_screenshot_squad_stands_in_when_nothing_else_can_see_it(
    monkeypatch,
):
    """The screenshot importer's whole point is being the route that needs no
    credentials. Throwing the parse away at the end of the request made it a
    one-off toy rather than a way in."""
    monkeypatch.setattr(service.fpl_client, "bootstrap", lambda: _elements([4, 5, 6]))

    squad, picks, reason = service._squad_for_advice(
        1,
        gameweek=1,
        started=False,
        cookie=None,
        saved={"element_ids": [4, 5, 6], "bench_ids": [6]},
    )

    assert reason is None
    assert [p["id"] for p in squad] == [4, 5, 6]
    benched = {p["element"] for p in picks if p["multiplier"] == 0}
    assert benched == {6}


def test_an_expired_cookie_falls_back_rather_than_failing(monkeypatch):
    """In-season the public route still works. A dead cookie should cost the
    user the pre-deadline view, not the whole screen."""
    monkeypatch.setattr(service.fpl_client, "bootstrap", lambda: _elements([7]))

    def _dead(entry, cookie=None):
        raise service.fpl_client.NotAuthenticated("expired")

    monkeypatch.setattr(service.fpl_client, "my_team", _dead)
    monkeypatch.setattr(
        service,
        "load_squad_state",
        lambda entry, gw: {"squad": [{"id": 7}], "picks": [{"element": 7}]},
    )

    squad, _, reason = service._squad_for_advice(
        1, gameweek=5, started=True, cookie="stale"
    )

    assert reason is None
    assert [p["id"] for p in squad] == [7]


def test_a_live_cookie_is_preferred_in_season_too(monkeypatch):
    """Mid-season the public picks are frozen at the last deadline. Advising a
    transfer they already made reads as a bug in us, not advice."""
    monkeypatch.setattr(service.fpl_client, "bootstrap", lambda: _elements([8, 9]))
    monkeypatch.setattr(
        service.fpl_client, "my_team", lambda entry, cookie=None: {"picks": [{"element": 9}]}
    )
    monkeypatch.setattr(
        service,
        "load_squad_state",
        lambda entry, gw: pytest.fail("public picks used while a live cookie exists"),
    )

    squad, _, reason = service._squad_for_advice(
        1, gameweek=5, started=True, cookie="live"
    )

    assert reason is None
    assert [p["id"] for p in squad] == [9]


# -------------------------------------------------------------- chip windows


def test_only_the_current_half_of_chips_is_offered():
    """Both halves are legal and both are real. Listing both doubles the
    section with items up to nineteen gameweeks away and buries the four that
    can actually be played now."""
    windows = chips.windows(BOOTSTRAP_CHIPS)
    available = chips.available(windows, used=[], gameweek=3)

    assert len(available) == 4  # both halves of both chips

    this_half = [c for c in available if chips.in_current_half(c["start"], 3)]
    assert {c["name"] for c in this_half} == {"wildcard", "bboost"}
    assert all(c["start"] < chips.HALF_BOUNDARY for c in this_half)


def test_the_second_half_becomes_current_after_the_boundary():
    assert chips.in_current_half(20, gameweek=25)
    assert not chips.in_current_half(2, gameweek=25)
    assert chips.in_current_half(1, gameweek=19)


# ------------------------------------------------------------- saved squads


def test_a_saved_squad_survives_being_replaced(tmp_path, monkeypatch):
    """One squad per user, latest wins — a second upload is a correction, not
    an addition."""
    monkeypatch.setattr(accounts, "DB_PATH", tmp_path / "t.db")
    accounts.init_db()
    user = accounts.register("a@b.com", "correct horse battery")

    accounts.store_squad(user["id"], [1, 2, 3], [3])
    accounts.store_squad(user["id"], [4, 5], [5])

    saved = accounts.saved_squad(user["id"])
    assert saved["element_ids"] == [4, 5]
    assert saved["bench_ids"] == [5]
    assert saved["source"] == "screenshot"


def test_no_saved_squad_is_none_not_an_empty_shell(tmp_path, monkeypatch):
    monkeypatch.setattr(accounts, "DB_PATH", tmp_path / "t.db")
    accounts.init_db()
    user = accounts.register("c@d.com", "correct horse battery")

    assert accounts.saved_squad(user["id"]) is None


def test_the_preview_fallback_keeps_squad_level_advice(monkeypatch):
    """A preview exists so the feature can be judged before kickoff.

    Stripping it back to buy targets, the way a genuinely unreadable squad is
    stripped, would leave the preview showing exactly what it was meant to
    demonstrate isn't all there is.
    """
    monkeypatch.setattr(service.fpl_client, "bootstrap", lambda: _elements([1, 2]))
    monkeypatch.setattr(
        service,
        "draft_squad",
        lambda horizon=5: {"squad": [{"id": 1}, {"id": 2}], "bench": [{"id": 2}]},
    )

    squad, picks, reason = service._squad_for_advice(
        1, gameweek=1, started=False, cookie=None
    )

    assert reason == "preview_draft"
    assert [p["id"] for p in squad] == [1, 2]
    assert {p["element"] for p in picks if p["multiplier"] == 0} == {2}


def test_a_saved_squad_beats_the_preview(monkeypatch):
    """Their own squad, however it arrived, always outranks our suggestion."""
    monkeypatch.setattr(service.fpl_client, "bootstrap", lambda: _elements([1, 2, 3]))
    monkeypatch.setattr(
        service, "draft_squad", lambda horizon=5: pytest.fail("preview preferred")
    )

    squad, _, reason = service._squad_for_advice(
        1, gameweek=1, started=False, cookie=None, saved={"element_ids": [3]}
    )

    assert reason is None
    assert [p["id"] for p in squad] == [3]


def test_a_failed_draft_degrades_to_saying_nothing_is_visible(monkeypatch):
    """The preview is a nicety. It must never turn into the error itself."""
    monkeypatch.setattr(service.fpl_client, "bootstrap", lambda: _elements([1]))

    def _broken(horizon=5):
        raise RuntimeError("optimiser unavailable")

    monkeypatch.setattr(service, "draft_squad", _broken)

    squad, _, reason = service._squad_for_advice(
        1, gameweek=1, started=False, cookie=None
    )

    assert squad == []
    assert reason == "no_cookie"


def test_the_preview_never_fires_once_the_season_is_under_way(monkeypatch):
    """In-season a missing squad means a bad entry id, and inventing one would
    hide that behind advice about players they do not own."""
    monkeypatch.setattr(service.fpl_client, "bootstrap", lambda: _elements([1]))
    monkeypatch.setattr(
        service, "load_squad_state", lambda e, g: (_ for _ in ()).throw(
            service.ServiceError("entry_not_found", "gone", status=404)
        )
    )
    monkeypatch.setattr(
        service, "draft_squad", lambda horizon=5: pytest.fail("previewed in-season")
    )

    squad, _, reason = service._squad_for_advice(
        1, gameweek=7, started=True, cookie=None
    )

    assert squad == []
    assert reason == "entry_not_found"


def test_rebuild_bypasses_the_draft_cache(monkeypatch):
    """The Rebuild button's whole job.

    Without this the hour-long cache answered and the control re-rendered
    identical bytes, which reads as a broken button rather than as a squad that
    genuinely has not moved.
    """
    service._draft_cache.clear()
    calls = []

    monkeypatch.setattr(
        service.fpl_client,
        "season_state",
        lambda: {"started": False, "gameweek": 1, "gameweek_name": "GW1", "deadline": "x"},
    )

    def _fake_projections(horizon, engine):
        calls.append(1)
        raise service.ServiceError("stop", "counted", status=500)

    monkeypatch.setattr(service, "projections_for", _fake_projections)

    for _ in range(2):
        with pytest.raises(service.ServiceError):
            service.draft_squad(horizon=5, refresh=True)

    # Both calls got through to the compute rather than one being served warm.
    assert len(calls) == 2


def test_without_refresh_the_cache_still_answers(monkeypatch):
    """The cache is there for a reason — an ordinary page load must not pay for
    a full optimiser run."""
    service._draft_cache.clear()
    # The strategy is part of the key now — without it, switching strategy
    # would be answered from whichever one was requested first.
    service._draft_cache["5:xpts:max_points:"] = (9e18, {"squad": [{"id": 1}]})

    monkeypatch.setattr(
        service.fpl_client,
        "season_state",
        lambda: {"started": False, "gameweek": 1, "gameweek_name": "GW1", "deadline": "x"},
    )
    monkeypatch.setattr(
        service,
        "projections_for",
        lambda h, e: pytest.fail("recomputed despite a warm cache"),
    )

    assert service.draft_squad(horizon=5) == {"squad": [{"id": 1}]}
    service._draft_cache.clear()


def test_each_strategy_gets_its_own_cache_entry(monkeypatch):
    """Otherwise the strategy selector silently does nothing.

    Exactly the bug the Rebuild button had: a control that changes a query
    parameter the cache key ignores returns the previous answer and looks
    broken.
    """
    service._draft_cache.clear()
    service._draft_cache["5:xpts:max_points:"] = (9e18, {"squad": [{"id": 1}]})

    monkeypatch.setattr(
        service.fpl_client,
        "season_state",
        lambda: {"started": False, "gameweek": 1, "gameweek_name": "GW1", "deadline": "x"},
    )
    recomputed = {"called": False}

    def _mark(horizon, engine):
        recomputed["called"] = True
        raise service.ServiceError("stop", "far enough", status=503)

    monkeypatch.setattr(service, "projections_for", _mark)

    with pytest.raises(service.ServiceError):
        service.draft_squad(horizon=5, strategy="differential")

    assert recomputed["called"], "a different strategy was served the cached one"
    service._draft_cache.clear()


def test_the_preview_squad_does_not_silence_the_buy_shortlists(monkeypatch):
    """The preview is the optimiser's own output, so almost nothing improves it.

    Filtering the shortlists on gain against it returned four empty lists and
    hid the whole buying section behind a technicality — the gain is only a
    real measurement when the squad is really theirs.
    """
    seen = {}

    def _capture(elements, projections, swings, owned, squad, horizon, **kwargs):
        seen["squad"] = squad
        seen["bench_ids"] = kwargs.get("bench_ids")
        return []

    monkeypatch.setattr(service.advice_mod, "buy_shortlists", _capture)
    monkeypatch.setattr(
        service, "_squad_for_advice", lambda *a, **k: ([{"id": 1}], [], "preview_draft")
    )
    monkeypatch.setattr(service.fpl_client, "season_state", lambda: {"gameweek": 1})
    monkeypatch.setattr(service.fpl_client, "fixtures", lambda: [])
    monkeypatch.setattr(service.fpl_client, "history", lambda entry: {})
    monkeypatch.setattr(
        service,
        "projections_for",
        lambda *a, **k: service.Projection(
            data={"elements": [], "teams": [], "chips": []},
            projections={},
            gameweeks=[],
            engine="xpts",
            engine_requested="xpts",
        ),
    )
    monkeypatch.setattr(service.ticker_mod, "build_ticker", lambda *a, **k: {"swings": []})

    service.actions_for(1)

    assert seen["squad"] == []
    assert seen["bench_ids"] is None
