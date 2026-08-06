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
