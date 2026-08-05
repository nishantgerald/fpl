"""Mini-league analysis.

The advice inverts on which side of the gap you sit, so the property that
matters most is that posture is right — telling someone chasing to cover the
template guarantees they finish exactly where they started.
"""

import pytest

from engine import leagues


def _row(entry, rank, total, name="M"):
    return {"entry": entry, "rank": rank, "total": total, "entry_name": name}


def _pick(element, multiplier=1):
    return {"element": element, "multiplier": multiplier}


# ---------------------------------------------------------------- discovery


def test_auto_enrolled_leagues_are_not_treated_as_rivalries():
    """FPL enrols everyone into country and club leagues with millions of
    members. Ranking advice about those is noise dressed as insight."""
    entry = {
        "leagues": {
            "classic": [
                {"id": 314, "name": "Overall", "league_type": "s"},
                {"id": 249, "name": "USA", "league_type": "s"},
                {"id": 431163, "name": "NBC Sports League", "league_type": "x"},
            ]
        }
    }

    classified = leagues.classify_leagues(entry)
    meaningful = [le for le in classified if le["meaningful"]]

    assert [le["id"] for le in meaningful] == [431163]


# ---------------------------------------------------------------- rivals


def test_only_the_managers_within_reach_are_analysed():
    """Someone forty places clear is not a rival, and fetching their squad
    costs an upstream request that buys nothing."""
    standings = [_row(i, i, 500 - i) for i in range(1, 41)]

    around = leagues.rivals_around(standings, entry_id=20, window=3)

    assert int(around["me"]["entry"]) == 20
    assert [r["entry"] for r in around["above"]] == [17, 18, 19]
    assert [r["entry"] for r in around["below"]] == [21, 22, 23]


def test_a_manager_outside_the_standings_yields_no_rivals():
    around = leagues.rivals_around([_row(1, 1, 100)], entry_id=999)
    assert around["me"] is None
    assert around["above"] == [] and around["below"] == []


# ------------------------------------------------- effective ownership


def test_captaincy_counts_twice_because_that_is_where_the_swing_is():
    """A player captained by a quarter of the league is far more dangerous to
    be without than raw ownership suggests."""
    squads = {
        1: [_pick(10, multiplier=2)],
        2: [_pick(10, multiplier=1)],
        3: [_pick(10, multiplier=0)],  # benched
        4: [_pick(99, multiplier=1)],
    }
    elements = {10: {"web_name": "Haaland"}, 99: {"web_name": "Other"}}

    ownership = leagues.effective_ownership(squads, elements)

    haaland = ownership[10]
    assert haaland["ownership"] == 75.0          # three of four squads
    assert haaland["started"] == 2               # one is benched
    assert haaland["captained"] == 1
    # (started + captained) / total = (2 + 1) / 4
    assert haaland["effective_ownership"] == 75.0


def test_a_benched_player_is_owned_but_not_exposure():
    squads = {1: [_pick(10, multiplier=0)], 2: [_pick(10, multiplier=0)]}
    ownership = leagues.effective_ownership(squads, {10: {"web_name": "X"}})

    assert ownership[10]["ownership"] == 100.0
    assert ownership[10]["effective_ownership"] == 0.0


# ---------------------------------------------------------------- gaps


def test_gaps_are_what_the_league_owns_and_you_do_not():
    ownership = {
        10: {"ownership": 90.0, "effective_ownership": 110.0},
        20: {"ownership": 20.0, "effective_ownership": 20.0},
        30: {"ownership": 50.0, "effective_ownership": 50.0},
    }
    elements = {i: {"web_name": f"P{i}", "now_cost": 60} for i in (10, 20, 30)}
    projections = {i: {"horizon_xpts": 20.0} for i in (10, 20, 30)}

    split = leagues.differentials([_pick(30)], ownership, projections, elements)

    # Ranked by exposure: the most-captained player you lack comes first.
    assert [g["id"] for g in split["gaps"]] == [10, 20]
    # And your own most-differential holding is the edge.
    assert split["edges"][0]["id"] == 30


# ---------------------------------------------------------------- posture


def test_the_leader_is_told_to_cover_not_to_gamble():
    me = _row(1, 1, 500)
    stance = leagues.posture(me, {"above": [], "below": [_row(2, 2, 480)]})

    assert stance["stance"] == "protect"
    assert "cover" in stance["advice"].lower()


def test_someone_well_behind_is_told_to_differentiate():
    """Matching the template preserves a gap rather than closing it."""
    me = _row(9, 9, 400)
    stance = leagues.posture(
        me, {"above": [_row(1, 1, 520)], "below": [_row(10, 10, 380)]}
    )

    assert stance["stance"] == "chase"
    assert "do not own" in stance["advice"] or "differ" in stance["advice"].lower()
    assert stance["gap_above"] == -120


def test_no_standings_yet_is_stated_rather_than_guessed():
    stance = leagues.posture(None, {"above": [], "below": []})
    assert stance["stance"] == "unknown"
