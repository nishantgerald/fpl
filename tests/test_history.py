"""Points that already happened, attributed.

FPL publishes the stats for a gameweek and the total, and nothing in between —
so "2 points" against a striker who played ninety minutes could be a quiet game
or a goal cancelled out by two yellows, and the app could not say which.
"""

import pytest

from engine import history


def _row(**overrides):
    row = {
        "round": 1,
        "minutes": 90,
        "total_points": 0,
        "goals_scored": 0,
        "assists": 0,
        "clean_sheets": 0,
        "goals_conceded": 0,
        "saves": 0,
        "bonus": 0,
        "yellow_cards": 0,
        "red_cards": 0,
        "own_goals": 0,
        "penalties_missed": 0,
        "penalties_saved": 0,
        "defensive_contribution": 0,
        "opponent_team": 2,
        "was_home": True,
    }
    row.update(overrides)
    return row


def test_a_defenders_full_afternoon_reconciles():
    """The real thing: De Cuyper's GW1 — goal, assist, clean sheet, two bonus."""
    got = history.breakdown(
        _row(
            total_points=17,
            goals_scored=1,
            assists=1,
            clean_sheets=1,
            bonus=2,
        ),
        "DEF",
    )

    assert got["parts"] == {
        "appearance": 2,
        "goals": 6,
        "assists": 3,
        "clean_sheet": 4,
        "bonus": 2,
    }
    assert got["unexplained"] == 0


def test_a_goal_is_worth_what_the_position_says():
    assert history.breakdown(_row(goals_scored=1), "FWD")["parts"]["goals"] == 4
    assert history.breakdown(_row(goals_scored=1), "MID")["parts"]["goals"] == 5
    assert history.breakdown(_row(goals_scored=1), "DEF")["parts"]["goals"] == 6


def test_a_substitute_gets_one_point_not_two():
    assert history.breakdown(_row(minutes=20), "MID")["parts"]["appearance"] == 1
    assert history.breakdown(_row(minutes=60), "MID")["parts"]["appearance"] == 2


def test_a_player_who_did_not_feature_earns_nothing():
    """Not even an appearance point, which is the difference between a blank
    and a benching."""
    assert history.breakdown(_row(minutes=0), "MID")["parts"] == {}


def test_a_clean_sheet_needs_the_hour():
    """FPL awards nothing to a substitute who came on at 80 with the sheet
    intact, and a breakdown that granted it would disagree with the total."""
    late = history.breakdown(_row(minutes=20, clean_sheets=1), "DEF")

    assert "clean_sheet" not in late["parts"]


def test_a_midfielders_clean_sheet_is_worth_one_and_a_forwards_nothing():
    assert history.breakdown(_row(clean_sheets=1), "MID")["parts"]["clean_sheet"] == 1
    assert "clean_sheet" not in history.breakdown(_row(clean_sheets=1), "FWD")["parts"]


def test_goals_conceded_cost_a_point_per_two_and_only_at_the_back():
    assert history.breakdown(_row(goals_conceded=3), "DEF")["parts"]["conceded"] == -1
    assert history.breakdown(_row(goals_conceded=4), "GKP")["parts"]["conceded"] == -2
    assert "conceded" not in history.breakdown(_row(goals_conceded=4), "MID")["parts"]


def test_saves_are_a_point_per_three_for_a_goalkeeper():
    assert history.breakdown(_row(saves=6), "GKP")["parts"]["saves"] == 2
    assert "saves" not in history.breakdown(_row(saves=6), "DEF")["parts"]


def test_cards_and_misses_come_off():
    got = history.breakdown(
        _row(yellow_cards=1, own_goals=1, penalties_missed=1), "MID"
    )

    assert got["parts"]["yellow_card"] == -1
    assert got["parts"]["own_goals"] == -2
    assert got["parts"]["penalties_missed"] == -2


def test_a_gap_against_fpls_total_is_reported_not_hidden():
    """The honest part. FPL's scoring has corners this does not model, and a
    breakdown that quietly rounded itself to match would hide exactly the cases
    worth knowing about."""
    got = history.breakdown(_row(total_points=99), "MID")

    assert got["attributed"] == 2
    assert got["unexplained"] == 97


def test_the_most_recent_gameweek_comes_first():
    """Ordered by gameweek rather than list position: a double puts two rows on
    one number and a postponement can land one out of sequence."""
    rows = history.recent(
        [_row(round=3, total_points=3), _row(round=1), _row(round=2)], "MID", 5
    )

    assert [r["gameweek"] for r in rows] == [3, 2, 1]


def test_the_limit_is_applied_after_ordering():
    rows = history.recent([_row(round=n) for n in range(1, 10)], "MID", 3)

    assert [r["gameweek"] for r in rows] == [9, 8, 7]


def test_an_empty_history_is_an_empty_list_not_an_error():
    assert history.recent([], "MID", 5) == []
