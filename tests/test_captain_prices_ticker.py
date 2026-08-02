"""Captain picks, price predictions and the fixture ticker."""

import pytest

from engine import captain, prices, ticker, xpts
from tests.conftest import make_element, make_team


# ------------------------------------------------------------------ captain


@pytest.fixture
def captain_setup(squad, fixtures, teams, events):
    projections = xpts.project_all(squad, fixtures, teams, events, 13, 1)
    team_index = {int(t["id"]): t for t in teams}
    return squad, projections, team_index


def _picks(squad, captain_id, vice_id):
    return [
        {
            "element": int(p["id"]),
            "is_captain": int(p["id"]) == captain_id,
            "is_vice_captain": int(p["id"]) == vice_id,
            "multiplier": 1,
        }
        for p in squad
    ]


def test_every_squad_member_is_ranked(captain_setup):
    squad, projections, team_index = captain_setup
    result = captain.rank_captains(
        squad, _picks(squad, squad[0]["id"], squad[1]["id"]),
        projections, team_index, 13,
    )
    assert len(result["picks"]) == len(squad)
    assert [p["rank"] for p in result["picks"]] == list(range(1, len(squad) + 1))


def test_current_captain_is_read_from_the_picks(captain_setup):
    squad, projections, team_index = captain_setup
    target = int(squad[4]["id"])
    result = captain.rank_captains(
        squad, _picks(squad, target, int(squad[5]["id"])), projections, team_index, 13
    )
    assert result["current_captain"]["id"] == target
    assert result["current_vice"]["id"] == int(squad[5]["id"])


def test_flagged_captain_produces_a_critical_warning(squad, fixtures, teams, events):
    squad = [dict(p) for p in squad]
    squad[3]["status"] = "i"
    projections = xpts.project_all(squad, fixtures, teams, events, 13, 1)
    team_index = {int(t["id"]): t for t in teams}

    result = captain.rank_captains(
        squad, _picks(squad, int(squad[3]["id"]), int(squad[4]["id"])),
        projections, team_index, 13,
    )
    assert any(w["severity"] == "critical" for w in result["warnings"])
    # And an unavailable player must never be the top recommendation.
    assert result["picks"][0]["id"] != int(squad[3]["id"])


def test_blanking_player_never_ranks_first(squad, teams, events):
    """Only team 1 has a fixture, so everyone else blanks."""
    fixtures = [
        {"id": 1, "event": 13, "team_h": 1, "team_a": 2,
         "team_h_difficulty": 2, "team_a_difficulty": 3, "finished": False}
    ]
    projections = xpts.project_all(squad, fixtures, teams, events, 13, 1)
    team_index = {int(t["id"]): t for t in teams}
    result = captain.rank_captains(
        squad, _picks(squad, int(squad[0]["id"]), int(squad[1]["id"])),
        projections, team_index, 13,
    )
    assert result["picks"][0]["blanking"] is False
    blanking_ranks = [p["rank"] for p in result["picks"] if p["blanking"]]
    playing_ranks = [p["rank"] for p in result["picks"] if not p["blanking"]]
    assert min(blanking_ranks) > max(playing_ranks)


def test_ceiling_tilt_favours_attackers_over_equal_xpts_defenders(captain_setup):
    squad, projections, team_index = captain_setup
    result = captain.rank_captains(
        squad, _picks(squad, int(squad[0]["id"]), int(squad[1]["id"])),
        projections, team_index, 13,
    )
    for entry in result["picks"]:
        assert entry["captain_score"] >= entry["xpts"]


def test_captain_ranking_is_deterministic(captain_setup):
    squad, projections, team_index = captain_setup
    picks = _picks(squad, int(squad[0]["id"]), int(squad[1]["id"]))
    first = captain.rank_captains(squad, picks, projections, team_index, 13)
    for _ in range(3):
        assert captain.rank_captains(squad, picks, projections, team_index, 13) == first


# ------------------------------------------------------------------ prices


def test_heavy_net_transfers_in_predict_a_rise():
    element = make_element(1, 3, 1, transfers_in=200_000, transfers_out=1_000)
    prediction = prices.predict_player(element, total_players=10_000_000)
    assert prediction["direction"] == "rise"
    assert prediction["verdict"] in ("rising_tonight", "rise_likely")


def test_heavy_net_transfers_out_predict_a_fall():
    element = make_element(1, 3, 1, transfers_in=500, transfers_out=200_000)
    prediction = prices.predict_player(element, total_players=10_000_000)
    assert prediction["direction"] == "fall"
    assert prediction["verdict"] in ("falling_tonight", "fall_likely")


def test_quiet_player_is_stable():
    element = make_element(1, 3, 1, transfers_in=100, transfers_out=90)
    assert prices.predict_player(element, 10_000_000)["verdict"] == "stable"


def test_a_player_who_already_moved_today_is_excluded():
    element = make_element(1, 3, 1, transfers_in=500_000, transfers_out=0)
    element["cost_change_event"] = 1
    prediction = prices.predict_player(element, 10_000_000)
    assert prediction["verdict"] == "already_changed"


def test_high_ownership_needs_more_net_transfers_to_move():
    low = make_element(1, 3, 1, transfers_in=100_000, transfers_out=0, ownership="1.0")
    high = make_element(2, 3, 1, transfers_in=100_000, transfers_out=0, ownership="50.0")
    assert (
        prices.predict_player(low, 10_000_000)["pressure"]
        > prices.predict_player(high, 10_000_000)["pressure"]
    )


def test_squad_view_reports_value_delta_and_implications(elements):
    boosted = [dict(e) for e in elements[:15]]
    boosted[0]["transfers_in_event"] = 500_000
    boosted[1]["transfers_out_event"] = 500_000

    result = prices.predict_all(
        boosted, 10_000_000, squad_ids=[p["id"] for p in boosted]
    )
    assert "your_squad" in result
    assert len(result["your_squad"]) == 15
    assert isinstance(result["squad_value_delta_tonight"], int)
    assert all(entry["implication"] for entry in result["your_squad"])


def test_every_response_admits_it_is_a_heuristic(elements):
    result = prices.predict_all(elements, 10_000_000)
    assert result["model"] == "heuristic"
    assert "unpublished" in result["accuracy_note"]


def test_price_prediction_is_deterministic(elements):
    first = prices.predict_all(elements, 10_000_000, squad_ids=[1, 2, 3])
    for _ in range(3):
        assert prices.predict_all(elements, 10_000_000, squad_ids=[1, 2, 3]) == first


# ------------------------------------------------------------------ ticker


def test_blank_cell_is_structurally_empty_not_an_easy_fixture(teams):
    fixtures = [
        {"id": 1, "event": 13, "team_h": 1, "team_a": 2,
         "team_h_difficulty": 2, "team_a_difficulty": 3, "finished": False}
    ]
    grid = ticker.build_ticker(fixtures, teams, 13, 3)
    row = next(r for r in grid["teams"] if r["id"] == 1)
    assert row["cells"][0]["fixtures"]
    assert row["cells"][1]["fixtures"] == []
    assert row["blanks"] == 2
    assert row["total_fixtures"] == 1


def test_average_fdr_divides_by_fixtures_played_not_gameweeks(teams):
    fixtures = [
        {"id": 1, "event": 13, "team_h": 1, "team_a": 2,
         "team_h_difficulty": 4, "team_a_difficulty": 2, "finished": False}
    ]
    grid = ticker.build_ticker(fixtures, teams, 13, 4)
    row = next(r for r in grid["teams"] if r["id"] == 1)
    assert row["avg_fdr"] == 4.0, "three blanks must not dilute the average"


def test_double_gameweek_is_counted_twice(teams):
    fixtures = [
        {"id": 1, "event": 13, "team_h": 1, "team_a": 2,
         "team_h_difficulty": 2, "team_a_difficulty": 3, "finished": False},
        {"id": 2, "event": 13, "team_h": 3, "team_a": 1,
         "team_h_difficulty": 3, "team_a_difficulty": 2, "finished": False},
    ]
    grid = ticker.build_ticker(fixtures, teams, 13, 1)
    row = next(r for r in grid["teams"] if r["id"] == 1)
    assert len(row["cells"][0]["fixtures"]) == 2
    assert row["doubles"] == 1
    assert row["total_fixtures"] == 2


def test_swing_is_detected_and_named():
    teams = [make_team(1), make_team(2)]
    hard = [
        {"id": i, "event": 13 + i, "team_h": 1, "team_a": 2,
         "team_h_difficulty": 5, "team_a_difficulty": 2, "finished": False}
        for i in range(3)
    ]
    easy = [
        {"id": 10 + i, "event": 16 + i, "team_h": 1, "team_a": 2,
         "team_h_difficulty": 2, "team_a_difficulty": 5, "finished": False}
        for i in range(3)
    ]
    grid = ticker.build_ticker(hard + easy, teams, 13, 6)
    swings = [s for s in grid["swings"] if s["team_id"] == 1]
    assert swings, "a 5.0 -> 2.0 change must register as a swing"
    assert swings[0]["direction"] == "improving"
    assert swings[0]["from_gameweek"] == 16
    assert "easier" in swings[0]["message"]


def test_no_swing_when_the_run_is_flat(teams, fixtures):
    grid = ticker.build_ticker(fixtures, teams, 13, 6)
    for swing in grid["swings"]:
        assert abs(swing["near_avg_fdr"] - swing["far_avg_fdr"]) >= ticker.SWING_THRESHOLD
