"""The expected-points model.

The properties that matter most are the ones FCPS got wrong: blanks must not
score, minutes must matter, and the decomposition must actually sum to the total
rather than being a plausible story printed next to it.
"""

import pytest

from engine import xpts
from tests.conftest import make_element, make_team


def _project(element, fixtures, teams, events, gameweeks=3, from_gw=13):
    result = xpts.project_all([element], fixtures, teams, events, from_gw, gameweeks)
    return result[element["id"]]


def test_components_sum_to_the_total(elements, fixtures, teams, events):
    projections = xpts.project_all(elements, fixtures, teams, events, 13, 5)
    for projection in projections.values():
        for entry in projection["per_gameweek"]:
            assert sum(entry["components"].values()) == pytest.approx(
                entry["xpts"], abs=0.01
            ), "the decomposition is the model, not a narrative"


def test_horizon_is_the_sum_of_the_gameweeks(elements, fixtures, teams, events):
    projections = xpts.project_all(elements, fixtures, teams, events, 13, 5)
    for projection in projections.values():
        assert projection["horizon_xpts"] == pytest.approx(
            sum(g["xpts"] for g in projection["per_gameweek"]), abs=0.01
        )


def test_injured_player_projects_zero_this_gameweek(fixtures, teams, events):
    injured = make_element(1, 3, 1, status="i")
    projection = _project(injured, fixtures, teams, events)
    assert projection["per_gameweek"][0]["xpts"] == 0.0
    assert projection["availability"] == 0.0


def test_suspended_player_projects_zero(fixtures, teams, events):
    projection = _project(make_element(1, 3, 1, status="s"), fixtures, teams, events)
    assert projection["horizon_xpts"] == 0.0


def test_doubtful_player_is_scaled_by_published_chance(fixtures, teams, events):
    healthy = _project(make_element(1, 3, 1), fixtures, teams, events)
    doubtful = _project(
        make_element(2, 3, 1, status="d", chance=25), fixtures, teams, events
    )
    assert doubtful["per_gameweek"][0]["xpts"] < healthy["per_gameweek"][0]["xpts"]
    assert doubtful["availability"] == pytest.approx(0.25)


def test_blank_gameweek_scores_zero_and_lists_no_fixtures(teams, events):
    """The FCPS bug in one test: a blank must never look like an easy fixture."""
    # Team 1 plays in GW13 only; GW14 and GW15 are blanks for them.
    fixtures = [
        {
            "id": 1, "event": 13, "team_h": 1, "team_a": 2,
            "team_h_difficulty": 2, "team_a_difficulty": 3, "finished": False,
        }
    ]
    projection = _project(make_element(1, 3, 1), fixtures, teams, events)
    gw13, gw14, gw15 = projection["per_gameweek"]

    assert gw13["xpts"] > 0
    assert gw14["xpts"] == 0.0
    assert gw14["fixtures"] == []
    assert gw15["xpts"] == 0.0
    assert all(v == 0.0 for v in gw14["components"].values())


def test_double_gameweek_scores_more_than_a_single(teams, events):
    single = [
        {"id": 1, "event": 13, "team_h": 1, "team_a": 2,
         "team_h_difficulty": 3, "team_a_difficulty": 3, "finished": False}
    ]
    double = single + [
        {"id": 2, "event": 13, "team_h": 3, "team_a": 1,
         "team_h_difficulty": 3, "team_a_difficulty": 3, "finished": False}
    ]

    one = _project(make_element(1, 3, 1), single, teams, events, gameweeks=1)
    two = _project(make_element(1, 3, 1), double, teams, events, gameweeks=1)

    assert len(two["per_gameweek"][0]["fixtures"]) == 2
    assert two["per_gameweek"][0]["xpts"] > one["per_gameweek"][0]["xpts"]


def test_minutes_matter(fixtures, teams, events):
    """A 25-minute cameo must not outrank an identical-rate 90-minute starter."""
    nailed = make_element(1, 3, 1, starts=12, minutes=1080)
    cameo = make_element(2, 3, 1, starts=1, minutes=300)
    assert (
        _project(cameo, fixtures, teams, events)["horizon_xpts"]
        < _project(nailed, fixtures, teams, events)["horizon_xpts"]
    )


def test_minutes_risk_bands(fixtures, teams, events):
    nailed = _project(make_element(1, 3, 1, starts=12, minutes=1080), fixtures, teams, events)
    rotated = _project(make_element(2, 3, 1, starts=3, minutes=300), fixtures, teams, events)
    assert nailed["minutes_risk"] == "low"
    assert rotated["minutes_risk"] == "high"


def test_goalkeepers_earn_save_points_and_outfielders_do_not(fixtures, teams, events):
    keeper = _project(make_element(1, 1, 1), fixtures, teams, events)
    midfielder = _project(make_element(2, 3, 1), fixtures, teams, events)
    assert keeper["per_gameweek"][0]["components"]["saves"] > 0
    assert midfielder["per_gameweek"][0]["components"]["saves"] == 0


def test_defenders_earn_more_clean_sheet_value_than_midfielders(fixtures, teams, events):
    defender = _project(make_element(1, 2, 1), fixtures, teams, events)
    midfielder = _project(make_element(2, 3, 1), fixtures, teams, events)
    assert (
        defender["per_gameweek"][0]["components"]["clean_sheet"]
        > midfielder["per_gameweek"][0]["components"]["clean_sheet"]
    )


def test_only_keepers_and_defenders_are_penalised_for_conceding(fixtures, teams, events):
    for element_type in (1, 2):
        projection = _project(make_element(1, element_type, 1), fixtures, teams, events)
        assert projection["per_gameweek"][0]["components"]["conceded"] < 0
    for element_type in (3, 4):
        projection = _project(make_element(1, element_type, 1), fixtures, teams, events)
        assert projection["per_gameweek"][0]["components"]["conceded"] == 0


def test_better_underlying_numbers_project_higher(fixtures, teams, events):
    weak = _project(make_element(1, 4, 1, xg90=0.1), fixtures, teams, events)
    strong = _project(make_element(2, 4, 1, xg90=0.9), fixtures, teams, events)
    assert strong["horizon_xpts"] > weak["horizon_xpts"]


def test_easier_fixtures_project_higher(events):
    teams = [make_team(1), make_team(2), make_team(20)]
    # Team 2 is weak defensively; team 20 is strong.
    teams[1].update(strength_defence_home=800, strength_defence_away=800)
    teams[2].update(strength_defence_home=1400, strength_defence_away=1400)

    def one_fixture(opponent):
        return [
            {"id": 1, "event": 13, "team_h": 1, "team_a": opponent,
             "team_h_difficulty": 2, "team_a_difficulty": 4, "finished": False}
        ]

    easy = _project(make_element(1, 4, 1), one_fixture(2), teams, events, gameweeks=1)
    hard = _project(make_element(1, 4, 1), one_fixture(20), teams, events, gameweeks=1)
    assert easy["per_gameweek"][0]["xpts"] > hard["per_gameweek"][0]["xpts"]


def test_finished_fixtures_are_excluded(teams, events):
    fixtures = [
        {"id": 1, "event": 13, "team_h": 1, "team_a": 2,
         "team_h_difficulty": 2, "team_a_difficulty": 3, "finished": True}
    ]
    projection = _project(make_element(1, 3, 1), fixtures, teams, events, gameweeks=1)
    assert projection["per_gameweek"][0]["fixtures"] == []
    assert projection["per_gameweek"][0]["xpts"] == 0.0


def test_xpts_per_million_reflects_price(fixtures, teams, events):
    cheap = _project(make_element(1, 3, 1, now_cost=45), fixtures, teams, events)
    dear = _project(make_element(2, 3, 1, now_cost=130), fixtures, teams, events)
    assert cheap["xpts_per_million"] > dear["xpts_per_million"]


def test_projection_is_deterministic(elements, fixtures, teams, events):
    first = xpts.project_all(elements, fixtures, teams, events, 13, 5)
    for _ in range(3):
        assert xpts.project_all(elements, fixtures, teams, events, 13, 5) == first


def test_missing_fields_degrade_rather_than_crash(fixtures, teams, events):
    """A field renamed upstream must zero one component, not 500 the endpoint."""
    sparse = {"id": 1, "element_type": 3, "team": 1, "now_cost": 50}
    projection = _project(sparse, fixtures, teams, events)
    assert projection["horizon_xpts"] >= 0.0
    assert len(projection["per_gameweek"]) == 3


def test_availability_recovers_over_the_horizon(fixtures, teams, events):
    doubtful = make_element(1, 3, 1, status="d", chance=25)
    projection = _project(doubtful, fixtures, teams, events, gameweeks=5)
    per_gw = [g["xpts"] for g in projection["per_gameweek"]]
    assert per_gw[0] < per_gw[-1], "a knock now says little about four weeks out"


def test_unavailable_players_never_recover(fixtures, teams, events):
    projection = _project(make_element(1, 3, 1, status="i"), fixtures, teams, events, 5)
    assert all(g["xpts"] == 0.0 for g in projection["per_gameweek"])
