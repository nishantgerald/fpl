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


# ── Between-seasons projections ─────────────────────────────────────────────


def test_a_reset_league_table_still_produces_a_footballing_projection():
    """The regression: pre-season xPts was bonus-minus-cards and nothing else.

    With `team_games` at 0, `minutes_profile` returned a zero play-probability —
    zeroing appearance, goals, assists and clean sheets — while `bonus_pg`
    divided a whole season's bonus haul by `max(1.0, 0)`. A striker came out at
    34.55 expected points a gameweek, composed of 36.55 bonus and -2.0 cards.
    """
    striker = make_element(
        1, 4, 1, 155, minutes=2953, starts=34, total_points=239,
        xg90=0.78, xa90=0.08,
    )
    # A full season's bonus haul, which is what the old code divided by one.
    striker["bonus"] = 43
    index = {1: {1: [{"opponent": 2, "home": True, "fdr": 3}]}}

    projection = xpts.project_player(striker, [1], index, {}, 0)
    gameweek = projection["per_gameweek"][0]
    components = gameweek["components"]

    # Bonus is capped at 3 a match in FPL; anything near a season total is wrong.
    assert components["bonus"] < 3.0
    # And the projection must actually contain football.
    assert components["appearance"] > 0
    assert components["goals"] > 0
    assert gameweek["xpts"] < 15.0


def test_a_nailed_starter_beats_a_bit_part_player_between_seasons():
    """The ordering the zero play-probability destroyed."""
    index = {1: {1: [{"opponent": 2, "home": True, "fdr": 3}]}}
    nailed = make_element(1, 3, 1, 90, minutes=3000, starts=34, total_points=180)
    fringe = make_element(2, 3, 1, 45, minutes=200, starts=1, total_points=15)

    a = xpts.project_player(nailed, [1], index, {}, 0)
    b = xpts.project_player(fringe, [1], index, {}, 0)

    assert a["per_gameweek"][0]["xpts"] > b["per_gameweek"][0]["xpts"]


def test_a_live_league_table_is_not_overridden_by_the_fallback():
    """Mid-season the real count must win; the fallback is only for a reset."""
    assert xpts.effective_team_games(12) == 12
    assert xpts.effective_team_games(0) == xpts.COMPLETED_SEASON_GAMES
    assert xpts.effective_team_games(1) == 1


def test_zeroed_strength_ratings_fall_back_to_fdr():
    """Pre-season the bootstrap ships every strength rating as 0.

    `build_team_index` still stamped a `_league` key on those teams, so the
    FDR fallback never fired. `con = opp_att / league_att` was then 0, which the
    clamp floored to FIXTURE_MULT_MIN -- declaring all twenty teams the softest
    attack in the division. Every fixture looked identical and clean-sheet
    probabilities were wildly inflated.
    """
    blank = [make_team(i) for i in (1, 2, 20)]
    for team in blank:
        team.update(
            strength_attack_home=0, strength_attack_away=0,
            strength_defence_home=0, strength_defence_away=0,
        )
    teams = xpts.build_team_index(blank)

    easy = {"opponent": 2, "home": True, "fdr": 2}
    hard = {"opponent": 20, "home": True, "fdr": 5}

    easy_att, easy_con = xpts.fixture_multipliers(easy, teams)
    hard_att, hard_con = xpts.fixture_multipliers(hard, teams)

    # A hard fixture must suppress attacking returns and raise the goals we
    # expect to concede -- not read the same as an easy one.
    assert easy_att > hard_att
    assert easy_con < hard_con
    # And nothing may sit on the floor purely because a rating was missing.
    assert easy_con > xpts.FIXTURE_MULT_MIN


def test_live_strength_ratings_still_beat_fdr():
    """The fallback must not shadow a bootstrap that does carry ratings."""
    teams = xpts.build_team_index([make_team(1), make_team(2), make_team(20)])
    teams[2].update(strength_attack_home=800, strength_attack_away=800)
    teams[20].update(strength_attack_home=1400, strength_attack_away=1400)

    # Identical FDR, so any difference has to come from the strength ratings.
    _, weak = xpts.fixture_multipliers({"opponent": 2, "home": True, "fdr": 3}, teams)
    _, strong = xpts.fixture_multipliers({"opponent": 20, "home": True, "fdr": 3}, teams)

    assert weak < strong


# ------------------------------------------------- defensive contribution


def test_defcon_is_a_probability_not_a_ratio():
    """The regression: `mean / threshold` is not a probability.

    It claimed a defender averaging exactly 10 defensive actions reached 10 in
    every match — the true answer is about half of them — and gave a midfielder
    averaging 6 a 50% chance of reaching 12, which is very nearly impossible.
    Defenders and midfielders each carried roughly half a point a gameweek of
    invented value as a result.
    """
    # Mean exactly at the threshold: near a coin flip, nowhere near certainty.
    at_threshold = xpts.defcon_probability(10.0, 1.0, 10)
    assert 0.4 < at_threshold < 0.65

    # Half the threshold: possible, but rare.
    well_below = xpts.defcon_probability(6.0, 1.0, 12)
    assert well_below < 0.05

    # Comfortably above: likely, never certain.
    well_above = xpts.defcon_probability(16.0, 1.0, 10)
    assert 0.9 < well_above < 1.0


def test_defcon_probability_rises_with_rate_and_minutes():
    assert xpts.defcon_probability(8.0, 1.0, 10) > xpts.defcon_probability(6.0, 1.0, 10)
    assert xpts.defcon_probability(8.0, 1.0, 10) > xpts.defcon_probability(8.0, 0.5, 10)
    assert xpts.defcon_probability(0.0, 1.0, 10) == 0.0
    assert xpts.defcon_probability(8.0, 1.0, 0) == 0.0


# ------------------------------------------------------- ep_next anchoring


def test_the_ep_next_anchor_fades_instead_of_switching_off():
    """GW1 and GW2 used to come from materially different models.

    The anchor applied to the first gameweek and then vanished, so the step
    between GW1 and GW2 was an artefact of a weight hitting zero rather than
    anything about the fixtures — a ~28% jump between seasons.
    """
    weights = [xpts.ep_next_weight(0, offset) for offset in range(5)]

    assert weights == sorted(weights, reverse=True)  # monotonically fading
    assert all(w > 0 for w in weights)  # never switches off mid-horizon
    # And no single step may be a cliff.
    for earlier, later in zip(weights, weights[1:]):
        assert later > earlier * 0.3


def test_the_preseason_anchor_is_lighter_than_it_looks():
    """FPL's pre-season ep_next is compressed — 570 players share 24 values,
    and Haaland at £15.5m reads the same 4.0 as Bruno at £12.0m. Leaning on it
    pulls the whole distribution toward the middle and costs the premiums most,
    so it anchors rather than co-models."""
    assert xpts.ep_next_weight(0, 0) < 0.5


# ------------------------------------------------------------ role priors


def test_price_implies_a_role_when_minutes_cannot():
    assert xpts.role_prior_from_price(90) > xpts.role_prior_from_price(55)
    assert xpts.role_prior_from_price(55) > xpts.role_prior_from_price(40)
    # Clamped at both ends: price is a coarse signal, not a verdict.
    assert 0.0 < xpts.role_prior_from_price(40) <= 0.2
    assert 0.85 <= xpts.role_prior_from_price(155) < 1.0


def test_a_transferred_player_is_not_judged_on_last_seasons_bench_time():
    """Isak: 694 minutes across an interrupted campaign read as a bit-part
    squad player, while £9.0m read as a first-choice striker at a new club.
    Pre-season the price wins, in proportion to how thin the minutes are."""
    moved = make_element(1, 4, 1, 90, minutes=694, starts=8, total_points=41)

    preseason = xpts.minutes_profile(moved, 0)
    mid_season = xpts.minutes_profile(moved, 20)

    assert preseason["p_start"] > 0.5  # the price is believed
    assert mid_season["p_start"] < 0.5  # in-season, the minutes are believed


def test_an_established_starter_is_unaffected_by_the_price_prior():
    """A full season of evidence must not be overridden by a cheap price."""
    nailed = make_element(2, 3, 1, 45, minutes=3100, starts=35, total_points=150)

    profile = xpts.minutes_profile(nailed, 0)

    assert profile["p_start"] > 0.85


# ------------------------------------------------------- rate reliability


def test_a_cameo_does_not_produce_a_superstar():
    """The regression: FPL divides season totals by minutes played.

    A player with two minutes on record carried `expected_goals_per_90` of 3.6
    and `defensive_contribution_per_90` of 45 — arithmetic, not evidence. Taken
    literally, a £5.0m substitute projected as the third-best captain in the
    league and would have been a lock for the squad builder.
    """
    priors = {"MID": {"expected_goals_per_90": 0.15}}

    # Two minutes: the observed rate is noise, so the prior dominates.
    assert xpts.shrink_rate(3.6, minutes=2, prior=0.15) < 0.2
    # A full season: the player's own rate is what it is.
    assert xpts.shrink_rate(0.60, minutes=3000, prior=0.15) > 0.5
    # And the shrinkage is monotone in minutes.
    rising = [xpts.shrink_rate(3.6, m, 0.15) for m in (0, 100, 600, 2000, 3000)]
    assert rising == sorted(rising)
    assert priors  # documents the shape the projector receives


def test_priors_are_built_only_from_players_with_real_minutes():
    """Otherwise the small samples poison the very prior meant to correct them."""
    squad = [
        make_element(1, 3, 1, minutes=2500, starts=28),
        make_element(2, 3, 1, minutes=2400, starts=27),
        # A cameo with an absurd rate: must not reach the prior.
        make_element(3, 3, 1, minutes=2, starts=0, xg90=3.6),
    ]
    for element in squad[:2]:
        element["expected_goals_per_90"] = 0.20
    squad[2]["expected_goals_per_90"] = 3.6

    priors = xpts.position_rate_priors(squad)

    assert priors["MID"]["expected_goals_per_90"] == pytest.approx(0.20)


def test_shrinkage_leaves_an_established_player_alone(fixtures, teams, events):
    """A full season of minutes must survive the correction untouched."""
    established = make_element(1, 4, 1, minutes=2900, starts=33, xg90=0.62)
    cameo = make_element(2, 4, 1, minutes=3, starts=0, xg90=3.60)

    projections = xpts.project_all(
        [established, cameo], fixtures, teams, events, 13, 3
    )

    assert projections[1]["horizon_xpts"] > projections[2]["horizon_xpts"]


# ------------------------------------------- what "Nailed" is allowed to mean


def _element(cost, owned, minutes=0, starts=0, status="a"):
    return {
        "id": 1, "web_name": "T", "element_type": 4, "team": 1,
        "now_cost": cost, "selected_by_percent": str(owned),
        "minutes": minutes, "starts": starts, "status": status,
    }


def _baseline(median_by_band):
    return {band: median_by_band for band in xpts.OWNERSHIP_BANDS}


def test_a_price_tag_alone_no_longer_makes_a_player_nailed():
    """The bug the user caught.

    Nicolas Jackson is £6.5m with 0.4% ownership and no Premier League minutes
    — he was on loan — and the app told users he was "Nailed". The price prior
    said so on its own: (65-40)/30 = 0.83, which cleared the 0.75 threshold.
    Every player at £6.5m or above read the same way.
    """
    priced_like_a_starter = _element(cost=65, owned=0.4)

    profile = xpts.minutes_profile(
        priced_like_a_starter, team_games=0, ownership=_baseline(2.1)
    )

    assert xpts.minutes_risk(profile) != "low", (
        "a price tag with no minutes and no owners is not a nailed starter"
    )


def test_the_market_can_only_pull_down_so_far():
    """Ownership is a crowd opinion, not a team sheet. A genuine differential
    should read as uncertain, not as a declared non-player."""
    unloved = _element(cost=65, owned=0.0)

    profile = xpts.minutes_profile(unloved, team_games=0, ownership=_baseline(2.1))

    assert profile["p_start"] >= xpts.role_prior_from_price(65) * xpts.MARKET_FLOOR - 1e-9


def test_a_well_owned_player_keeps_the_full_price_prior():
    """João Pedro is £7.5m and 54% owned — the market agrees with the price, so
    the correction must not penalise him."""
    backed = _element(cost=75, owned=54.2)

    profile = xpts.minutes_profile(backed, team_games=0, ownership=_baseline(13.1))

    assert profile["p_start"] == pytest.approx(xpts.role_prior_from_price(75))


def test_ownership_is_judged_against_its_own_price_band():
    """An absolute threshold cannot tell a popular budget defender from an
    avoided premium: 1.4% is high for £4.5m and conspicuously low for £9m."""
    cheap_and_popular = _element(cost=45, owned=1.4)
    dear_and_avoided = _element(cost=95, owned=1.4)

    baseline = {band: 0.0 for band in xpts.OWNERSHIP_BANDS}
    baseline[xpts._band_for(45)] = 0.3
    baseline[xpts._band_for(95)] = 12.1

    assert xpts.market_agreement(cheap_and_popular, baseline) == 1.0
    assert xpts.market_agreement(dear_and_avoided, baseline) < 0.6


def test_an_estimate_is_not_reported_as_an_observation():
    """Before a ball is kicked there are no minutes, so the number is a price
    tag corrected by a crowd opinion. Printing that identically to thirty
    observed starts is lying by omission."""
    no_football_yet = _element(cost=65, owned=5.0)
    a_full_season = _element(cost=65, owned=5.0, minutes=2900, starts=34)

    assert xpts.minutes_basis(
        xpts.minutes_profile(no_football_yet, team_games=0)
    ) == "estimated"
    assert xpts.minutes_basis(
        xpts.minutes_profile(a_full_season, team_games=0)
    ) == "observed"


def test_real_minutes_outweigh_the_market_once_they_exist():
    """The correction is on the prior, so it fades as evidence arrives — a
    player who is actually starting should not stay damped by low ownership."""
    starting_but_unfancied = _element(cost=65, owned=0.1, minutes=2900, starts=34)

    profile = xpts.minutes_profile(
        starting_but_unfancied, team_games=0, ownership=_baseline(2.1)
    )

    assert profile["p_start"] > 0.75


def test_the_baseline_ignores_players_who_have_left_the_league():
    """Status 'u' players sit at near-zero ownership and would drag every median
    down, making the market look more sceptical than it is."""
    elements = [
        _element(cost=65, owned=5.0),
        _element(cost=65, owned=5.0),
        _element(cost=65, owned=0.0, status="u"),
    ]

    baseline = xpts.ownership_baseline(elements)

    assert baseline[xpts._band_for(65)] == 5.0
