"""FCPS must keep producing the number it always produced.

These tests exist to stop a well-meaning refactor from quietly redefining a score
the user recognises. The reference values are recomputed by hand from the
original ``calculate_fcps()`` formula, not captured from this implementation's
output — a snapshot of your own bug is not a regression test.
"""

import pytest

from engine import fcps
from tests.conftest import make_element


def test_weights_are_the_original_weights():
    assert fcps.LEGACY_WEIGHTS == {
        "total_points_weight": 0.20,
        "form_weight": 0.40,
        "fdr_weight": 0.25,
        "ict_index_weight": 0.15,
    }
    assert fcps.MAX_NEXT_3_FDR == 15


def test_score_matches_the_original_formula_by_hand():
    element = {"id": 1, "total_points": 100, "form": "5.0", "ict_index": "80.0"}
    divisors = {
        "total_points": 200.0,
        "form": 10.0,
        "next_3_fdr": 15.0,
        "ict_index": 160.0,
    }
    # 0.20*0.5 + 0.40*0.5 + 0.25*(1 - 6/15) + 0.15*0.5
    expected_raw = 0.10 + 0.20 + 0.25 * 0.6 + 0.075
    result = fcps.score_player(element, fdr_sum=6, divisors=divisors)

    assert result["fcps"] == pytest.approx(round(expected_raw, 3) * 1000, abs=0.05)
    assert result["components"]["fdr_term"] == pytest.approx(0.6)


def test_preseason_zero_maxima_do_not_divide_by_zero():
    """Every player on 0 points and 0 form is the state the original crashed in."""
    element = {"id": 1, "total_points": 0, "form": "0.0", "ict_index": "0.0"}
    divisors = {"total_points": 0.0, "form": 0.0, "next_3_fdr": 15.0, "ict_index": 0.0}

    result = fcps.score_player(element, fdr_sum=9, divisors=divisors)

    # Only the fixture term survives: 0.25 * (1 - 9/15) = 0.10
    assert result["fcps"] == pytest.approx(100.0, abs=0.5)


def test_fdr_term_is_clamped_when_a_double_pushes_the_sum_past_fifteen():
    """A run of hard doubles used to drive the term negative and the score down.

    The original had no clamp, so a team facing four difficulty-5 fixtures in
    three gameweeks scored *worse* than the arithmetic floor allowed.
    """
    element = {"id": 1, "total_points": 0, "form": "0", "ict_index": "0"}
    divisors = {"total_points": 1.0, "form": 1.0, "next_3_fdr": 15.0, "ict_index": 1.0}

    result = fcps.score_player(element, fdr_sum=25, divisors=divisors)

    assert result["components"]["fdr_term"] == 0.0
    assert result["fcps"] == pytest.approx(0.0, abs=0.5)


def test_next_n_fdr_counts_the_fixtures_it_actually_found(fixtures):
    """`fixtures_counted` is the honesty term the original lacked.

    With fewer than three fixtures scheduled the FDR sum is still divided by
    fifteen, so the score reads as an easy run. The number is unchanged; the
    count lets the caller say so.
    """
    team_fixtures = fcps.build_team_fixtures(fixtures, from_gameweek=19)

    fdr_sum, counted = fcps.next_n_fdr(1, team_fixtures)

    assert counted == 2  # only GW19 and GW20 remain in the fixture set
    assert fdr_sum == sum(e["difficulty"] for e in team_fixtures[1][:2])


def test_score_all_covers_every_outfield_player_and_no_managers(elements, fixtures):
    elements = list(elements) + [make_element(9001, 5, team=1)]

    scored = fcps.score_all(elements, fixtures, from_gameweek=13)

    assert 9001 not in scored
    assert len(scored) == len(elements) - 1
    assert all(entry["fcps"] >= 0 for entry in scored.values())


def test_top_by_position_returns_the_prompt_shortlist_in_fcps_order(elements, fixtures):
    scored = fcps.score_all(elements, fixtures, from_gameweek=13)

    shortlist = fcps.top_by_position(scored, elements)

    positions = [
        {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}[e["element_type"]] for e in shortlist
    ]
    assert positions.count("GKP") == 5
    assert positions.count("DEF") == 15
    assert positions.count("MID") == 25
    assert positions.count("FWD") == 25

    keepers = [scored[int(e["id"])]["fcps"] for e in shortlist[:5]]
    assert keepers == sorted(keepers, reverse=True)


def test_unavailable_players_are_excluded_from_the_shortlist(elements, fixtures):
    injured = make_element(9002, 3, team=1, status="i", total_points=300)
    elements = list(elements) + [injured]
    scored = fcps.score_all(elements, fixtures, from_gameweek=13)

    shortlist = fcps.top_by_position(scored, elements)

    assert 9002 not in {int(e["id"]) for e in shortlist}


def test_scoring_is_deterministic(elements, fixtures):
    first = fcps.score_all(elements, fixtures, 13)
    second = fcps.score_all(elements, fixtures, 13)
    assert first == second
