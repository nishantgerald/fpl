"""The two invariants the ML package stands on.

**No lookahead.** A feature for gameweek *g* must not contain gameweek *g*.
Every metric in ``PRDs/ml-methodology.md`` is worthless if this slips, and it
slips silently — a forgotten ``.shift(1)`` makes the backtest look *better*, so
nothing alerts.

**Feature parity.** The historical builder and the live builder must produce the
same numbers for the same underlying facts. Without this the model is trained on
one quantity and served another, and again nothing alerts: the endpoint returns
plausible numbers that mean nothing.

Both are checked here against hand-computed values, on a panel small enough to
verify by inspection.

Skipped when pandas/numpy aren't installed — the web process doesn't need them
(see ``requirements-ml.txt``), so a bare install must not fail the suite.
"""

import pytest

pytest.importorskip("pandas")
pytest.importorskip("numpy")

import pandas as pd  # noqa: E402

from ml import baselines, features  # noqa: E402

SEASON = "2024-25"
ELEMENT = 42

# Six gameweeks for one midfielder. Chosen so every accumulator has a distinct,
# hand-checkable value.
POINTS = [2, 6, 1, 9, 5, 12]
MINUTES = [90, 90, 20, 90, 75, 90]
GOALS = [0, 1, 0, 2, 0, 3]


def make_panel():
    rows = []
    for index, gameweek in enumerate(range(1, 7)):
        rows.append(
            {
                "season": SEASON,
                "element": ELEMENT,
                "GW": gameweek,
                "position": "MID",
                "team_id": 7,
                "total_points": POINTS[index],
                "minutes": MINUTES[index],
                "is_start": 1.0 if MINUTES[index] >= 60 else 0.0,
                "goals_scored": GOALS[index],
                "assists": 1,
                "clean_sheets": 0,
                "goals_conceded": 1,
                "saves": 0,
                "bps": 20,
                "bonus": 1,
                "ict_index": 10.0,
                "influence": 20.0,
                "creativity": 30.0,
                "threat": 40.0,
                "expected_goals": 0.5,
                "expected_assists": 0.25,
                "expected_goals_conceded": 1.1,
                "expected_goal_involvements": 0.75,
                "xP": 4.5,
                "value": 75,
                "n_fixtures": 1,
                "fdr": 3.0,
                "home_share": 1.0,
                "has_xstats": 1.0,
                "next3_fdr": 9.0,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def panel():
    return make_panel()


@pytest.fixture
def frame(panel):
    return features.build_frame(panel)


def row_for(frame, gameweek):
    match = frame[frame["GW"] == gameweek]
    assert not match.empty, f"no feature row for GW{gameweek}"
    return match.iloc[0]


# ---------------------------------------------------------------- no lookahead


def test_target_gameweek_is_excluded_from_every_accumulator(frame):
    row = row_for(frame, 6)

    # Five prior gameweeks: GW1-5.
    assert row["team_games_played"] == 5
    # Minutes GW1-5 = 365, not 455.
    assert row["minutes_per_team_game"] == pytest.approx(365 / 5)
    # Points GW1-5 = 23 over 5 appearances.
    assert row["points_per_game"] == pytest.approx(23 / 5)
    # Goals GW1-5 = 3, over 365 minutes.
    assert row["goals_per_90"] == pytest.approx(3 * 90 / 365)


def test_form_is_the_mean_of_the_previous_four_gameweeks_only(frame):
    # GW2-5 = 6, 1, 9, 5.
    assert row_for(frame, 6)["form"] == pytest.approx((6 + 1 + 9 + 5) / 4)
    # GW1-4 = 2, 6, 1, 9.
    assert row_for(frame, 5)["form"] == pytest.approx((2 + 6 + 1 + 9) / 4)


def test_the_label_is_the_target_gameweeks_own_points(frame):
    assert row_for(frame, 6)[features.TARGET] == 12
    assert row_for(frame, 6)["gw_minutes"] == 90


def test_assert_no_lookahead_passes_on_a_correct_frame(panel, frame):
    features.assert_no_lookahead(panel, frame, SEASON, ELEMENT, 6)


def test_assert_no_lookahead_catches_an_unshifted_column(panel, frame):
    """The guard has to actually fail when the invariant is broken."""
    broken = frame.copy()
    broken.loc[broken["GW"] == 6, "minutes_per_team_game"] = 999.0

    with pytest.raises(AssertionError, match="minutes_per_team_game"):
        features.assert_no_lookahead(panel, broken, SEASON, ELEMENT, 6)


def test_rows_before_the_minimum_gameweek_are_dropped(frame):
    from ml import config

    assert frame["GW"].min() >= config.MIN_GAMEWEEK


# ---------------------------------------------------------------- parity


def bootstrap_equivalent():
    """A live element whose season totals equal the panel's GW1-5 accumulation."""
    prior = slice(0, 5)
    minutes = sum(MINUTES[prior])
    points = sum(POINTS[prior])
    return {
        "id": ELEMENT,
        "element_type": 3,
        "team": 7,
        "now_cost": 75,
        "minutes": minutes,
        "total_points": points,
        "starts": sum(1 for m in MINUTES[prior] if m >= 60),
        "points_per_game": points / 5,
        "form": (6 + 1 + 9 + 5) / 4,
        "goals_scored": sum(GOALS[prior]),
        "assists": 5,
        "clean_sheets": 0,
        "goals_conceded": 5,
        "saves": 0,
        "bps": 100,
        "bonus": 5,
        "ict_index": 50.0,
        "influence": 100.0,
        "creativity": 150.0,
        "threat": 200.0,
        "expected_goals": 2.5,
        "expected_assists": 1.25,
        "expected_goals_conceded": 5.5,
        "ep_next": 4.5,
    }


def test_live_and_historical_builders_produce_the_same_feature_names():
    live = features.from_bootstrap(
        bootstrap_equivalent(),
        team_games=5,
        fixture_context={"n_fixtures": 1, "mean_fdr": 3.0, "home_share": 1.0},
    )
    assert tuple(live) == features.FEATURES


def test_live_and_historical_builders_agree_on_every_value(frame):
    """The parity check. A drift here is invisible in production and fatal."""
    historical = row_for(frame, 6)
    live = features.from_bootstrap(
        bootstrap_equivalent(),
        team_games=5,
        fixture_context={"n_fixtures": 1, "mean_fdr": 3.0, "home_share": 1.0},
    )

    mismatched = []
    for name in features.FEATURES:
        expected = float(historical[name])
        actual = float(live[name])
        if expected != pytest.approx(actual, rel=1e-6, abs=1e-9):
            mismatched.append(f"{name}: historical={expected!r} live={actual!r}")

    assert not mismatched, "feature parity broken:\n  " + "\n  ".join(mismatched)


def test_vector_orders_features_canonically():
    live = features.from_bootstrap(
        bootstrap_equivalent(),
        team_games=5,
        fixture_context={"n_fixtures": 1, "mean_fdr": 3.0, "home_share": 1.0},
    )
    assert features.vector(live) == [float(live[n]) for n in features.FEATURES]


# ---------------------------------------------------------------- blanks


def test_a_blank_gameweek_has_no_fixtures_and_no_difficulty():
    context = features.fixture_context_for(7, gameweek=29, fixture_index={})

    assert context["n_fixtures"] == 0
    assert context["mean_fdr"] != context["mean_fdr"]  # NaN
    assert context["home_share"] != context["home_share"]


def test_a_double_gameweek_averages_its_fixtures():
    index = {
        7: {
            29: [
                {"opponent": 3, "home": True, "fdr": 2},
                {"opponent": 9, "home": False, "fdr": 4},
            ]
        }
    }

    context = features.fixture_context_for(7, 29, index)

    assert context["n_fixtures"] == 2
    assert context["mean_fdr"] == pytest.approx(3.0)
    assert context["home_share"] == pytest.approx(0.5)


# ---------------------------------------------------------------- baselines


def test_the_fcps_baseline_uses_the_same_weights_as_the_live_engine():
    """Two copies of the weights, one meaning. Drift here rigs the comparison."""
    from engine import fcps as engine_fcps

    assert baselines.FCPS_WEIGHTS == engine_fcps.LEGACY_WEIGHTS
    assert baselines.FCPS_MAX_NEXT_3_FDR == float(engine_fcps.MAX_NEXT_3_FDR)


def test_fcps_calibration_is_fitted_on_train_only(frame):
    train = frame[frame["GW"] < 6].copy()
    test = frame[frame["GW"] >= 6].copy()
    if train.empty or test.empty:
        pytest.skip("panel too small to split")

    report = baselines.add_all(train, test)

    assert report["fitted_on"] == "train split only"
    assert "bl_fcps_points" in test.columns
    assert "bl_fcps" in test.columns


def test_baseline_inputs_are_never_features():
    """`bl_` columns feed the comparators and must not leak into a model."""
    assert not set(features.BASELINE_INPUTS) & set(features.FEATURES)
    assert not set(features.LABEL_EXTRAS) & set(features.FEATURES)
    assert features.TARGET not in features.FEATURES


# ── Between-seasons inference ───────────────────────────────────────────────
#
# Both regressions below only appear when `team_games` is 0 and element totals
# describe a completed season — that is, exactly when a manager is picking their
# opening squad, and never during the mid-season window the backtest measures.


def _element(**overrides):
    element = {
        "id": 1,
        "element_type": 2,
        "team": 1,
        "now_cost": 50,
        "minutes": 2953,
        "starts": 34,
        "total_points": 239,
        "points_per_game": "6.8",
        "ict_index": "302.3",
        "influence": "1180.4",
        "creativity": "320.1",
        "threat": "1520.0",
        "goals_scored": 27,
        "assists": 8,
        "clean_sheets": 13,
        "goals_conceded": 32,
        "saves": 0,
        "bps": 952,
        "bonus": 43,
        "expected_goals": "25.50",
        "expected_assists": "2.67",
        "expected_goals_conceded": "38.6",
        "form": "0.0",
        "ep_next": "4.0",
    }
    element.update(overrides)
    return element


CONTEXT = {"n_fixtures": 1, "mean_fdr": 3.0, "home_share": 0.5}


def test_a_reset_league_table_does_not_divide_season_totals_by_one():
    """`start_rate` is a ratio bounded 0-1; it was reaching 34.

    Between seasons the API reports 0 games played while the element still
    carries 38 games of totals. Dividing by `max(1, 0)` inflated every per-game
    feature ~38x, far outside the splits the trees were fitted on.
    """
    row = features.from_bootstrap(_element(), 0, CONTEXT)

    assert 0.0 <= row["start_rate"] <= 1.0
    assert row["ict_per_game"] < 20
    assert row["minutes_per_team_game"] <= 90.0


def test_a_live_league_table_is_still_used_when_present():
    """The fallback must not override a real mid-season count."""
    row = features.from_bootstrap(_element(minutes=900, starts=10), 10, CONTEXT)

    assert row["start_rate"] == pytest.approx(1.0)
    assert row["minutes_per_team_game"] == pytest.approx(90.0)


def test_a_player_who_has_never_played_is_not_scored_above_a_proven_one():
    """The regression that put £46m signings above a 209-point defender.

    With no minutes every performance feature is 0.0, indistinguishable from a
    genuine rate of zero, so the model priced these players off `fpl_xp` alone
    and returned ~9-12 points with no evidence behind it.
    """
    predict = pytest.importorskip("ml.predict")
    predictor = predict.get_predictor()
    if predictor is None:
        pytest.skip("no trained artifact on this machine")

    unseen = _element(
        minutes=0, starts=0, total_points=0, points_per_game="0.0",
        ict_index="0.0", influence="0.0", creativity="0.0", threat="0.0",
        goals_scored=0, assists=0, clean_sheets=0, goals_conceded=0, bps=0,
        bonus=0, expected_goals="0.0", expected_assists="0.0",
        expected_goals_conceded="0.0", ep_next="2.2",
    )
    scored = predictor.score([unseen], {1: {1: [{"difficulty": 3, "home": True}]}}, [1], 0)
    projection = scored.get(1, {}).get(1, 0.0)

    # Capped at FPL's own estimate: weak, but honest, and never invented.
    assert projection <= float(unseen["ep_next"]) + 1e-6
