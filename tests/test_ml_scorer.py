"""The ML seam, tested with no artifact present — which is the common case.

A deployment without a trained model must behave exactly like the one before this
pass: same numbers, same latency, no error. The interesting assertions here are
all about *degradation*, because that is the path production will spend most of
its time on.
"""

import pytest

from engine import ml_scorer, xpts


@pytest.fixture(autouse=True)
def _no_artifact(monkeypatch, tmp_path):
    """Point the artifact directory somewhere empty, whatever the machine has.

    Without this, the suite would pass on a laptop with no trained model and fail
    on one that has trained — which is the wrong way round for a test about the
    *absent* case.
    """
    try:
        from ml import config, predict
    except ImportError:
        yield  # the ml package isn't installed; there is definitionally no artifact
        return

    monkeypatch.setattr(config, "ARTIFACT_DIR", tmp_path)
    predict.reset()
    yield
    predict.reset()


def test_xpts_engine_is_untouched_by_the_ml_layer(
    elements, fixtures, teams, events
):
    direct = xpts.project_all(elements, fixtures, teams, events, 13, 5)
    through, used = ml_scorer.project_all(
        elements, fixtures, teams, events, 13, 5, engine="xpts"
    )

    assert used == "xpts"
    assert through == direct


def test_requesting_ml_without_an_artifact_falls_back_and_says_so(
    elements, fixtures, teams, events
):
    projections, used = ml_scorer.project_all(
        elements, fixtures, teams, events, 13, 5, engine="ml"
    )

    assert used == "xpts"
    assert projections == xpts.project_all(elements, fixtures, teams, events, 13, 5)


def test_an_unknown_engine_name_degrades_to_the_default(
    elements, fixtures, teams, events
):
    _projections, used = ml_scorer.project_all(
        elements, fixtures, teams, events, 13, 5, engine="wishful-thinking"
    )
    assert used == "xpts"


def test_describe_reports_unavailability_with_a_reason():
    described = ml_scorer.describe()

    assert set(described) >= {"available", "engines"}
    if not described["available"]:
        assert described["engines"] == ["xpts"]
        assert "ml.train" in described["reason"]


def test_engine_list_is_stable():
    assert ml_scorer.ENGINES == ("xpts", "ml", "blend")
    assert ml_scorer.DEFAULT_ENGINE in ml_scorer.ENGINES


def test_rescale_preserves_the_total_it_is_given():
    components = {"goals": 2.0, "assists": 1.0, "bonus": 1.0}

    rescaled = ml_scorer._rescale(components, old_total=4.0, new_total=6.0)

    assert sum(rescaled.values()) == pytest.approx(6.0)
    assert rescaled["goals"] == pytest.approx(3.0)


def test_rescale_handles_a_zero_baseline_without_dividing_by_zero():
    rescaled = ml_scorer._rescale({"a": 0.0, "b": 0.0}, old_total=0.0, new_total=3.0)
    assert sum(rescaled.values()) == pytest.approx(3.0)


def test_merge_applies_availability_and_never_learns_it(elements):
    """Injury flags aren't in the training data; they're applied afterwards."""
    injured = dict(elements[0])
    injured["status"] = "i"

    base = {
        "per_gameweek": [
            {"gameweek": 13, "xpts": 5.0, "fixtures": [{}], "components": {"goals": 5.0}}
        ],
        "horizon_xpts": 5.0,
        "xpts_next": 5.0,
    }

    merged = ml_scorer._merge(base, {13: 8.0}, injured, engine="ml")

    assert merged["per_gameweek"][0]["xpts"] == 0.0
    assert merged["horizon_xpts"] == 0.0


def test_merge_flags_components_as_estimated(elements):
    base = {
        "per_gameweek": [
            {"gameweek": 13, "xpts": 4.0, "fixtures": [{}], "components": {"goals": 4.0}}
        ],
        "horizon_xpts": 4.0,
        "xpts_next": 4.0,
    }

    merged = ml_scorer._merge(base, {13: 6.0}, elements[0], engine="ml")

    entry = merged["per_gameweek"][0]
    assert entry["components_estimated"] is True
    assert entry["xpts_baseline"] == 4.0
    assert entry["xpts"] == pytest.approx(6.0)
    assert sum(entry["components"].values()) == pytest.approx(6.0, abs=0.01)


def test_blend_is_the_mean_of_the_two_engines(elements):
    base = {
        "per_gameweek": [
            {"gameweek": 13, "xpts": 4.0, "fixtures": [{}], "components": {"goals": 4.0}}
        ],
        "horizon_xpts": 4.0,
        "xpts_next": 4.0,
    }

    merged = ml_scorer._merge(base, {13: 8.0}, elements[0], engine="blend")

    assert merged["per_gameweek"][0]["xpts"] == pytest.approx(6.0)


# ── Early-season gate ───────────────────────────────────────────────────────


def test_the_gate_is_derived_from_the_training_window_not_hardcoded():
    """Retraining with a different MIN_GAMEWEEK must move the gate with it."""
    from ml import config

    assert ml_scorer.min_team_games() == config.MIN_GAMEWEEK - 1


def test_the_model_is_not_used_before_it_has_seen_a_comparable_season(
    monkeypatch, elements, fixtures, teams, events
):
    """Between seasons the features are all zero and the model invents numbers.

    Asking for `ml` this early must return xpts and *say* xpts, not relabel
    hand-built numbers as model output.
    """
    monkeypatch.setattr(ml_scorer.xpts, "team_games_played", lambda *a, **k: 0)

    projections, engine_used = ml_scorer.project_all(
        elements, fixtures, teams, events, from_gameweek=1, horizon=3, engine="ml"
    )

    assert engine_used == "xpts"
    baseline = ml_scorer.xpts.project_all(
        elements, fixtures, teams, events, 1, 3
    )
    assert projections == baseline


def test_blend_is_gated_too(monkeypatch, elements, fixtures, teams, events):
    """`blend` averages in the model, so it inherits the same problem."""
    monkeypatch.setattr(ml_scorer.xpts, "team_games_played", lambda *a, **k: 2)

    _, engine_used = ml_scorer.project_all(
        elements, fixtures, teams, events, from_gameweek=3, horizon=3, engine="blend"
    )

    assert engine_used == "xpts"


class _StubPredictor:
    """Stands in for a loaded artifact, so this doesn't need a trained model.

    The file's autouse fixture deliberately hides any real artifact; what's
    under test here is the *early-season* branch, which only exists when a
    model is present, so one has to be faked.
    """

    name = "stub"
    trained_on = ["2019-20"]
    feature_names = ["form"]
    metadata: dict = {}


def test_the_status_response_explains_why_the_model_is_inactive(monkeypatch):
    """A silent fallback looks like a broken engine picker."""
    monkeypatch.setattr(ml_scorer, "_predictor", lambda: _StubPredictor())

    described = ml_scorer.describe(team_games=0)

    assert described["available"] is True
    assert described["active"] is False
    assert described["engines"] == ["xpts"]
    assert "0 league game" in described["reason"]


def test_the_status_response_reports_the_model_live_once_the_season_is_old_enough(
    monkeypatch,
):
    monkeypatch.setattr(ml_scorer, "_predictor", lambda: _StubPredictor())

    described = ml_scorer.describe(team_games=ml_scorer.min_team_games())

    assert described["active"] is True
    assert described["engines"] == list(ml_scorer.ENGINES)
    assert "reason" not in described


# ------------------------- a model answering about a season it never saw


def _teams(played):
    return [{"id": i, "short_name": f"T{i}", "played": played} for i in range(1, 21)]


def test_a_season_younger_than_the_training_floor_falls_back(monkeypatch):
    """`ml.config.MIN_GAMEWEEK` drops the opening gameweeks from training,
    because a season's cumulative per-90 features are computed over a handful of
    minutes there and are mostly noise. Serving those weeks anyway asks the
    model about a state it was never shown, and trees answer regardless by
    extrapolating from the nearest thing they know.
    """
    assert ml_scorer._season_is_younger_than_training(_teams(1), [])
    assert ml_scorer._season_is_younger_than_training(_teams(4), [])


def test_once_the_season_is_old_enough_the_model_is_used_again():
    assert not ml_scorer._season_is_younger_than_training(_teams(5), [])
    assert not ml_scorer._season_is_younger_than_training(_teams(20), [])


def test_the_rollover_counts_as_younger_than_anything():
    """0 played is the youngest a season gets, and squarely inside the window."""
    assert ml_scorer._season_is_younger_than_training(_teams(0), [])


def test_the_caller_is_told_which_engine_actually_answered(monkeypatch):
    """The response must not label hand-built numbers as model output. The same
    honesty that covers a missing artifact has to cover this."""
    monkeypatch.setattr(ml_scorer, "_predictor", lambda: object())
    monkeypatch.setattr(
        ml_scorer.xpts, "project_all", lambda *a, **k: {1: {"horizon_xpts": 5.0}}
    )

    _, used = ml_scorer.project_all(
        [], [], _teams(2), [], from_gameweek=1, horizon=5, engine="ml"
    )

    assert used == "xpts"
