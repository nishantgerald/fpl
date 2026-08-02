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
