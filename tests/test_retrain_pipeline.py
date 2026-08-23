"""The two properties that make retraining safe to repeat.

A pipeline you run once by hand needs neither. One that runs on a schedule needs
both, because nobody is watching the run that goes wrong.
"""

import pytest

from ml import config, train


# ------------------------------------------------------- splits roll forward


def test_the_two_newest_seasons_are_held_out():
    train_s, valid_s, test_s = config.rolled_splits(
        ("2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25")
    )

    assert test_s == ("2024-25",)
    assert valid_s == ("2023-24",)
    assert train_s == ("2019-20", "2020-21", "2021-22", "2022-23")


def test_adding_a_season_rolls_the_whole_protocol_forward():
    """Written out by hand, a retrain that appended a season would have kept
    testing on the season it had just started training on — the leak the split
    module exists to prevent, introduced by the act of retraining."""
    before = config.rolled_splits(("2022-23", "2023-24", "2024-25"))
    after = config.rolled_splits(("2022-23", "2023-24", "2024-25", "2025-26"))

    assert before[2] == ("2024-25",)
    assert after[2] == ("2025-26",)
    assert "2024-25" in after[0] + after[1], "last year's test set is now usable"
    assert "2025-26" not in after[0] + after[1], "and this year's is not"


def test_seasons_out_of_order_are_still_split_chronologically():
    train_s, valid_s, test_s = config.rolled_splits(
        ("2024-25", "2019-20", "2023-24", "2022-23")
    )

    assert test_s == ("2024-25",)
    assert valid_s == ("2023-24",)


def test_too_few_seasons_holds_nothing_out_rather_than_pretending():
    """Better an empty split the trainer refuses on than a held-out set that
    isn't one."""
    train_s, valid_s, test_s = config.rolled_splits(("2023-24", "2024-25"))

    assert valid_s == () and test_s == ()
    assert train_s == ("2023-24", "2024-25")


# --------------------------------------------------------------- the gate


class _Frame(dict):
    """Enough of a DataFrame for the gate: copy() and item assignment."""

    def copy(self):
        return _Frame(self)


def test_a_worse_candidate_does_not_replace_the_deployed_model(monkeypatch):
    """The property that matters. A pipeline that always saves will eventually
    deploy a regression and tell nobody."""
    monkeypatch.setattr(train.models, "load", lambda: ("incumbent", {}))
    monkeypatch.setattr(train.models, "predict", lambda m, f: [0.0])
    monkeypatch.setattr(
        train.metrics, "evaluate", lambda f, c: {train.SELECTION_METRIC: 0.80}
    )

    assert train.incumbent_validation_score(_Frame()) == 0.80


def test_no_artifact_means_nothing_to_beat(monkeypatch):
    monkeypatch.setattr(train.models, "load", lambda: (None, None))

    assert train.incumbent_validation_score(_Frame()) is None


def test_an_unscoreable_artifact_is_treated_as_absent(monkeypatch):
    """A feature list that no longer matches is a deliberate reset, not a
    regression, and must not block the first run after one."""

    def _boom(*a, **k):
        raise ValueError("feature mismatch")

    monkeypatch.setattr(train.models, "load", lambda: ("incumbent", {}))
    monkeypatch.setattr(train.models, "predict", _boom)

    assert train.incumbent_validation_score(_Frame()) is None


def test_the_margin_is_not_zero():
    """Two refits of the same data differ by noise. Shipping on any improvement
    means shipping on rounding about half the time."""
    assert train.GATE_MARGIN > 0
