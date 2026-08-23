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


def test_the_incumbent_is_judged_on_the_same_exam(monkeypatch):
    """The subtlety the first cut of this got wrong.

    The artifact was refit on train + validation before being frozen, so
    scoring it on validation now scores it on rows it was fitted to. It wins
    that by construction, and a gate built on it keeps the first model ever
    trained for ever. Its metadata carries the score its *train-only* fit
    earned, which is what the candidate's number is.
    """
    monkeypatch.setattr(
        train.models,
        "load",
        lambda: (
            "model",
            {
                "model": "hgb_poisson",
                "valid_seasons": list(config.VALID_SEASONS),
                "validation_scores": {"hgb_poisson": {train.SELECTION_METRIC: 0.76}},
            },
        ),
    )

    score, why = train.incumbent_validation_score()

    assert score == pytest.approx(0.76)
    assert "hgb_poisson" in why


def test_a_moved_split_is_not_a_comparison(monkeypatch):
    """Seasons roll forward. Once they do, the recorded number describes a
    different set of gameweeks — two models, two exams. Comparing across that
    silently would gate on a regime change."""
    monkeypatch.setattr(
        train.models,
        "load",
        lambda: (
            "model",
            {
                "model": "hgb_poisson",
                "valid_seasons": ["1999-00"],
                "validation_scores": {"hgb_poisson": {train.SELECTION_METRIC: 0.99}},
            },
        ),
    )

    score, why = train.incumbent_validation_score()

    assert score is None
    assert "split moved" in why


def test_no_artifact_means_nothing_to_beat(monkeypatch):
    monkeypatch.setattr(train.models, "load", lambda: (None, None))

    score, why = train.incumbent_validation_score()

    assert score is None
    assert "no deployed artifact" in why


def test_an_artifact_with_no_recorded_score_is_treated_as_absent(monkeypatch):
    """A model frozen before the gate existed has no comparable number, and must
    not block the first run after it."""
    monkeypatch.setattr(
        train.models,
        "load",
        lambda: ("model", {"model": "hgb", "valid_seasons": list(config.VALID_SEASONS)}),
    )

    score, _ = train.incumbent_validation_score()

    assert score is None


def test_the_margin_is_not_zero():
    """Two refits of the same data differ by noise. Shipping on any improvement
    means shipping on rounding about half the time."""
    assert train.GATE_MARGIN > 0
