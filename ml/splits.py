"""Time-based splits. There is no random split in this package, by design.

A random train/test split on a player-gameweek panel is close to worthless here,
and worse than worthless because it produces flattering numbers. Two mechanisms:

*Within-player leakage.* Put GW30 in train and GW12 of the same season in test,
and the model has effectively been told how the player's season went before
predicting its middle. The features are cumulative, so the same underlying
observations appear on both sides.

*Regime leakage.* Fixture difficulty, team quality, price inflation and the
scoring rules are all season-scoped. A random split lets the model calibrate to a
season it is then scored on.

So: train on the past, test on the future, always. Three protocols are provided.

``season_split``
    Whole seasons to train / validate / test, per :mod:`ml.config`. The headline
    number. Honest but pessimistic — a deployed model would have been retrained
    during the test season rather than frozen before it.

``walk_forward``
    Expanding window inside a season: to predict gameweek *g*, train on
    everything up to *g-1*. This is what deployment actually looks like, and the
    gap between it and ``season_split`` is the cost of not retraining.

``purged_season_split``
    ``season_split`` with a gap of gameweeks dropped at the boundary. Features are
    built from a rolling window, so the last few rows of train and the first few
    of test share observations; the purge removes the overlap. Standard practice
    in financial backtesting for exactly this reason.
"""

from __future__ import annotations

from typing import Iterator, Sequence

from . import config


def season_split(frame, train=None, valid=None, test=None):
    """``(train, valid, test)`` frames by season."""
    train = tuple(train or config.TRAIN_SEASONS)
    valid = tuple(valid or config.VALID_SEASONS)
    test = tuple(test or config.TEST_SEASONS)

    overlap = (set(train) & set(valid)) | (set(train) & set(test)) | (set(valid) & set(test))
    if overlap:
        raise ValueError(f"seasons appear in more than one split: {sorted(overlap)}")

    return (
        frame[frame["season"].isin(train)].copy(),
        frame[frame["season"].isin(valid)].copy(),
        frame[frame["season"].isin(test)].copy(),
    )


PURGE_GAMEWEEKS = 4  # the width of the rolling `form` window in ml.features


def purged_season_split(frame, purge_gameweeks: int = PURGE_GAMEWEEKS, **kwargs):
    """``season_split`` with the last ``purge_gameweeks`` of each train season cut.

    The rolling ``form`` window is 4 gameweeks wide, so a train row at GW38 and a
    test row at GW1 of the next season never actually overlap — seasons are
    disjoint. The purge matters for :func:`walk_forward`, where the boundary is
    inside a season, and is offered here for symmetry when a caller splits by
    gameweek rather than by season.
    """
    train, valid, test = season_split(frame, **kwargs)
    if purge_gameweeks > 0:
        cutoff = 38 - purge_gameweeks
        train = train[train["GW"] <= cutoff].copy()
    return train, valid, test


def walk_forward(
    frame,
    test_season: str,
    start_gameweek: int = 8,
    step: int = 1,
    history_seasons: Sequence[str] | None = None,
) -> Iterator[tuple]:
    """Yield ``(train, test, gameweek)`` for an expanding window.

    ``train`` is every prior season plus this season up to ``gameweek - 1``;
    ``test`` is exactly ``gameweek``. ``step`` > 1 retrains less often, which is
    both faster and a closer match to how anyone would actually operate it.

    Starting at gameweek 8 rather than 1 is not arbitrary: before then the
    within-season history is thin enough that the split is really just
    "train on other seasons", which ``season_split`` already measures.
    """
    history_seasons = tuple(
        history_seasons
        if history_seasons is not None
        else [s for s in config.SEASONS if s < test_season]
    )
    prior = frame[frame["season"].isin(history_seasons)]
    current = frame[frame["season"] == test_season]
    if current.empty:
        return

    import pandas as pd

    gameweeks = sorted(int(g) for g in current["GW"].unique())
    for gameweek in gameweeks:
        if gameweek < start_gameweek or (gameweek - start_gameweek) % step:
            continue
        train = pd.concat(
            [prior, current[current["GW"] < gameweek]], ignore_index=True
        )
        test = current[current["GW"] == gameweek]
        if train.empty or test.empty:
            continue
        yield train, test, gameweek
