"""Paths, seasons and the split boundaries. One place to change any of them."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("FPL_ML_DATA", ROOT / "_data"))
ARTIFACT_DIR = Path(os.getenv("FPL_ML_ARTIFACTS", ROOT / "artifacts"))
REPORT_DIR = Path(os.getenv("FPL_ML_REPORTS", ROOT / "reports"))

# vaastav/Fantasy-Premier-League, the standard historical mirror of the FPL API.
# Pinned to raw.githubusercontent so we get files, not HTML.
VAASTAV_RAW = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
)

# Seasons in ascending order. Earlier seasons exist in the mirror but predate the
# `expected_*` columns entirely and predate the 2019 price/bonus changes, so they
# are more distribution shift than signal.
SEASONS: tuple[str, ...] = (
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
)

# Time-based split. Never random — a random split puts gameweek 30 of a season in
# train and gameweek 12 of the same season in test, which leaks a player's
# season-long form backwards and inflates every metric.
#
# Derived from SEASONS rather than written out, so adding a season rolls the
# whole protocol forward: the newest becomes the held-out test, the one before
# it validates, everything earlier trains. Written out by hand, a retrain that
# appended a season would have kept testing on 2024-25 while training on it —
# the leak this module exists to prevent, introduced by the act of retraining.


def rolled_splits(
    seasons: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """``(train, validation, test)`` — the two newest seasons held out, in order.

    Needs three seasons to be meaningful. Below that everything trains and the
    caller gets empty held-out sets, which the trainer reports rather than
    papering over with a random split.
    """
    ordered = tuple(sorted(seasons))
    if len(ordered) < 3:
        return ordered, (), ()
    return ordered[:-2], ordered[-2:-1], ordered[-1:]


TRAIN_SEASONS, VALID_SEASONS, TEST_SEASONS = rolled_splits(SEASONS)

# Gameweeks 1-4 of a season have almost no history to build features from, so
# they are excluded from training and reported separately in the backtest rather
# than being quietly averaged into the headline number.
MIN_GAMEWEEK = 5

# The defensive-contribution rules arrived in 2025-26 and change the points
# distribution for defenders and midfielders. Recorded here so the methodology
# document and the code can't drift apart about it.
RULE_CHANGES = {
    "2025-26": "Defensive contribution points (+2) introduced for DEF/MID/FWD.",
    "2024-25": "No scoring change.",
    "2020-21": "Bonus/BPS unchanged; season played behind closed doors (no home advantage).",
}

ARTIFACT_NAME = "xpts_model.joblib"
METADATA_NAME = "xpts_model.json"


def artifact_path() -> Path:
    return ARTIFACT_DIR / ARTIFACT_NAME


def metadata_path() -> Path:
    return ARTIFACT_DIR / METADATA_NAME


def season_dir(season: str) -> Path:
    return DATA_DIR / season


def ensure_dirs() -> None:
    for path in (DATA_DIR, ARTIFACT_DIR, REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)
