"""What a player actually scored, and where each point came from.

Everything else in this package projects. This reports: it takes FPL's own
per-gameweek record and attributes the points to the things that earned them.

FPL publishes the *stats* for a gameweek and the *total*, and nothing in
between. So a manager sees "2 points" against a striker who played ninety
minutes and cannot tell whether that was a quiet game or a goal cancelled out by
two yellow cards without opening a separate site. The attribution here is
derived from the published scoring rules and is checked against FPL's own total
— where the two disagree the difference is reported rather than hidden, because
a breakdown that silently disagrees with the headline is worse than no
breakdown.

Pure: a history row in, a breakdown out. No I/O.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from .xpts import CLEAN_SHEET_POINTS, GOAL_POINTS

#: Two points for sixty minutes or more, one for anything less than that but
#: more than none.
APPEARANCE_LONG_MINUTES = 60

#: Goals conceded cost a point per two, for goalkeepers and defenders only.
CONCEDED_PER_POINT = 2
CONCEDED_POSITIONS = ("GKP", "DEF")

#: Saves are worth a point per three, goalkeepers only.
SAVES_PER_POINT = 3

#: Introduced for 2025-26. Defenders need ten defensive actions, everyone else
#: twelve, for two points.
DEFCON_POINTS = 2
DEFCON_THRESHOLD = {"GKP": 999, "DEF": 10, "MID": 12, "FWD": 12}

CARD_POINTS = {"yellow_cards": -1, "red_cards": -3}
PENALTY_MISS_POINTS = -2
PENALTY_SAVE_POINTS = 5
OWN_GOAL_POINTS = -2


def _int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def breakdown(row: Mapping, position: str) -> dict:
    """One gameweek's points, attributed.

    ``row`` is an entry from FPL's ``element-summary`` history. Returns the
    parts, their sum, FPL's own total, and the gap between them.

    The gap is the honest part. FPL's scoring has corners this does not model —
    a goalkeeper's penalty save bonus tiers, mid-season rule tweaks — and a
    breakdown that quietly rounds itself to match the headline would hide
    exactly the cases worth knowing about.
    """
    minutes = _int(row.get("minutes"))
    parts: dict[str, int] = {}

    if minutes > 0:
        parts["appearance"] = 2 if minutes >= APPEARANCE_LONG_MINUTES else 1

    goals = _int(row.get("goals_scored"))
    if goals:
        parts["goals"] = goals * GOAL_POINTS.get(position, 4)

    assists = _int(row.get("assists"))
    if assists:
        parts["assists"] = assists * 3

    # A clean sheet only counts for a player who was on for an hour; FPL awards
    # nothing to a substitute who came on at 80 minutes with the sheet intact.
    if _int(row.get("clean_sheets")) and minutes >= APPEARANCE_LONG_MINUTES:
        points = CLEAN_SHEET_POINTS.get(position, 0)
        if points:
            parts["clean_sheet"] = points

    conceded = _int(row.get("goals_conceded"))
    if position in CONCEDED_POSITIONS and conceded >= CONCEDED_PER_POINT:
        parts["conceded"] = -(conceded // CONCEDED_PER_POINT)

    saves = _int(row.get("saves"))
    if position == "GKP" and saves >= SAVES_PER_POINT:
        parts["saves"] = saves // SAVES_PER_POINT

    defcon = _int(row.get("defensive_contribution"))
    if defcon >= DEFCON_THRESHOLD.get(position, 999):
        parts["defensive_contribution"] = DEFCON_POINTS

    bonus = _int(row.get("bonus"))
    if bonus:
        parts["bonus"] = bonus

    for key, per in CARD_POINTS.items():
        count = _int(row.get(key))
        if count:
            parts[key.replace("_cards", "_card")] = count * per

    for key, per in (
        ("own_goals", OWN_GOAL_POINTS),
        ("penalties_missed", PENALTY_MISS_POINTS),
        ("penalties_saved", PENALTY_SAVE_POINTS),
    ):
        count = _int(row.get(key))
        if count:
            parts[key] = count * per

    attributed = sum(parts.values())
    total = _int(row.get("total_points"))
    return {
        "gameweek": _int(row.get("round")),
        "minutes": minutes,
        "opponent": _int(row.get("opponent_team")),
        "was_home": bool(row.get("was_home")),
        "total_points": total,
        "parts": parts,
        "attributed": attributed,
        # Nonzero means the rules above did not fully explain FPL's number.
        # Surfaced rather than absorbed: a breakdown that disagrees with the
        # headline and says so is useful; one that hides it is a lie.
        "unexplained": total - attributed,
    }


def recent(
    history: Sequence[Mapping], position: str, limit: int = 5
) -> list[dict]:
    """The last ``limit`` gameweeks, most recent first.

    Ordered by gameweek rather than by list position: FPL returns these in
    order, but a double gameweek puts two rows on the same number and a
    postponement can land one out of sequence.
    """
    rows = sorted(history, key=lambda r: _int(r.get("round")), reverse=True)
    return [breakdown(row, position) for row in rows[:limit]]
