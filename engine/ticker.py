"""Fixture ticker and swing detection.

``/api/fixtures`` has existed since the first commit and no client has ever
called it. Meanwhile the only fixture signal in the product was ``next_3_fdr``,
a sum divided by a hard-coded 15 — which meant a *blank* gameweek scored as an
easy run.

Here blanks and doubles are represented structurally: a gameweek cell holds a
list of fixtures, so a blank is an empty list and a double has two entries. No
sum can be mistaken for the other.

The swing detector is the part a colour grid can't give you: it names the teams
whose fixture run changes character, instead of leaving the user to read it out
of the shading.

Pure: fixtures and teams in, grid out.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from .xpts import build_fixture_index, build_team_index, fixture_multipliers

# A run has to change by this much on the 1-5 FDR scale before we call it a swing.
SWING_THRESHOLD = 1.0

# Gameweeks either side of the split point when comparing runs.
SWING_WINDOW = 3


def build_ticker(
    fixtures: Sequence[Mapping],
    teams: Sequence[Mapping],
    start_gameweek: int,
    count: int = 6,
) -> dict:
    """A team x gameweek difficulty grid, plus detected swings."""
    index = build_fixture_index(fixtures, start_gameweek)
    team_index = build_team_index(teams)
    gameweeks = list(range(start_gameweek, start_gameweek + count))

    rows: list[dict] = []
    for team_id, team in sorted(team_index.items()):
        cells = []
        difficulties: list[int] = []
        att_mults: list[float] = []
        def_mults: list[float] = []
        blanks = doubles = total = 0

        for gw in gameweeks:
            team_fixtures = list(index.get(team_id, {}).get(gw, []))
            if not team_fixtures:
                blanks += 1
            elif len(team_fixtures) > 1:
                doubles += 1
            total += len(team_fixtures)

            for fixture in team_fixtures:
                difficulties.append(fixture["fdr"])
                att, con = fixture_multipliers(fixture, team_index)
                att_mults.append(att)
                def_mults.append(con)

            cells.append(
                {
                    "gameweek": gw,
                    "fixtures": [
                        {
                            "opponent": _short(team_index, fx["opponent"]),
                            "home": fx["home"],
                            "fdr": fx["fdr"],
                        }
                        for fx in team_fixtures
                    ],
                }
            )

        rows.append(
            {
                "id": team_id,
                "short_name": team.get("short_name", "UNK"),
                "name": team.get("name", ""),
                "cells": cells,
                # Averaged over fixtures actually played, so a blank doesn't
                # flatter the run the way a fixed divisor would.
                "avg_fdr": round(sum(difficulties) / len(difficulties), 2)
                if difficulties
                else None,
                "total_fixtures": total,
                "blanks": blanks,
                "doubles": doubles,
                "attack_rating": round(sum(att_mults) / len(att_mults), 3)
                if att_mults
                else None,
                "defence_rating": round(sum(def_mults) / len(def_mults), 3)
                if def_mults
                else None,
            }
        )

    return {
        "start_gameweek": start_gameweek,
        "count": count,
        "teams": rows,
        "swings": detect_swings(rows, gameweeks),
    }


def detect_swings(rows: Sequence[Mapping], gameweeks: Sequence[int]) -> list[dict]:
    """Name the teams whose fixture run changes character.

    Compares the mean difficulty of the first window against the second. Only
    emits when the delta clears the threshold, so the list stays short enough to
    be worth reading.
    """
    if len(gameweeks) < SWING_WINDOW * 2:
        return []

    split = SWING_WINDOW
    swings: list[dict] = []

    for row in rows:
        near = _window_avg(row["cells"][:split])
        far = _window_avg(row["cells"][split : split * 2])
        if near is None or far is None:
            continue
        delta = near - far
        if abs(delta) < SWING_THRESHOLD:
            continue

        improving = delta > 0
        pivot = gameweeks[split]
        swings.append(
            {
                "team": row["short_name"],
                "team_id": row["id"],
                "direction": "improving" if improving else "worsening",
                "from_gameweek": pivot,
                "near_avg_fdr": round(near, 2),
                "far_avg_fdr": round(far, 2),
                "message": (
                    f"{row['name'] or row['short_name']}'s next {split} average "
                    f"{near:.1f} FDR, then {far:.1f} from GW{pivot}. "
                    + (
                        "Their run gets notably easier — a buy-back window."
                        if improving
                        else "Their run gets notably harder — consider selling before then."
                    )
                ),
            }
        )

    swings.sort(key=lambda s: (-abs(s["near_avg_fdr"] - s["far_avg_fdr"]), s["team_id"]))
    return swings


def _window_avg(cells: Sequence[Mapping]) -> float | None:
    difficulties = [fx["fdr"] for cell in cells for fx in cell["fixtures"]]
    if not difficulties:
        return None
    return sum(difficulties) / len(difficulties)


def _short(teams: Mapping[int, Mapping], team_id: int) -> str:
    team = teams.get(int(team_id))
    return str(team.get("short_name", "UNK")) if team else "UNK"
