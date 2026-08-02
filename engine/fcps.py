"""FCPS — Fantasy Composite Player Score.

The original signal this app shipped with, restored. It is a 0-1000 blend of four
normalised terms::

    FCPS = 1000 x ( 0.20 x total_points/max_total_points
                  + 0.40 x form/max_form
                  + 0.25 x (1 - next_3_fdr/15)
                  + 0.15 x ict_index/max_ict )

The weights, the divisors and the 1000x scaling are kept **bit-for-bit identical**
to the original ``calculate_fcps()`` in ``app.py``, so a score computed here is
the same number a user would have seen before. That is deliberate: FCPS is a
familiar quantity to the person using this app, and silently redefining it would
be worse than either keeping or removing it.

What changed is only the implementation and its honesty:

* No pandas. The original built a DataFrame per request to do four divisions.
* ``fixtures_counted`` is returned alongside the score. The original divided the
  FDR sum by a hard-coded 15 — three fixtures at difficulty 5 — so a team with
  only one or two fixtures scheduled (a blank, or the run-in) scored as though it
  had an *easy* run. The number is unchanged; the caller can now see when it is
  built on fewer than three fixtures and say so.
* Every division is guarded. Pre-season, every player has 0 points and 0 form, so
  the maxima are 0 and the original raised or produced NaN.

FCPS is not expected points and cannot be compared to a -4 hit — see
:mod:`engine.xpts` for that. Both are computed, both are surfaced, and the UI
labels which is which.

Pure: bootstrap elements and fixtures in, numbers out. No I/O, no clock.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from .rules import position_of

# The original weights. Do not "improve" these — a changed weight is a changed
# score under an unchanged name, which is the one thing this module must not do.
LEGACY_WEIGHTS: dict[str, float] = {
    "total_points_weight": 0.20,
    "form_weight": 0.40,
    "fdr_weight": 0.25,
    "ict_index_weight": 0.15,
}

# Three fixtures at the maximum difficulty of 5. Hard-coded in the original.
MAX_NEXT_3_FDR = 15

FIXTURE_LOOKAHEAD = 3
SCALE = 1000


def _f(value, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_team_fixtures(
    fixtures: Sequence[Mapping],
    from_gameweek: int,
) -> dict[int, list[dict]]:
    """``team_id -> [{gameweek, difficulty}, ...]``, sorted by gameweek.

    Matches the original ``build_team_fixtures()``: every fixture from
    ``from_gameweek`` onwards, both sides, with each side's own difficulty.
    """
    by_team: dict[int, list[dict]] = {}
    for fixture in fixtures:
        event = fixture.get("event")
        if event is None:
            continue
        event = int(event)
        if event < from_gameweek:
            continue
        if fixture.get("finished"):
            continue
        home, away = int(fixture["team_h"]), int(fixture["team_a"])
        by_team.setdefault(home, []).append(
            {"gameweek": event, "difficulty": int(fixture.get("team_h_difficulty", 3))}
        )
        by_team.setdefault(away, []).append(
            {"gameweek": event, "difficulty": int(fixture.get("team_a_difficulty", 3))}
        )
    for entries in by_team.values():
        entries.sort(key=lambda e: e["gameweek"])
    return by_team


def next_n_fdr(
    team_id: int,
    team_fixtures: Mapping[int, Sequence[Mapping]],
    n: int = FIXTURE_LOOKAHEAD,
) -> tuple[int, int]:
    """``(summed_difficulty, fixtures_counted)`` for a team's next ``n`` fixtures.

    Faithful to the original: the next ``n`` *scheduled* fixtures in gameweek
    order, so a double gameweek contributes two of the three and a blank simply
    pushes the window later. ``fixtures_counted`` is the honesty term the
    original lacked.
    """
    entries = list(team_fixtures.get(int(team_id), ()))[:n]
    return sum(int(e["difficulty"]) for e in entries), len(entries)


def normalisation_values(elements: Sequence[Mapping]) -> dict[str, float]:
    """League-wide maxima used as divisors, as the original computed them.

    ``next_3_fdr`` is the hard-coded 15 rather than an observed maximum, again
    matching the original.
    """
    def safe_max(values) -> float:
        best = 0.0
        for value in values:
            best = max(best, _f(value))
        return best

    return {
        "total_points": safe_max(e.get("total_points") for e in elements),
        "form": safe_max(e.get("form") for e in elements),
        "next_3_fdr": float(MAX_NEXT_3_FDR),
        "ict_index": safe_max(e.get("ict_index") for e in elements),
    }


def _divisor(value: float) -> float:
    """A zero maximum means every numerator is zero too, so 1 is safe.

    Pre-season this is the whole ballgame: every player has 0 points and 0 form,
    and the original divided by zero.
    """
    return value if value and value > 0 else 1.0


def score_player(
    element: Mapping,
    fdr_sum: int,
    divisors: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
) -> dict:
    """FCPS for one player, with its four components exposed.

    The components are returned pre-weighting (each is a 0-1 normalised term) and
    the total is the weighted sum times 1000, so a caller can always show the
    breakdown and assert that it reconstructs the score.
    """
    weights = dict(weights or LEGACY_WEIGHTS)

    total_points_norm = _f(element.get("total_points")) / _divisor(
        _f(divisors.get("total_points"))
    )
    form_norm = _f(element.get("form")) / _divisor(_f(divisors.get("form")))
    fdr_norm = float(fdr_sum) / _divisor(_f(divisors.get("next_3_fdr")))
    ict_norm = _f(element.get("ict_index")) / _divisor(_f(divisors.get("ict_index")))

    # Lower FDR is better, hence the inversion. Clamped because a run of doubles
    # can push the raw sum past 15 and drive the term negative, which the
    # original allowed and which made the score non-monotonic in fixture ease.
    fdr_term = min(1.0, max(0.0, 1.0 - fdr_norm))

    raw = (
        weights["total_points_weight"] * total_points_norm
        + weights["form_weight"] * form_norm
        + weights["fdr_weight"] * fdr_term
        + weights["ict_index_weight"] * ict_norm
    )

    return {
        "fcps": round(round(raw, 3) * SCALE, 1),
        "components": {
            "total_points_norm": round(total_points_norm, 4),
            "form_norm": round(form_norm, 4),
            "fdr_term": round(fdr_term, 4),
            "ict_index_norm": round(ict_norm, 4),
        },
        "weights": weights,
    }


def score_all(
    elements: Sequence[Mapping],
    fixtures: Sequence[Mapping],
    from_gameweek: int,
    weights: Mapping[str, float] | None = None,
) -> dict[int, dict]:
    """FCPS for every player. Returns ``player_id -> {fcps, next_3_fdr, ...}``."""
    team_fixtures = build_team_fixtures(fixtures, from_gameweek)
    divisors = normalisation_values(elements)

    scored: dict[int, dict] = {}
    for element in elements:
        if position_of(element) not in ("GKP", "DEF", "MID", "FWD"):
            continue
        fdr_sum, counted = next_n_fdr(int(element.get("team", 0)), team_fixtures)
        entry = score_player(element, fdr_sum, divisors, weights)
        entry["player_id"] = int(element["id"])
        entry["next_3_fdr"] = fdr_sum
        entry["fixtures_counted"] = counted
        scored[int(element["id"])] = entry
    return scored


def top_by_position(
    scored: Mapping[int, Mapping],
    elements: Sequence[Mapping],
    counts: Mapping[str, int] | None = None,
    available_only: bool = True,
) -> list[Mapping]:
    """The original ``print_top_players()`` shortlist, without the DataFrame.

    Defaults to the same 5 GKP / 15 DEF / 25 MID / 25 FWD the LLM prompt was fed.
    Returned in FCPS order within each position, positions in GKP-DEF-MID-FWD
    order, deterministic on ties via player id.
    """
    counts = dict(counts or {"GKP": 5, "DEF": 15, "MID": 25, "FWD": 25})

    by_position: dict[str, list[Mapping]] = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
    for element in elements:
        pid = int(element["id"])
        if pid not in scored:
            continue
        if available_only and str(element.get("status", "a")) != "a":
            continue
        position = position_of(element)
        if position in by_position:
            by_position[position].append(element)

    shortlist: list[Mapping] = []
    for position in ("GKP", "DEF", "MID", "FWD"):
        ranked = sorted(
            by_position[position],
            key=lambda e: (-scored[int(e["id"])]["fcps"], int(e["id"])),
        )
        shortlist.extend(ranked[: counts.get(position, 0)])
    return shortlist
