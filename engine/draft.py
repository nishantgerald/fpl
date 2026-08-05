"""Pick an opening 15 — the one question the app couldn't answer.

:mod:`engine.optimizer` solves transfers *from* an existing squad, given a bank
and a free-transfer count. Before the Gameweek 1 deadline none of those inputs
exist: FPL has no picks for an entry, so every advice route correctly returns
``season_not_started`` and the app has nothing to say for the three weeks when
the only thing anyone is doing is picking their initial fifteen.

This module fills that window. Two things about the formulation matter, because
both are easy to get wrong and both change the answer:

*The objective is the starting XI, not the fifteen.* Bench players score nothing
unless someone ahead of them doesn't play. Maximising the sum of all fifteen
buys four expensive substitutes and starves the pitch; every real squad spends
the bench to the floor. So squad membership and XI selection are separate
decisions and only the XI is in the objective.

*It is solved exactly where possible.* Budget, the three-per-club cap, the
positional quotas and the formation interact, so a greedy points-per-million
pick is routinely globally wrong. When SciPy is present this is a mixed-integer
program. When it isn't — the web process is not required to have the ML stack —
it degrades to a documented greedy pass with local swaps rather than failing.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from . import rules

BUDGET_TENTHS = 1000  # £100.0m, in the integer tenths used everywhere else.
SQUAD_QUOTAS = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_CLUB = 3
XI_SIZE = 11
# The eight legal formations collapse to per-position bounds on the XI.
XI_BOUNDS = {"GKP": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}


def candidates(
    elements: Sequence[Mapping],
    projections: Mapping[int, Mapping],
    teams: Mapping[int, str],
) -> list[dict]:
    """Selectable players, as flat rows.

    Unavailable players are dropped outright — an opening squad has no reason to
    carry a known injury. Nothing else is filtered: with the between-seasons
    denominator fixed, :mod:`engine.xpts` already scores a player with no
    minutes at zero, so the optimiser will not reach for one.
    """
    rows = []
    for element in elements:
        pid = int(element["id"])
        projection = projections.get(pid)
        if not projection:
            continue
        if str(element.get("status", "a")) in rules.UNAVAILABLE_STATUSES:
            continue
        value = float(projection.get("horizon_xpts") or 0.0)
        if value <= 0:
            continue
        rows.append(
            {
                "id": pid,
                "web_name": element.get("web_name", ""),
                "name": f"{element.get('first_name', '')} "
                f"{element.get('second_name', '')}".strip(),
                "position": rules.position_of(element),
                "team_id": int(element.get("team", 0)),
                "team": teams.get(int(element.get("team", 0)), "UNK"),
                "price_tenths": int(element.get("now_cost", 0)),
                "price": int(element.get("now_cost", 0)) / 10.0,
                "value": value,
                "xpts_next": float(projection.get("xpts_next") or 0.0),
                "total_points": int(element.get("total_points", 0)),
                "selected_by_percent": _f(element.get("selected_by_percent")),
                "status": str(element.get("status", "a")),
            }
        )
    return rows


def _f(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build(
    rows: Sequence[Mapping],
    budget_tenths: int = BUDGET_TENTHS,
    pinned: Sequence[str] = (),
) -> dict | None:
    """A legal fifteen maximising projected XI points, or ``None``.

    ``pinned`` names players (by ``web_name``) that must appear — the seam for
    research the projections can't see, such as a rule the model was never
    trained on or a pre-season signing.
    """
    solved = _solve_exact(rows, budget_tenths, set(pinned))
    if solved is None:
        solved = _solve_greedy(rows, budget_tenths, set(pinned))
    if solved is None:
        return None
    return _shape(solved, rows)


def _solve_exact(rows, budget_tenths: int, pinned: set[str]) -> list[int] | None:
    """Mixed-integer program. Returns squad indices, or None if SciPy is absent."""
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError:
        return None

    n = len(rows)
    if n < 15:
        return None

    value = np.array([r["value"] for r in rows], dtype="float64")
    price = np.array([r["price_tenths"] for r in rows], dtype="float64")
    positions = [r["position"] for r in rows]
    clubs = [r["team_id"] for r in rows]

    # Variables: squad membership, then XI membership. Only the XI scores.
    objective = np.concatenate([np.zeros(n), -value])
    constraints = []

    def row(squad_part, xi_part):
        return np.concatenate([squad_part, xi_part])

    constraints.append(LinearConstraint(row(np.ones(n), np.zeros(n)), 15, 15))
    constraints.append(LinearConstraint(row(np.zeros(n), np.ones(n)), XI_SIZE, XI_SIZE))
    constraints.append(
        LinearConstraint(row(price, np.zeros(n)), -np.inf, float(budget_tenths))
    )

    for position, quota in SQUAD_QUOTAS.items():
        mask = np.array([1.0 if p == position else 0.0 for p in positions])
        constraints.append(LinearConstraint(row(mask, np.zeros(n)), quota, quota))
        low, high = XI_BOUNDS[position]
        constraints.append(LinearConstraint(row(np.zeros(n), mask), low, high))

    for club in sorted(set(clubs)):
        mask = np.array([1.0 if c == club else 0.0 for c in clubs])
        constraints.append(
            LinearConstraint(row(mask, np.zeros(n)), -np.inf, MAX_PER_CLUB)
        )

    # A player can only start if they are in the squad.
    implication = np.zeros((n, 2 * n))
    for i in range(n):
        implication[i, i] = -1.0
        implication[i, n + i] = 1.0
    constraints.append(LinearConstraint(implication, -np.inf, 0))

    for i, candidate in enumerate(rows):
        if candidate["web_name"] in pinned:
            pin = np.zeros(2 * n)
            pin[i] = 1.0
            constraints.append(LinearConstraint(pin, 1, 1))

    result = milp(
        c=objective,
        constraints=constraints,
        integrality=np.ones(2 * n),
        bounds=Bounds(0, 1),
    )
    if not result.success:
        return None

    chosen = np.round(result.x).astype(int)
    return [i for i in range(n) if chosen[i] == 1]


def _solve_greedy(rows, budget_tenths: int, pinned: set[str]) -> list[int] | None:
    """Fallback for a web process without SciPy.

    Fills each position by value-per-million, respecting budget and the club
    cap, then makes one upgrade pass. Materially worse than the exact solve —
    the interactions between budget and quotas are what greedy misses — but a
    legal squad beats a 500.
    """
    order = sorted(
        range(len(rows)),
        key=lambda i: -(rows[i]["value"] / max(rows[i]["price_tenths"], 1)),
    )
    chosen: list[int] = []
    spent = 0
    per_position = {p: 0 for p in SQUAD_QUOTAS}
    per_club: dict[int, int] = {}

    for pool_pass in (True, False):
        for i in order:
            if len(chosen) == 15:
                break
            candidate = rows[i]
            if i in chosen:
                continue
            # First pass takes pinned players only, so they can't be crowded out.
            if pool_pass != (candidate["web_name"] in pinned):
                continue
            position = candidate["position"]
            if per_position[position] >= SQUAD_QUOTAS[position]:
                continue
            club = candidate["team_id"]
            if per_club.get(club, 0) >= MAX_PER_CLUB:
                continue
            if spent + candidate["price_tenths"] > budget_tenths:
                continue
            chosen.append(i)
            spent += candidate["price_tenths"]
            per_position[position] += 1
            per_club[club] = per_club.get(club, 0) + 1

    if len(chosen) != 15:
        return None
    return chosen


def _shape(indices: Sequence[int], rows: Sequence[Mapping]) -> dict:
    """Squad indices to the payload the client renders."""
    squad = [dict(rows[i]) for i in indices]
    xi = _best_xi(squad)
    xi_ids = {p["id"] for p in xi}
    bench = [p for p in squad if p["id"] not in xi_ids]

    # Outfield first, keepers last: an outfield substitution is far likelier.
    bench.sort(key=lambda p: (p["position"] == "GKP", -p["value"]))

    ranked = sorted(xi, key=lambda p: -p["value"])
    captain = ranked[0] if ranked else None
    vice = ranked[1] if len(ranked) > 1 else None

    return {
        "squad": squad,
        "starting_xi": xi,
        "bench": bench,
        "formation": _formation(xi),
        "captain": captain,
        "vice_captain": vice,
        "cost": round(sum(p["price"] for p in squad), 1),
        "budget": BUDGET_TENTHS / 10.0,
        "remaining": round(BUDGET_TENTHS / 10.0 - sum(p["price"] for p in squad), 1),
        "xi_projected": round(sum(p["value"] for p in xi), 1),
    }


def _best_xi(squad: Sequence[Mapping]) -> list[dict]:
    """Highest-projecting legal eleven from a fifteen.

    Enumerates the legal shapes rather than assuming one: which formation is
    best depends on the squad, and picking it wrong understates a squad that was
    optimised for a different one.
    """
    by_position = {p: [] for p in SQUAD_QUOTAS}
    for player in squad:
        by_position.setdefault(player["position"], []).append(dict(player))
    for group in by_position.values():
        group.sort(key=lambda p: -p["value"])

    best: list[dict] = []
    best_value = -1.0
    for defenders in range(XI_BOUNDS["DEF"][0], XI_BOUNDS["DEF"][1] + 1):
        for midfielders in range(XI_BOUNDS["MID"][0], XI_BOUNDS["MID"][1] + 1):
            forwards = XI_SIZE - 1 - defenders - midfielders
            if not (XI_BOUNDS["FWD"][0] <= forwards <= XI_BOUNDS["FWD"][1]):
                continue
            counts = {"GKP": 1, "DEF": defenders, "MID": midfielders, "FWD": forwards}
            if any(len(by_position[p]) < c for p, c in counts.items()):
                continue
            eleven = [q for p, c in counts.items() for q in by_position[p][:c]]
            total = sum(q["value"] for q in eleven)
            if total > best_value:
                best_value, best = total, eleven
    return best


def _formation(xi: Sequence[Mapping]) -> str:
    counts = {p: 0 for p in SQUAD_QUOTAS}
    for player in xi:
        counts[player["position"]] = counts.get(player["position"], 0) + 1
    return f"{counts['DEF']}-{counts['MID']}-{counts['FWD']}"
