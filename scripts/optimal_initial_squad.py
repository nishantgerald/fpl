"""Pick a rules-legal opening 15 that maximises projected points.

Not part of the web app. The optimiser in ``engine/optimizer.py`` solves a
different problem — transfers *from* an existing squad, given a bank and a
free-transfer count — and there is no code path for picking a squad from
nothing. This is that missing piece, run offline.

Two things about the formulation are worth stating, because both are easy to get
wrong and both change the answer:

*The objective is the starting XI, not the fifteen.* Bench players score nothing
unless someone in front of them doesn't play. Maximising the sum of all fifteen
buys four expensive substitutes and starves the XI; every real FPL squad instead
spends the bench down to the floor and puts the money on the pitch. So squad
membership and XI selection are separate variables, and only the XI is in the
objective.

*It is solved exactly, not greedily.* Greedy points-per-million picks fail here:
the budget, the 3-per-club cap, the positional quotas and the formation interact,
so a locally optimal pick is routinely globally wrong. This is a mixed-integer
program handed to a real solver.

Usage:
    python scripts/optimal_initial_squad.py [--horizon 5] [--min-points 40]
"""

from __future__ import annotations

import argparse
import json
import urllib.request

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

BUDGET = 100.0
SQUAD_QUOTAS = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_CLUB = 3
XI_SIZE = 11
# The eight legal shapes collapse to per-position bounds on the XI.
XI_BOUNDS = {"GKP": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}


def fetch(base: str, horizon: int) -> list[dict]:
    url = f"{base}/api/players?engine=ml&horizon={horizon}"
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)["data"]


def solve(players: list[dict], value_key: str, force: set[str] | None = None) -> dict | None:
    """Exact MILP. Returns the chosen squad and XI, or None if infeasible.

    ``force`` pins named players into the squad. That is the hybrid seam: where
    research knows something the model cannot see — defensive-contribution
    points it was never trained on, a manager change, a pre-season hat-trick —
    the human fixes that pick and the solver optimises everything around it
    under the same budget and legality constraints.
    """
    force = force or set()
    n = len(players)
    value = np.array([float(p.get(value_key) or 0.0) for p in players])
    price = np.array([float(p["price"]) for p in players])
    positions = [p["position"] for p in players]
    clubs = [p["team_id"] for p in players]

    # Variables: x (in squad) for each player, then y (in starting XI).
    # Maximising, so the objective is negated for a minimising solver.
    objective = np.concatenate([np.zeros(n), -value])

    constraints = []

    def row(squad_part, xi_part):
        return np.concatenate([squad_part, xi_part])

    # Exactly fifteen in the squad, eleven on the pitch.
    constraints.append(LinearConstraint(row(np.ones(n), np.zeros(n)), 15, 15))
    constraints.append(LinearConstraint(row(np.zeros(n), np.ones(n)), XI_SIZE, XI_SIZE))

    # Budget applies to the whole squad, bench included.
    constraints.append(LinearConstraint(row(price, np.zeros(n)), -np.inf, BUDGET))

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

    for i, player in enumerate(players):
        if player["web_name"] in force:
            pin = np.zeros(2 * n)
            pin[i] = 1.0
            constraints.append(LinearConstraint(pin, 1, 1))

    # A player can only start if they are in the squad: y_i - x_i <= 0.
    starter_implies_squad = np.zeros((n, 2 * n))
    for i in range(n):
        starter_implies_squad[i, i] = -1.0
        starter_implies_squad[i, n + i] = 1.0
    constraints.append(LinearConstraint(starter_implies_squad, -np.inf, 0))

    result = milp(
        c=objective,
        constraints=constraints,
        integrality=np.ones(2 * n),
        bounds=Bounds(0, 1),
    )
    if not result.success:
        return None

    chosen = np.round(result.x).astype(int)
    squad = [players[i] for i in range(n) if chosen[i] == 1]
    xi_ids = {players[i]["id"] for i in range(n) if chosen[n + i] == 1}
    return {
        "squad": squad,
        "xi": [p for p in squad if p["id"] in xi_ids],
        "bench": [p for p in squad if p["id"] not in xi_ids],
        "xi_value": float(sum(p[value_key] or 0 for p in squad if p["id"] in xi_ids)),
        "cost": float(sum(p["price"] for p in squad)),
    }


def show(title: str, plan: dict, value_key: str, note: str = "") -> None:
    order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    if note:
        print(note + "\n")
    print(f"Cost £{plan['cost']:.1f}m / £{BUDGET:.0f}m    "
          f"Projected XI total: {plan['xi_value']:.1f} pts")

    print("\nStarting XI")
    for p in sorted(plan["xi"], key=lambda p: (order[p["position"]], -p[value_key])):
        print(f"  {p['position']}  {p['web_name']:<18} {p['team']:<4} "
              f"£{p['price']:>4.1f}  {p[value_key]:>6.2f}  "
              f"(last-season pts {p['total_points']}, {p['selected_by_percent']}% owned)")

    print("\nBench")
    for p in sorted(plan["bench"], key=lambda p: (order[p["position"]], -p[value_key])):
        print(f"  {p['position']}  {p['web_name']:<18} {p['team']:<4} "
              f"£{p['price']:>4.1f}  {p[value_key]:>6.2f}  "
              f"(last-season pts {p['total_points']})")

    captain = max(plan["xi"], key=lambda p: p[value_key])
    vice = sorted(plan["xi"], key=lambda p: -p[value_key])[1]
    print(f"\nCaptain: {captain['web_name']}   Vice: {vice['web_name']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:5001")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument(
        "--min-points",
        type=int,
        default=40,
        help="Last-season points floor, as a stand-in for the minutes signal the "
             "model lacks before a ball is kicked.",
    )
    parser.add_argument(
        "--force",
        default="",
        help="Comma-separated web_names to require in the squad. Research decides "
             "these; the solver optimises everything around them.",
    )
    parser.add_argument(
        "--also-allow",
        default="",
        help="Comma-separated web_names exempt from the points floor — for players "
             "research has verified as starters despite no Premier League record.",
    )
    parser.add_argument(
        "--exclude",
        default="",
        help="Comma-separated web_names to bar (bad fixtures, rotation risk).",
    )
    args = parser.parse_args()

    players = fetch(args.base, args.horizon)
    value_key = "xpts_horizon"

    available = [
        p for p in players
        if p.get("status") == "Available" and p.get(value_key) is not None
    ]
    print(f"{len(players)} players, {len(available)} available.")

    raw = solve(available, value_key)
    if raw:
        show(
            f"A. Raw ML optimum (horizon {args.horizon})",
            raw,
            value_key,
            "Availability filter only. This is what the model literally says.",
        )

    force = {s.strip() for s in args.force.split(",") if s.strip()}
    allow = {s.strip() for s in args.also_allow.split(",") if s.strip()}
    exclude = {s.strip() for s in args.exclude.split(",") if s.strip()}

    credible = [
        p for p in available
        if p["web_name"] not in exclude
        and (
            (p.get("total_points") or 0) >= args.min_points
            or p["web_name"] in allow
            or p["web_name"] in force
        )
    ]
    print(f"\n{len(credible)} players in the pool after research filters.")
    filtered = solve(credible, value_key, force=force)
    if filtered:
        missing = force - {p["web_name"] for p in filtered["squad"]}
        if missing:
            print(f"WARNING: forced but absent: {sorted(missing)}")
        show(
            f"B. Hybrid: research-pinned, ML-optimised (horizon {args.horizon})",
            filtered,
            value_key,
            f"Pinned by research: {', '.join(sorted(force)) or 'none'}.",
        )
    else:
        print("INFEASIBLE — the pinned set cannot fit the budget or the quotas.")


if __name__ == "__main__":
    main()
