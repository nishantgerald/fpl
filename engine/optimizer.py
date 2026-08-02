"""The transfer optimiser.

Replaces the GPT-4o-mini prompt that used to ask a language model to respect
FPL's rules in English. Here the rules are constraints, the objective is
expected points, and every plan that leaves this module has been re-verified
from scratch against :mod:`engine.rules` before it is serialised.

Determinism is a hard requirement, so every sort key ends in the player ids and
nothing consults a clock or a random source.

Search is staged rather than exhaustive, because 15 x 600 x 3 is not something
you solve inside a request:

  depth 0  the do-nothing baseline                            1 evaluation
  depth 1  every (out, in) pair, exhaustive over the pool   ~540
  depth 2  top-K single moves combined pairwise             ~276
  depth 3  best pairs extended by a third single            ~290

Depth 1 is exact. Depths 2-3 prune the *candidate moves* but evaluate every
surviving combination exactly, so a returned plan's score is never an estimate.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from . import money, rules

# Per-gameweek discount. Near-term projections are more trustworthy, and this
# stops the engine chasing a fixture swing eight weeks out.
DEFAULT_DECAY = 0.85

# Candidate pool size per position, taken twice: once by raw horizon xPts and
# once by xPts per million. The second list is what puts cheap enablers in the
# pool by construction rather than by luck.
POOL_PER_POSITION = 18

# How many single moves survive into the combination stages.
TOP_SINGLES_FOR_PAIRS = 24
TOP_PAIRS_FOR_TRIPLES = 12

MAX_PLANS_RETURNED = 5

# Internal keys stripped before a plan is serialised.
_INTERNAL_KEYS = ("_pairs", "_squad", "_tiebreak")


def optimise(
    squad: Sequence[Mapping],
    all_elements: Sequence[Mapping],
    projections: Mapping[int, Mapping],
    gameweeks: Sequence[int],
    bank: int,
    free_transfers: int,
    selling_prices: Mapping[int, int],
    quotas: Mapping[str, int],
    club_limit: int,
    max_transfers: int = 3,
    include_hits: bool = True,
    decay: float = DEFAULT_DECAY,
) -> dict:
    """Find the best legal transfer plans. Returns plans plus the hold baseline.

    All money is integer tenths. ``bank`` is what the manager actually has;
    ``selling_prices`` maps squad player ids to what they sell for.
    """
    squad_by_id = {int(p["id"]): p for p in squad}
    scores = _scores_by_gameweek(projections, gameweeks, decay)

    baseline = _squad_value(squad, scores)
    baseline_detail = _per_gameweek_detail(squad, scores)

    context = _Context(
        squad=list(squad),
        squad_by_id=squad_by_id,
        scores=scores,
        baseline=baseline,
        bank=bank,
        selling_prices=selling_prices,
        quotas=quotas,
        club_limit=club_limit,
    )

    pool = _candidate_pool(all_elements, projections, squad_by_id)

    singles = _search_singles(context, pool)
    top_singles = singles[:TOP_SINGLES_FOR_PAIRS]

    plans: list[dict] = []
    if max_transfers >= 1:
        plans.extend(singles)
    if max_transfers >= 2:
        pairs = _combine(context, top_singles, top_singles, depth=2)
        plans.extend(pairs)
        if max_transfers >= 3:
            plans.extend(
                _combine(context, pairs[:TOP_PAIRS_FOR_TRIPLES], top_singles, depth=3)
            )

    # Price the hits, drop anything that doesn't beat doing nothing.
    scored: list[dict] = []
    for plan in plans:
        n = plan["n_transfers"]
        if not include_hits and n > free_transfers:
            continue
        cost = rules.hit_cost(n, free_transfers)
        plan["hit_cost"] = cost
        plan["net_gain"] = round(plan["gross_gain"] - cost, 2)
        if plan["net_gain"] > 0:
            scored.append(plan)

    scored.sort(key=lambda p: (-p["net_gain"], p["n_transfers"], p["_tiebreak"]))

    # Final gate: re-verify from scratch. A plan that fails here is a bug, and a
    # bug must not reach the user dressed as advice.
    verified = [p for p in scored if _verify(p, context)]
    top = verified[:MAX_PLANS_RETURNED]

    # Guarantee: a squad member who is unavailable (projects zero because
    # they're injured/suspended, not merely out of form) is the one hold
    # that is *always* wrong, even when a healthy player's upgrade posts a
    # bigger raw gain. Force their best legal single swap into the returned
    # set if pure gain-ranking would otherwise have dropped it.
    unavailable_ids = {
        pid
        for pid, proj in projections.items()
        if int(pid) in squad_by_id and proj.get("availability", 1.0) <= 0.0
    }
    covered = {int(t["out"]["id"]) for p in top for t in p["transfers"]}
    for pid in unavailable_ids - covered:
        best_for_player = next(
            (
                p
                for p in verified
                if any(int(t["out"]["id"]) == pid for t in p["transfers"])
            ),
            None,
        )
        if best_for_player is not None and best_for_player not in top:
            top = top[: MAX_PLANS_RETURNED - 1] + [best_for_player]

    top.sort(key=lambda p: (-p["net_gain"], p["n_transfers"], p["_tiebreak"]))

    for rank, plan in enumerate(top, start=1):
        plan["rank"] = rank
        plan["per_gameweek"] = _plan_gameweek_detail(plan["_squad"], scores, baseline_detail)
        plan["reasons"] = _reasons(plan, projections)

    best_rejected = _best_rejected(scored, singles, free_transfers, bool(top))

    for plan in top:
        for key in _INTERNAL_KEYS:
            plan.pop(key, None)

    return {
        "plans": top,
        "hold": _hold(baseline, baseline_detail, best_rejected, free_transfers),
        "baseline_xpts": round(baseline, 2),
    }


class _Context:
    """Everything the search needs that doesn't change between evaluations."""

    __slots__ = (
        "squad", "squad_by_id", "scores", "baseline", "bank",
        "selling_prices", "quotas", "club_limit",
    )

    def __init__(self, squad, squad_by_id, scores, baseline, bank,
                 selling_prices, quotas, club_limit):
        self.squad = squad
        self.squad_by_id = squad_by_id
        self.scores = scores
        self.baseline = baseline
        self.bank = bank
        self.selling_prices = selling_prices
        self.quotas = quotas
        self.club_limit = club_limit


# ---------------------------------------------------------------- scoring


def _scores_by_gameweek(
    projections: Mapping[int, Mapping],
    gameweeks: Sequence[int],
    decay: float,
) -> list[tuple[int, float, dict[int, float]]]:
    """``[(gameweek, weight, {player_id: xpts}), ...]`` aligned to the horizon."""
    by_gw: dict[int, dict[int, float]] = {gw: {} for gw in gameweeks}
    for pid, proj in projections.items():
        for entry in proj.get("per_gameweek", ()):
            gw = entry["gameweek"]
            if gw in by_gw:
                by_gw[gw][pid] = float(entry["xpts"])
    return [(gw, decay ** offset, by_gw[gw]) for offset, gw in enumerate(gameweeks)]


def _squad_value(
    squad: Sequence[Mapping],
    scores: Sequence[tuple[int, float, Mapping[int, float]]],
) -> float:
    """Discounted sum of best-XI-plus-captain across the horizon."""
    total = 0.0
    for _gw, weight, row in scores:
        _xi, _shape, points, _captain = rules.best_xi_with_captain(squad, row)
        total += weight * points
    return total


def _per_gameweek_detail(
    squad: Sequence[Mapping],
    scores: Sequence[tuple[int, float, Mapping[int, float]]],
) -> list[dict]:
    detail = []
    for gw, _weight, row in scores:
        _xi, shape, points, captain = rules.best_xi_with_captain(squad, row)
        detail.append(
            {
                "gameweek": gw,
                "xpts": round(points, 2),
                "formation": rules.formation_str(shape),
                "captain_id": captain,
            }
        )
    return detail


def _plan_gameweek_detail(
    new_squad: Sequence[Mapping],
    scores,
    baseline_detail: Sequence[Mapping],
) -> list[dict]:
    after = _per_gameweek_detail(new_squad, scores)
    return [
        {
            "gameweek": a["gameweek"],
            "before": b["xpts"],
            "after": a["xpts"],
            "delta": round(a["xpts"] - b["xpts"], 2),
            "formation": a["formation"],
            "captain_id": a["captain_id"],
        }
        for a, b in zip(after, baseline_detail)
    ]


# ---------------------------------------------------------------- candidates


def _candidate_pool(
    all_elements: Sequence[Mapping],
    projections: Mapping[int, Mapping],
    squad_by_id: Mapping[int, Mapping],
) -> dict[str, list[Mapping]]:
    """Transfer-in candidates by position.

    Two lists are unioned: the best by horizon xPts, and the best by xPts per
    million. Without the second list the pool is all premiums and the optimiser
    can never find the cheap enabler that makes a premium affordable — exactly
    what the old FCPS-truncated top-70 pool got wrong.
    """
    by_position: dict[str, list[Mapping]] = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
    for element in all_elements:
        pid = int(element["id"])
        if pid in squad_by_id or pid not in projections:
            continue
        if not rules.is_available(element):
            continue
        position = rules.position_of(element)
        if position in by_position:
            by_position[position].append(element)

    pool: dict[str, list[Mapping]] = {}
    for position, elements in by_position.items():
        by_points = sorted(
            elements,
            key=lambda e: (-projections[int(e["id"])]["horizon_xpts"], int(e["id"])),
        )[:POOL_PER_POSITION]
        by_value = sorted(
            elements,
            key=lambda e: (-projections[int(e["id"])]["xpts_per_million"], int(e["id"])),
        )[:POOL_PER_POSITION]

        seen: set[int] = set()
        merged: list[Mapping] = []
        for element in by_points + by_value:
            pid = int(element["id"])
            if pid not in seen:
                seen.add(pid)
                merged.append(element)
        merged.sort(key=lambda e: int(e["id"]))
        pool[position] = merged
    return pool


# ---------------------------------------------------------------- search


def _search_singles(context: _Context, pool: Mapping[str, Sequence[Mapping]]) -> list[dict]:
    """Exhaustive over (every squad player) x (every same-position candidate)."""
    plans: list[dict] = []
    for out_player in context.squad:
        position = rules.position_of(out_player)
        for in_player in pool.get(position, ()):
            plan = _evaluate([(out_player, in_player)], context)
            if plan:
                plans.append(plan)
    plans.sort(key=lambda p: (-p["gross_gain"], p["_tiebreak"]))
    return plans


def _combine(
    context: _Context,
    seeds: Sequence[Mapping],
    singles: Sequence[Mapping],
    depth: int,
) -> list[dict]:
    """Extend each seed by one more single move and re-evaluate exactly."""
    plans: list[dict] = []
    seen: set[tuple] = set()

    for seed in seeds:
        seed_pairs = seed["_pairs"]
        seed_outs = {int(o["id"]) for o, _ in seed_pairs}
        seed_ins = {int(i["id"]) for _, i in seed_pairs}

        for single in singles:
            out_player, in_player = single["_pairs"][0]
            out_id, in_id = int(out_player["id"]), int(in_player["id"])
            if out_id in seed_outs or in_id in seed_ins:
                continue

            key = (
                tuple(sorted(seed_outs | {out_id})),
                tuple(sorted(seed_ins | {in_id})),
            )
            if key in seen:
                continue
            seen.add(key)

            plan = _evaluate(list(seed_pairs) + [(out_player, in_player)], context)
            if plan and plan["n_transfers"] == depth:
                plans.append(plan)

    plans.sort(key=lambda p: (-p["gross_gain"], p["_tiebreak"]))
    return plans


def _evaluate(pairs: Sequence[tuple[Mapping, Mapping]], context: _Context) -> dict | None:
    """Build, check and score one candidate plan. ``None`` if illegal.

    ``pairs`` is an explicit ``(out, in)`` list, so a transfer's two halves stay
    bound together and the position-preserving check in :func:`_verify` means
    what it says.
    """
    outs = [o for o, _ in pairs]
    ins = [i for _, i in pairs]
    out_ids = {int(p["id"]) for p in outs}
    in_ids = {int(p["id"]) for p in ins}
    if len(out_ids) != len(outs) or len(in_ids) != len(ins) or (out_ids & in_ids):
        return None

    spend = money.transfer_spend(ins, outs, context.selling_prices)
    if not money.is_affordable(spend, context.bank):
        return None

    new_squad = [p for p in context.squad if int(p["id"]) not in out_ids] + list(ins)
    legality = rules.check_squad(new_squad, context.quotas, context.club_limit)
    if not legality["all_ok"]:
        return None

    gross = _squad_value(new_squad, context.scores) - context.baseline

    ordered = sorted(pairs, key=lambda pair: int(pair[0]["id"]))
    return {
        "transfers": [
            {
                "out": _player_ref(o, context.selling_prices.get(int(o["id"]))),
                "in": _player_ref(i, None),
            }
            for o, i in ordered
        ],
        "n_transfers": len(pairs),
        "gross_gain": round(gross, 2),
        "spend": spend,
        "bank_after": context.bank - spend,
        "legality": legality,
        "_pairs": ordered,
        "_squad": new_squad,
        "_tiebreak": tuple(sorted(out_ids)) + tuple(sorted(in_ids)),
    }


def _player_ref(player: Mapping, selling_price: int | None) -> dict:
    full_name = f"{player.get('first_name', '')} {player.get('second_name', '')}".strip()
    ref = {
        "id": int(player["id"]),
        "name": full_name or str(player.get("web_name", "")),
        "web_name": str(player.get("web_name", "") or full_name),
        "position": rules.position_of(player),
        "team_id": int(player.get("team", 0)),
        "now_cost": int(player.get("now_cost", 0)),
        "status": player.get("status", "a"),
    }
    if selling_price is not None:
        ref["selling_price"] = int(selling_price)
    return ref


# ---------------------------------------------------------------- verification


def _verify(plan: Mapping, context: _Context) -> bool:
    """Independent re-check of a plan, from scratch.

    This deliberately does not trust anything computed during the search. It is
    the last line of defence against emitting a recommendation the user cannot
    legally execute — which is worse than emitting nothing, because it costs
    them the time to discover it's impossible.
    """
    new_squad = plan.get("_squad")
    if not new_squad:
        return False

    for transfer in plan["transfers"]:
        out_id, in_id = transfer["out"]["id"], transfer["in"]["id"]
        if out_id not in context.squad_by_id:
            return False                                  # selling someone we don't own
        if in_id in context.squad_by_id:
            return False                                  # buying someone we already own
        if transfer["out"]["position"] != transfer["in"]["position"]:
            return False                                  # quotas make this illegal
        if str(transfer["in"].get("status", "a")) in rules.UNAVAILABLE_STATUSES:
            return False                                  # never buy an unavailable player

    if not money.is_affordable(plan["spend"], context.bank):
        return False
    if plan["bank_after"] != context.bank - plan["spend"]:
        return False
    if plan["hit_cost"] < 0:
        return False

    return rules.check_squad(new_squad, context.quotas, context.club_limit)["all_ok"]


# ---------------------------------------------------------------- narration


def _reasons(plan: Mapping, projections: Mapping[int, Mapping]) -> list[str]:
    """Short factual reasons, generated from computed numbers only.

    Deliberately not LLM output. The optional narrative layer adds a separate
    field and never edits these.
    """
    reasons: list[str] = []
    horizon_gws = len(plan.get("per_gameweek", ())) or 1
    reasons.append(
        f"Gains {plan['net_gain']:.1f} pts over {horizon_gws} GW"
        f"{'s' if horizon_gws != 1 else ''}"
        + (f" after a -{plan['hit_cost']} hit" if plan["hit_cost"] else "")
    )

    for transfer in plan["transfers"]:
        out_proj = projections.get(transfer["out"]["id"], {})
        in_proj = projections.get(transfer["in"]["id"], {})
        out_name = transfer["out"]["web_name"]
        in_name = transfer["in"]["web_name"]

        if out_proj.get("availability", 1.0) < 1.0:
            reasons.append(
                f"{out_name} is flagged "
                f"({out_proj['availability'] * 100:.0f}% chance of playing)"
            )
        if out_proj.get("minutes_risk") == "high":
            reasons.append(
                f"{out_name} is a rotation risk "
                f"({out_proj.get('p_start', 0) * 100:.0f}% start rate)"
            )
        if in_proj and out_proj:
            delta = in_proj.get("horizon_xpts", 0) - out_proj.get("horizon_xpts", 0)
            if delta > 0:
                reasons.append(
                    f"{in_name} projects {delta:.1f} pts more than {out_name} "
                    f"over the horizon"
                )

    if plan["spend"] < 0:
        reasons.append(
            f"Frees up {money.tenths_to_str(-plan['spend'])}m for a future move"
        )

    return reasons[:4]


def _best_rejected(
    scored: Sequence[Mapping],
    singles: Sequence[Mapping],
    free_transfers: int,
    had_plans: bool,
) -> dict | None:
    """The best move we *didn't* recommend, so the hold card can cite it."""
    if had_plans:
        return None
    source = scored[0] if scored else (
        max(singles, key=lambda p: p["gross_gain"]) if singles else None
    )
    if not source:
        return None
    n = source["n_transfers"]
    cost = source.get("hit_cost", rules.hit_cost(n, free_transfers))
    return {
        "transfers": source["transfers"],
        "gross_gain": source["gross_gain"],
        "hit_cost": cost,
        "net_gain": round(source["gross_gain"] - cost, 2),
    }


def _hold(
    baseline: float,
    detail: Sequence[Mapping],
    best_rejected: Mapping | None,
    free_transfers: int,
) -> dict:
    """The do-nothing option, stated as a recommendation rather than a failure.

    Most tools only ever suggest moves, because suggesting moves feels like
    value. Holding is frequently correct, and saying so is what earns trust.
    """
    if best_rejected:
        out_name = best_rejected["transfers"][0]["out"]["web_name"]
        in_name = best_rejected["transfers"][0]["in"]["web_name"]
        banked = min(5, free_transfers + 1)
        reason = (
            f"The best available move ({out_name} to {in_name}) gains "
            f"{best_rejected['gross_gain']:.1f} pts and costs "
            f"{best_rejected['hit_cost']}. Bank the transfer — you'd have "
            f"{banked} next gameweek."
        )
    else:
        reason = (
            "No transfer improves your projected points over this horizon. "
            "Bank the transfer."
        )

    return {
        "baseline_xpts": round(baseline, 2),
        "per_gameweek": list(detail),
        "reason": reason,
        "best_rejected": best_rejected,
    }
