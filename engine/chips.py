"""When to play each chip, given this squad and these fixtures.

A chip is worth whatever it adds *over doing nothing*, and that number is
computable rather than a matter of taste. Every recommendation here is a
points delta against the same squad without the chip, so the advice can be
checked instead of trusted.

The four chips answer four different questions:

**Triple Captain** — one extra captain's worth of points. Worth the most on a
double gameweek, when the armband applies twice, and otherwise on the single
best fixture a premium has all season.

**Bench Boost** — the four bench players' points, which is normally zero. Its
value is entirely a property of the bench, so it rewards planning a week ahead
far more than it rewards the gameweek itself.

**Free Hit** — the gap between the best possible eleven that week and the one
this squad can field. That gap is small in a normal gameweek and enormous in a
blank, which is the whole case for holding it.

**Wildcard** — the gap between this squad and the best squad over the horizon,
less the transfers that would fix it anyway. It is the only chip whose value
grows the longer you wait to use it, and the only one with a deadline that
punishes waiting.

The rules are read from FPL's own bootstrap rather than hardcoded: the chip
list carries each chip's legal window, and inventing our own copy of that is
how an app ends up recommending a chip a fortnight after it expired.
"""

from __future__ import annotations

from typing import Mapping, Sequence

CHIP_LABELS = {
    "3xc": "Triple Captain",
    "bboost": "Bench Boost",
    "freehit": "Free Hit",
    "wildcard": "Wildcard",
}

# Below this, a chip is not worth burning. A Bench Boost returning four points
# is not a bad play so much as a wasted one — the chip is gone and could have
# been a double gameweek.
MIN_WORTHWHILE = {
    "3xc": 6.0,
    "bboost": 12.0,
    "freehit": 15.0,
    "wildcard": 12.0,
}

# Chips left this late in a half are use-it-or-lose-it.
URGENT_WINDOW = 4


def windows(chips: Sequence[Mapping]) -> list[dict]:
    """Each chip's legal gameweek range, straight from the bootstrap.

    Two of each exist per season — one per half — and the halves have different
    windows. Hardcoding that is how an app recommends a chip that expired a
    fortnight ago.
    """
    out = []
    for chip in chips or []:
        name = str(chip.get("name", ""))
        if name not in CHIP_LABELS:
            continue
        out.append(
            {
                "name": name,
                "label": CHIP_LABELS[name],
                "start": int(chip.get("start_event") or 1),
                "stop": int(chip.get("stop_event") or 38),
                "kind": chip.get("chip_type", "team"),
            }
        )
    return sorted(out, key=lambda c: (c["start"], c["name"]))


def available(
    all_windows: Sequence[Mapping],
    used: Sequence[Mapping],
    gameweek: int,
) -> list[dict]:
    """Chips this manager can still play, in the half they are currently in.

    ``used`` is the entry history's chip list. A chip played in the first half
    does not consume the second-half copy, which is the part managers most
    often get wrong.
    """
    spent = {
        (str(c.get("name", "")), _half(int(c.get("event") or 0)))
        for c in used or []
    }
    out = []
    for window in all_windows:
        if gameweek > window["stop"]:
            continue
        if (window["name"], _half(window["start"])) in spent:
            continue
        out.append(
            {
                **window,
                "playable_now": window["start"] <= gameweek <= window["stop"],
                "expires_in": window["stop"] - gameweek,
            }
        )
    return out


def _half(gameweek: int) -> int:
    return 1 if gameweek < 20 else 2


def triple_captain_value(
    squad_projections: Sequence[Mapping], gameweek_index: int = 0
) -> tuple[float, str | None]:
    """Extra points from tripling rather than doubling the armband.

    The chip adds exactly one more captain's score, so its value is the best
    single projection in the squad — not twice it.
    """
    best, name = 0.0, None
    for player in squad_projections:
        per_gw = player.get("per_gameweek") or []
        if gameweek_index >= len(per_gw):
            continue
        points = float(per_gw[gameweek_index].get("xpts") or 0.0)
        if points > best:
            best, name = points, player.get("web_name")
    return round(best, 1), name


def bench_boost_value(
    bench_projections: Sequence[Mapping], gameweek_index: int = 0
) -> float:
    """Points the bench would score, which is otherwise nothing."""
    total = 0.0
    for player in bench_projections:
        per_gw = player.get("per_gameweek") or []
        if gameweek_index < len(per_gw):
            total += float(per_gw[gameweek_index].get("xpts") or 0.0)
    return round(total, 1)


def free_hit_value(
    xi_points: float, best_possible_xi: float
) -> float:
    """What a one-week unlimited squad would add over the eleven available."""
    return round(max(0.0, best_possible_xi - xi_points), 1)


def wildcard_value(
    squad_horizon: float, optimal_horizon: float, free_transfers: int
) -> float:
    """Gap to the best squad, less what ordinary transfers would recover.

    A wildcard is only worth its own use if the squad is further from optimal
    than a couple of weeks of free transfers could fix. Charging that against
    it is the difference between advice and salesmanship — roughly four points
    per transfer is what the optimiser typically finds.
    """
    recoverable = free_transfers * 4.0
    return round(max(0.0, optimal_horizon - squad_horizon - recoverable), 1)


def recommend(
    chip: Mapping,
    value: float,
    gameweek: int,
    detail: str = "",
) -> dict:
    """Turn a value into a verdict, with the reason attached.

    Three states, because "not yet" and "no" are different advice: one means
    hold and watch, the other means this chip will not pay here.
    """
    threshold = MIN_WORTHWHILE.get(chip["name"], 10.0)
    expires_in = chip.get("expires_in", 99)
    urgent = expires_in <= URGENT_WINDOW

    if not chip.get("playable_now"):
        verdict, action = "locked", (
            f"Unlocks in GW{chip['start']}."
        )
    elif value >= threshold:
        verdict, action = "play", f"Worth about {value:.0f} points here."
    elif urgent:
        verdict, action = "expiring", (
            f"Only {expires_in} gameweek{'s' if expires_in != 1 else ''} left "
            f"to use it. Best available now is ~{value:.0f} points — take it "
            "rather than lose it."
        )
    else:
        verdict, action = "hold", (
            f"Only worth ~{value:.0f} points now, against ~{threshold:.0f} "
            "for a good week. Hold."
        )

    return {
        "name": chip["name"],
        "label": chip["label"],
        "verdict": verdict,
        "value": value,
        "expires_gameweek": chip["stop"],
        "expires_in": expires_in,
        "urgent": urgent and verdict != "locked",
        "action": action,
        "detail": detail,
    }
