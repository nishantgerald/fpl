"""Price-change prediction.

Every ``bootstrap-static`` fetch already carries ``transfers_in_event``,
``transfers_out_event``, ``cost_change_event`` and ``total_players``. The old
app downloaded all four and used none of them.

FPL's actual price algorithm is unpublished. This is the well-established public
approximation — net transfers as a share of the total player base, against a
threshold that scales with ownership — and every response says so. A confidently
wrong prediction costs more trust than a hedged right one earns, so the output is
a pressure value plus a categorical verdict, never a false-precision claim.

The differentiating part isn't the global riser list (that's a commodity); it's
the per-squad ``implication`` string, which turns data into a decision.

Pure: bootstrap elements in, verdicts out.
"""

from __future__ import annotations

from typing import Mapping, Sequence

# Base net-transfer share needed to move a price, before ownership scaling.
BASE_THRESHOLD = 0.0075

# Heavily-owned players need proportionally more net transfers to move, and
# falls trigger more easily than rises.
OWNERSHIP_SCALING = 0.045
FALL_THRESHOLD_RATIO = 0.72

VERDICTS = (
    (1.0, "rising_tonight"),
    (0.75, "rise_likely"),
    (0.4, "watch"),
)

ACCURACY_NOTE = (
    "FPL's price algorithm is unpublished. This is a net-transfer heuristic — "
    "treat it as directional, not as a guarantee."
)


def _threshold(ownership: float, falling: bool) -> float:
    base = BASE_THRESHOLD * (1.0 + OWNERSHIP_SCALING * max(0.0, ownership))
    return base * FALL_THRESHOLD_RATIO if falling else base


def predict_player(element: Mapping, total_players: int) -> dict:
    """Rise/fall pressure for one player."""
    if total_players <= 0:
        total_players = 1

    transfers_in = int(element.get("transfers_in_event", 0) or 0)
    transfers_out = int(element.get("transfers_out_event", 0) or 0)
    net = transfers_in - transfers_out
    ratio = net / total_players
    ownership = _float(element.get("selected_by_percent"))
    already_moved = int(element.get("cost_change_event", 0) or 0)

    if net >= 0:
        pressure = ratio / _threshold(ownership, falling=False)
        direction = "rise"
    else:
        pressure = -ratio / _threshold(ownership, falling=True)
        direction = "fall"

    pressure = max(0.0, pressure)
    verdict = _verdict(pressure, direction, already_moved)

    return {
        "id": int(element["id"]),
        "web_name": element.get("web_name", ""),
        "now_cost": int(element.get("now_cost", 0)),
        "net_transfers": net,
        "ownership": round(ownership, 2),
        "pressure": round(min(pressure, 3.0), 3),
        "direction": direction,
        "verdict": verdict,
        "cost_change_event": already_moved,
    }


def _verdict(pressure: float, direction: str, already_moved: int) -> str:
    if already_moved != 0:
        # FPL caps a player at one change per day; if they've already moved,
        # tonight's prediction is meaningless.
        return "already_changed"
    for level, name in VERDICTS:
        if pressure >= level:
            if direction == "fall":
                return {
                    "rising_tonight": "falling_tonight",
                    "rise_likely": "fall_likely",
                    "watch": "watch",
                }[name]
            return name
    return "stable"


def predict_all(
    elements: Sequence[Mapping],
    total_players: int,
    squad_ids: Sequence[int] | None = None,
    top_n: int = 20,
) -> dict:
    """Global risers/fallers plus, if a squad is supplied, the personal view."""
    predictions = [predict_player(e, total_players) for e in elements]
    by_id = {p["id"]: p for p in predictions}

    risers = sorted(
        (p for p in predictions if p["direction"] == "rise" and p["pressure"] >= 0.4),
        key=lambda p: (-p["pressure"], p["id"]),
    )[:top_n]
    fallers = sorted(
        (p for p in predictions if p["direction"] == "fall" and p["pressure"] >= 0.4),
        key=lambda p: (-p["pressure"], p["id"]),
    )[:top_n]

    result = {
        "risers": risers,
        "fallers": fallers,
        "model": "heuristic",
        "accuracy_note": ACCURACY_NOTE,
    }

    if squad_ids is not None:
        squad_view = []
        delta = 0
        for pid in squad_ids:
            prediction = by_id.get(int(pid))
            if not prediction:
                continue
            entry = dict(prediction)
            entry["owned"] = True
            entry["implication"] = _implication(entry, owned=True)
            if entry["verdict"] == "rising_tonight":
                delta += 1
            elif entry["verdict"] == "falling_tonight":
                delta -= 1
            squad_view.append(entry)

        squad_view.sort(key=lambda e: (-e["pressure"], e["id"]))
        result["your_squad"] = squad_view
        result["squad_value_delta_tonight"] = delta

    for prediction in risers + fallers:
        prediction.setdefault("implication", _implication(prediction, owned=False))

    return result


def _implication(prediction: Mapping, owned: bool) -> str:
    """The sentence that turns a number into a decision."""
    verdict = prediction["verdict"]
    name = prediction["web_name"]

    if verdict == "already_changed":
        moved = prediction["cost_change_event"]
        direction = "rose" if moved > 0 else "fell"
        return f"{name} already {direction} today — no further change tonight."

    if owned:
        if verdict == "rising_tonight":
            return "Already yours — you gain £0.1m tonight."
        if verdict == "rise_likely":
            return "Likely to rise tonight, which adds to your team value."
        if verdict == "falling_tonight":
            return "Selling after tonight costs you £0.1m. Move before the change."
        if verdict == "fall_likely":
            return "At risk of falling. If you're selling this week, sell now."
        if verdict == "watch":
            return "Drifting — worth watching over the next day."
        return "No change expected tonight."

    if verdict == "rising_tonight":
        return f"{name} rises tonight — buy now or pay £0.1m more."
    if verdict == "rise_likely":
        return f"{name} is close to a rise."
    if verdict == "falling_tonight":
        return f"{name} falls tonight — wait a day if you're buying."
    if verdict == "fall_likely":
        return f"{name} is close to a fall."
    return f"{name} is drifting."


def _float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
