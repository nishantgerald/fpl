"""Advice that has to earn its place on the screen.

The first version of the actions engine had one failure mode, and it produced
every complaint made about it: **it emitted an item whenever a condition was
true, rather than when acting on it would gain points.**

Three consequences, all of them real:

*Sell advice with no buyer.* A club's fixtures worsening is a fact about the
club. Whether to sell the midfielder you own there depends on whether anyone
you could afford would actually score more — and for a genuinely elite player
the answer is usually no. Telling someone to sell Fernandes because Manchester
United's run hardens is advice that ignores that he outscores every midfielder
they could buy with the money. The fixture swing is real and the conclusion
does not follow.

*Buy targets with no floor.* "The best player at a club whose run eases" is not
the same as "a player worth owning". At a weak club the best available player
can project two points a gameweek, and the old code would recommend him with a
straight face, quoting the number that disqualified him.

*A shopping list with no comparison.* A transfer is a swap. A recommendation
that names an incoming player without naming what he replaces leaves the entire
decision undone, and cannot say whether it is worth doing at all.

So every function here answers the same question: **what does the manager gain,
and against what?** An item that cannot answer is not returned. The bars are
constants at the top rather than buried, because their whole purpose is to be
argued with.

The buy shortlists borrow the shape that worked for squad building. One
"best players" list is a single opinion; four lists answering four different
questions — who is in form, whose fixtures turn, who nobody else owns, who is
cheap for what he returns — let a manager pick the question that matches their
season. Same engine, different constraint, same as the squad strategies.

Pure functions: elements, projections and fixtures in, advice out. No network,
no clock.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from .rules import POSITIONS

# ------------------------------------------------------------------- the bars

#: A player has to be projected at least this many points per gameweek before
#: he is worth recommending at all. Below it the recommendation quotes a number
#: that argues against itself — the old engine suggested a forward on ten points
#: across five gameweeks, which is two a week, which is a bench player.
MIN_POINTS_PER_GAMEWEEK = 3.2

#: A transfer has to gain at least this many points across the horizon to be
#: worth a free transfer. Anything under it is inside the model's own error, and
#: spending a transfer on noise costs the flexibility to make a real one later.
MIN_TRANSFER_GAIN = 2.5

#: How much more than the outgoing player's price we assume is available.
#: Without the bank balance this is a guess, so it is a small one and the advice
#: says the price rather than implying the transfer is definitely affordable.
ASSUMED_BANK = 0.5

#: A player's own projection has to fall by this much per gameweek before a
#: club-level fixture swing is worth acting on for him specifically. A swing at
#: a club whose player is barely affected is trivia about the club.
MATERIAL_DROP_PER_GAMEWEEK = 0.45

#: Ownership at or below this counts as a differential. Above it, owning the
#: player is the crowd's position rather than a bet against it.
DIFFERENTIAL_OWNERSHIP = 8.0

#: Form is points per game over the last thirty days. This is roughly the level
#: at which a player is outscoring what a mid-price starter returns.
IN_FORM = 5.0


def _price(element: Mapping) -> float:
    return int(element.get("now_cost", 0)) / 10.0


def _position(element: Mapping) -> str:
    return POSITIONS.get(int(element.get("element_type", 0)), "")


def _horizon(projections: Mapping, player_id: int) -> float:
    return float((projections.get(int(player_id)) or {}).get("horizon_xpts") or 0.0)


def _per_gameweek(projections: Mapping, player_id: int) -> list[Mapping]:
    return list((projections.get(int(player_id)) or {}).get("per_gameweek") or [])


def _float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------- what the calendar says


def schedule_shape(fixtures: Sequence[Mapping], from_gameweek: int) -> dict:
    """Doubles and blanks that are actually on the fixture list.

    Chip advice kept asserting that Free Hit is "biggest in a blank gameweek"
    and Bench Boost belongs on a double, which is true and useless: it never
    said whether either was scheduled. A manager reading it cannot tell the
    difference between "hold this for the double" and "there is no double".

    The distinction that matters, and the reason this reads the raw fixture list
    rather than the ticker's window:

    * a **double** is a team with two fixtures in one gameweek — unambiguous;
    * a **blank** is a team with no fixture in a gameweek that other teams are
      playing — which is only meaningful once that gameweek is scheduled at all;
    * an **unscheduled** fixture carries ``event: null``, and is neither. Early
      in a season every team plays every gameweek and nothing here fires, which
      is the correct answer rather than a gap in the data.

    Blanks and doubles are created when cup rounds displace league fixtures, so
    for most of the first half the honest answer is "none yet".
    """
    by_gameweek: dict[int, dict[int, int]] = {}
    unscheduled = 0

    for fixture in fixtures or []:
        event = fixture.get("event")
        if event is None:
            unscheduled += 1
            continue
        gameweek = int(event)
        if gameweek < from_gameweek:
            continue
        slot = by_gameweek.setdefault(gameweek, {})
        for side in ("team_h", "team_a"):
            team = fixture.get(side)
            if team is None:
                continue
            slot[int(team)] = slot.get(int(team), 0) + 1

    doubles: list[dict] = []
    blanks: list[dict] = []
    # Teams are only known to blank relative to the teams that do play, so the
    # roster is taken from the fixture list itself rather than assumed to be 20.
    all_teams = {team for slot in by_gameweek.values() for team in slot}

    for gameweek in sorted(by_gameweek):
        slot = by_gameweek[gameweek]
        doubled = sorted(team for team, count in slot.items() if count > 1)
        if doubled:
            doubles.append({"gameweek": gameweek, "teams": doubled})
        # A gameweek nobody plays is an unscheduled gameweek, not 20 blanks.
        if slot:
            missing = sorted(all_teams - set(slot))
            if missing:
                blanks.append({"gameweek": gameweek, "teams": missing})

    return {
        "doubles": doubles,
        "blanks": blanks,
        "unscheduled_fixtures": unscheduled,
        "next_double": doubles[0]["gameweek"] if doubles else None,
        "next_blank": blanks[0]["gameweek"] if blanks else None,
    }


def chip_schedule_note(chip_name: str, schedule: Mapping, teams: Mapping) -> str:
    """One sentence about the calendar, or an honest admission there is none.

    Silence would be read as "no double", and a generic line about doubles being
    good would be read as "there is one". Both are worse than saying which.
    """
    def _names(team_ids: Sequence[int], limit: int = 4) -> str:
        shorts = [
            str((teams.get(int(t)) or {}).get("short_name", t)) for t in team_ids
        ]
        if len(shorts) > limit:
            return f"{', '.join(shorts[:limit])} and {len(shorts) - limit} more"
        return ", ".join(shorts)

    if chip_name in ("3xc", "bboost"):
        double = (schedule.get("doubles") or [None])[0]
        if not double:
            return (
                "No double gameweek is scheduled yet — they appear when cup "
                "rounds displace league fixtures, usually in the second half."
            )
        return (
            f"GW{double['gameweek']} is a double for "
            f"{_names(double['teams'])}."
            + (
                " The armband applies to both fixtures."
                if chip_name == "3xc"
                else " Four bench players with two fixtures each is the ceiling."
            )
        )

    if chip_name == "freehit":
        blank = (schedule.get("blanks") or [None])[0]
        if not blank:
            return (
                "No blank gameweek is scheduled yet. Free Hit is worth most in "
                "one, so there is rarely a reason to spend it before then."
            )
        return (
            f"GW{blank['gameweek']} is blank for {len(blank['teams'])} clubs "
            f"({_names(blank['teams'])}) — the gameweek this chip exists for."
        )

    return ""


# ------------------------------------------------------------------ selling


def sell_advice(
    squad: Sequence[Mapping],
    elements: Mapping[int, Mapping],
    projections: Mapping,
    swings: Mapping[int, Mapping],
    owned_ids: set[int],
    horizon: int,
    teams: Mapping[int, Mapping] | None = None,
) -> list[dict]:
    """Sell recommendations that name the replacement and the points gained.

    A worsening run is the *trigger*, not the argument. Two further things have
    to be true before selling is the right move, and the old engine checked
    neither:

    1. **The player himself is affected.** A premium on a hard run still
       outscores a mid-price player on an easy one, because the fixture is one
       term in his projection and not the largest.
    2. **Someone better is affordable.** If nothing you could buy with the money
       beats him, the correct action is to hold, and the fixture run is
       something to sit through rather than something to trade.

    Point 2 is why the elite-player complaint was correct: the item was raised
    against a player no affordable transfer improves on, so the advice had no
    achievable version.
    """
    items: list[dict] = []

    for player in squad:
        player_id = int(player.get("id", 0))
        swing = swings.get(int(player.get("team", 0)))
        if not swing or swing.get("direction") != "worsening":
            continue

        per_gameweek = _per_gameweek(projections, player_id)
        if len(per_gameweek) < 4:
            continue

        # Split the horizon at the swing's own pivot rather than the middle: the
        # ticker already decided where the run changes character.
        pivot = int(swing.get("from_gameweek") or 0)
        before = [g for g in per_gameweek if int(g.get("gameweek", 0)) < pivot]
        after = [g for g in per_gameweek if int(g.get("gameweek", 0)) >= pivot]
        if not before or not after:
            split = len(per_gameweek) // 2
            before, after = per_gameweek[:split], per_gameweek[split:]

        near = sum(_float(g.get("xpts")) for g in before) / len(before)
        far = sum(_float(g.get("xpts")) for g in after) / len(after)
        drop = near - far
        if drop < MATERIAL_DROP_PER_GAMEWEEK:
            continue

        name = str(player.get("web_name") or "")
        replacement = best_replacement(
            player, elements, projections, owned_ids, horizon, teams
        )
        if not replacement:
            # The trigger fired and the conclusion does not follow. Saying
            # nothing is the honest outcome; the alternative is advice whose
            # only achievable version loses points.
            continue

        items.append(
            {
                "priority": 3,
                "kind": "fixtures_worsening",
                "player": name,
                "headline": f"{name} → {replacement['player']} gains "
                f"{replacement['gain']:.0f} pts",
                "detail": (
                    f"{swing.get('message', '')} {name} drops from "
                    f"{near:.1f} to {far:.1f} points a gameweek across it, and "
                    f"{replacement['player']} (£{replacement['price']:.1f}m, "
                    f"{replacement['team']}) projects "
                    f"{replacement['horizon_xpts']:.0f} over the same "
                    f"{horizon} gameweeks against his "
                    f"{replacement['outgoing_xpts']:.0f}."
                ).strip(),
                "action": (
                    f"Sell {name} for {replacement['player']} before "
                    f"GW{pivot} — about {replacement['gain']:.0f} points over "
                    f"{horizon} gameweeks."
                    + (
                        f" Needs £{replacement['extra_cost']:.1f}m from the bank."
                        if replacement["extra_cost"] > 0
                        else ""
                    )
                ),
                "gain": replacement["gain"],
            }
        )

    items.sort(key=lambda i: -i["gain"])
    return items


def best_replacement(
    player: Mapping,
    elements: Mapping[int, Mapping],
    projections: Mapping,
    owned_ids: set[int],
    horizon: int,
    teams: Mapping[int, Mapping] | None = None,
) -> dict | None:
    """The best affordable, available player in the same position — or nothing.

    Returning ``None`` is the important case, and the reason this is a separate
    function: "there is no upgrade" is the answer for every genuinely elite
    player, and an engine that cannot express it will always find a reason to
    sell one.
    """
    outgoing_id = int(player.get("id", 0))
    outgoing_xpts = _horizon(projections, outgoing_id)
    budget = _price(player) + ASSUMED_BANK
    position = _position(player) or str(player.get("position") or "")

    best: dict | None = None
    for element in elements.values():
        element_id = int(element.get("id", 0))
        if element_id in owned_ids or element_id == outgoing_id:
            continue
        if _position(element) != position:
            continue
        if str(element.get("status", "a")) != "a":
            continue
        price = _price(element)
        if price > budget:
            continue
        candidate_xpts = _horizon(projections, element_id)
        gain = candidate_xpts - outgoing_xpts
        if gain < MIN_TRANSFER_GAIN:
            continue
        if candidate_xpts / max(horizon, 1) < MIN_POINTS_PER_GAMEWEEK:
            continue
        if best is None or gain > best["gain"]:
            best = {
                "player": str(element.get("web_name") or ""),
                "player_id": element_id,
                "price": price,
                "team": _team_short(teams, element),
                "horizon_xpts": candidate_xpts,
                "outgoing_xpts": outgoing_xpts,
                "gain": round(gain, 1),
                "extra_cost": round(max(0.0, price - _price(player)), 1),
            }
    return best


def _team_short(teams: Mapping[int, Mapping] | None, element: Mapping) -> str:
    team = (teams or {}).get(int(element.get("team", 0))) or {}
    return str(team.get("short_name") or team.get("name") or "")


# ------------------------------------------------------------------- buying


#: The four questions the shortlists answer. Same engine, different constraint —
#: the shape that made the squad strategies useful, applied to single transfers.
SHORTLISTS = (
    (
        "form",
        "In form right now",
        "Scoring heavily over the last month, whatever the fixture list says.",
    ),
    (
        "fixtures",
        "Fixtures turning",
        "Their run eases from a named gameweek — buy a week early to own all of it.",
    ),
    (
        "differential",
        "Nobody else owns them",
        "Under 8% ownership. A hit here gains ground on the field rather than "
        "keeping pace with it.",
    ),
    (
        "value",
        "Most points per million",
        "Cheap for what they return, which is what frees money for a premium "
        "elsewhere.",
    ),
)


def buy_shortlists(
    elements: Mapping[int, Mapping],
    projections: Mapping,
    swings: Mapping[int, Mapping],
    owned_ids: set[int],
    squad: Sequence[Mapping],
    horizon: int,
    per_list: int = 3,
    teams: Mapping[int, Mapping] | None = None,
) -> list[dict]:
    """Four ranked shortlists, each answering a different question.

    Every candidate on every list clears :data:`MIN_POINTS_PER_GAMEWEEK` first.
    That single bar is what the old "worth buying" section lacked, and it is why
    it could recommend a player while printing the number that disqualified him.

    Where the squad is known, each entry also names the player it would replace
    and the net gain, because a transfer is a swap and a name on its own leaves
    the decision undone.
    """
    weakest = _weakest_by_position(squad, projections)

    candidates: list[dict] = []
    for element in elements.values():
        element_id = int(element.get("id", 0))
        if element_id in owned_ids:
            continue
        if str(element.get("status", "a")) != "a":
            continue
        position = _position(element)
        if position not in ("GKP", "DEF", "MID", "FWD"):
            continue
        horizon_xpts = _horizon(projections, element_id)
        if horizon_xpts / max(horizon, 1) < MIN_POINTS_PER_GAMEWEEK:
            continue

        price = _price(element)
        entry = {
            "player": str(element.get("web_name") or ""),
            "player_id": element_id,
            "position": position,
            "team": _team_short(teams, element),
            "team_id": int(element.get("team", 0)),
            "price": price,
            "horizon_xpts": round(horizon_xpts, 1),
            "per_gameweek": round(horizon_xpts / max(horizon, 1), 1),
            "form": _float(element.get("form")),
            "ownership": _float(element.get("selected_by_percent")),
            "xpts_per_million": round(horizon_xpts / max(price, 0.1), 2),
        }
        entry.update(_upgrade_over(entry, weakest.get(position)))
        candidates.append(entry)

    lists: list[dict] = []
    for key, title, subtitle in SHORTLISTS:
        picked = _rank(key, candidates, swings)[:per_list]
        if not picked:
            continue
        lists.append(
            {
                "key": key,
                "title": title,
                "subtitle": subtitle,
                "players": picked,
            }
        )
    return lists


def _rank(
    key: str, candidates: Sequence[Mapping], swings: Mapping[int, Mapping]
) -> list[dict]:
    if key == "form":
        pool = [c for c in candidates if c["form"] >= IN_FORM]
        return sorted(pool, key=lambda c: (-c["form"], -c["horizon_xpts"]))

    if key == "fixtures":
        out = []
        for candidate in candidates:
            swing = swings.get(candidate["team_id"])
            if not swing or swing.get("direction") != "improving":
                continue
            out.append(
                {**candidate, "note": f"Run eases from GW{swing['from_gameweek']}."}
            )
        return sorted(out, key=lambda c: -c["horizon_xpts"])

    if key == "differential":
        pool = [c for c in candidates if c["ownership"] <= DIFFERENTIAL_OWNERSHIP]
        return sorted(pool, key=lambda c: -c["horizon_xpts"])

    if key == "value":
        return sorted(candidates, key=lambda c: -c["xpts_per_million"])

    return []


def _weakest_by_position(
    squad: Sequence[Mapping], projections: Mapping
) -> dict[str, dict]:
    """The player in each position this manager would actually drop.

    The realistic transfer is out-the-worst rather than out-the-best, so this is
    what an incoming player has to beat for the gain to be honest.
    """
    weakest: dict[str, dict] = {}
    for player in squad:
        position = _position(player) or str(player.get("position") or "")
        if not position:
            continue
        horizon_xpts = _horizon(projections, int(player.get("id", 0)))
        current = weakest.get(position)
        if current is None or horizon_xpts < current["horizon_xpts"]:
            weakest[position] = {
                "player": str(player.get("web_name") or ""),
                "horizon_xpts": horizon_xpts,
                "price": _price(player),
            }
    return weakest


def _upgrade_over(candidate: Mapping, weakest: Mapping | None) -> dict:
    """What this player gains against the one he would replace, if we know it."""
    if not weakest:
        return {"replaces": None, "gain": None, "affordable": None}
    gain = candidate["horizon_xpts"] - weakest["horizon_xpts"]
    return {
        "replaces": weakest["player"],
        "gain": round(gain, 1),
        # Without the bank balance this is the conservative reading: a
        # like-for-like price swap is always possible, anything dearer may not be.
        "affordable": candidate["price"] <= weakest["price"] + ASSUMED_BANK,
    }
