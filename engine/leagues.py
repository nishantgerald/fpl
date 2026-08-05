"""Mini-league analysis: who you are actually playing against.

Overall rank is a number nobody feels. The league that matters is the one with
twelve people you know in it, and the decisions that win it are different from
the decisions that maximise expected points: when you lead, you cover what your
rivals own; when you chase, you deliberately do not.

Two ideas carry the module.

**Effective ownership within the league.** Global ownership says how many of ten
million managers hold a player. It is the wrong denominator entirely. What
decides your green arrow is how many of *your eleven rivals* hold him, doubled
where they captain him — because a template captain scoring is a wash, and a
differential captain scoring is the gameweek.

**Cover versus differential.** The same player is a good pick or a bad one
depending on where you sit. Leading, an unowned high-ownership player is a
liability you should probably fix. Chasing, he is noise, and the player nobody
around you owns is the only thing that closes a gap.

Pure: payloads in, analysis out. No I/O, so a season of league shapes can be
tested in memory.
"""

from __future__ import annotations

from typing import Mapping, Sequence

# Fetching a rival's squad costs one upstream request each, so a 200-member
# league would cost 200. Analysis is capped at the managers close enough to
# matter — the ones you can realistically catch or be caught by.
DEFAULT_RIVAL_WINDOW = 8
MAX_RIVALS = 24

# Leagues everyone is enrolled in automatically. They are not mini-leagues in
# any meaningful sense and analysing them would be nonsense.
GLOBAL_LEAGUE_IDS = {314}


def is_meaningful_league(league: Mapping) -> bool:
    """Whether a league is one the manager actually competes in.

    FPL auto-enrols every entry into leagues for their country, their favourite
    club, and the gameweek they started. Those have millions of members and no
    social meaning, so ranking advice about them is noise.
    """
    if int(league.get("id", 0)) in GLOBAL_LEAGUE_IDS:
        return False
    # `league_type` "s" is a system league; "x" is one a human created.
    return str(league.get("league_type", "x")) == "x"


def classify_leagues(entry_payload: Mapping) -> list[dict]:
    """The manager's leagues, with the auto-enrolled ones marked.

    Discovered from their entry rather than typed in: asking a user to find a
    League ID is asking them to leave and not come back.
    """
    leagues = (entry_payload or {}).get("leagues", {}).get("classic", []) or []
    return [
        {
            "id": int(league.get("id", 0)),
            "name": league.get("name", ""),
            "rank": league.get("entry_rank"),
            "last_rank": league.get("entry_last_rank"),
            "size": league.get("rank_count"),
            "meaningful": is_meaningful_league(league),
        }
        for league in leagues
    ]


def rivals_around(
    standings: Sequence[Mapping],
    entry_id: int,
    window: int = DEFAULT_RIVAL_WINDOW,
) -> dict:
    """The managers close enough to catch, or to be caught by.

    Returns the entry's own row plus the ``window`` above and below. Someone
    forty places clear is not a rival in any actionable sense, and fetching
    their squad costs a request that buys nothing.
    """
    rows = list(standings or [])
    position = next(
        (i for i, r in enumerate(rows) if int(r.get("entry", 0)) == entry_id), None
    )
    if position is None:
        return {"me": None, "above": [], "below": []}

    return {
        "me": rows[position],
        "above": rows[max(0, position - window) : position],
        "below": rows[position + 1 : position + 1 + window],
    }


def effective_ownership(
    squads: Mapping[int, Sequence[Mapping]],
    elements: Mapping[int, Mapping],
) -> dict[int, dict]:
    """How exposed the league is to each player, counting captaincy twice.

    ``squads`` maps entry id to that manager's picks. A player started by half
    the league and captained by a quarter of it is far more dangerous to be
    without than raw ownership suggests, which is exactly what effective
    ownership measures and plain ownership misses.
    """
    total = len(squads) or 1
    tally: dict[int, dict] = {}

    for picks in squads.values():
        for pick in picks or []:
            element_id = int(pick.get("element", 0))
            multiplier = int(pick.get("multiplier", 0))
            entry = tally.setdefault(
                element_id,
                {"owned": 0, "started": 0, "captained": 0, "element_id": element_id},
            )
            entry["owned"] += 1
            if multiplier > 0:
                entry["started"] += 1
            if multiplier > 1:
                entry["captained"] += 1

    for element_id, entry in tally.items():
        element = elements.get(element_id, {})
        entry["web_name"] = element.get("web_name", "")
        entry["ownership"] = round(entry["owned"] / total * 100, 1)
        # Started once, captained twice — the standard formulation.
        entry["effective_ownership"] = round(
            (entry["started"] + entry["captained"]) / total * 100, 1
        )
    return tally


def differentials(
    my_picks: Sequence[Mapping],
    ownership: Mapping[int, Mapping],
    projections: Mapping[int, Mapping],
    elements: Mapping[int, Mapping],
    limit: int = 8,
) -> dict:
    """What separates this squad from the league, in both directions.

    ``gaps`` are players the league owns and this manager does not — every one
    is a gameweek where a rival can gain without doing anything. ``edges`` are
    the reverse: the players that can actually move this manager up.
    """
    mine = {int(p.get("element", 0)) for p in my_picks or []}

    def row(element_id: int, entry: Mapping) -> dict:
        element = elements.get(element_id, {})
        projection = projections.get(element_id, {})
        return {
            "id": element_id,
            "web_name": element.get("web_name", ""),
            "price": int(element.get("now_cost", 0)) / 10,
            "league_ownership": entry.get("ownership", 0.0),
            "effective_ownership": entry.get("effective_ownership", 0.0),
            "xpts": round(float(projection.get("horizon_xpts") or 0.0), 1),
        }

    gaps = [
        row(eid, entry)
        for eid, entry in ownership.items()
        if eid not in mine and entry.get("effective_ownership", 0) > 0
    ]
    edges = [
        row(eid, ownership.get(eid, {"ownership": 0.0, "effective_ownership": 0.0}))
        for eid in mine
    ]

    # Gaps rank by how exposed they leave you; edges by how little the league
    # holds them, since a player everyone owns is not an edge at all.
    gaps.sort(key=lambda r: (-r["effective_ownership"], -r["xpts"]))
    edges.sort(key=lambda r: (r["effective_ownership"], -r["xpts"]))

    return {"gaps": gaps[:limit], "edges": edges[:limit]}


def posture(me: Mapping | None, rivals: Mapping) -> dict:
    """Whether to protect a lead or chase one, and what that implies.

    The advice inverts on which side of the gap you sit, and getting it
    backwards is worse than saying nothing: covering the template while you
    chase guarantees you finish exactly where you started.
    """
    if not me:
        return {
            "stance": "unknown",
            "headline": "Not in this league's standings yet.",
            "advice": "Standings appear once the first gameweek has scored.",
        }

    my_points = int(me.get("total", 0))
    above = rivals.get("above") or []
    below = rivals.get("below") or []
    rank = int(me.get("rank", 0) or 0)

    leader_gap = my_points - int(above[0].get("total", my_points)) if above else 0
    chaser_gap = my_points - int(below[0].get("total", my_points)) if below else 0

    if not above:
        return {
            "stance": "protect",
            "headline": f"Top of the league on {my_points} points.",
            "advice": (
                "Cover what the pack owns. A differential that fails costs you "
                "the lead; a template player that hauls costs you nothing."
            ),
            "gap_above": 0,
            "gap_below": chaser_gap,
        }

    if leader_gap > -20 and rank <= 3:
        return {
            "stance": "protect",
            "headline": f"{abs(leader_gap)} points off the lead, and close.",
            "advice": (
                "Stay near the template on captaincy. Take differentials only "
                "where the projection genuinely justifies them."
            ),
            "gap_above": leader_gap,
            "gap_below": chaser_gap,
        }

    return {
        "stance": "chase",
        "headline": f"{abs(leader_gap)} points behind the manager above you.",
        "advice": (
            "Matching the template preserves the gap rather than closing it. "
            "Look for players your rivals do not own — and captain differently "
            "when the projection is close, because the armband is where the "
            "swing is largest."
        ),
        "gap_above": leader_gap,
        "gap_below": chaser_gap,
    }
