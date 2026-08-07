"""Captain picks.

The armband is worth roughly a third of a weekly score and is the one decision a
manager makes every single gameweek. The old app rendered ``is_captain`` as a
badge and offered no opinion at all.

Captaincy is a *ceiling* decision, not a mean one: two players on 6.0 xPts are
not equivalent if one is a striker who can return 15 and the other is a defender
whose upside is a clean sheet. So the ranking tilts toward the volatile
components — goals, assists, bonus — while always showing the plain xPts
alongside, so the tilt is visible rather than smuggled in.

Pure: projections and picks in, ranking out.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from . import rules

# How much extra weight the volatile (ceiling) components carry.
CEILING_TILT = 0.35

# Above this share of managers, captaining a player is a low-variance choice.
SAFE_CAPTAINCY_SHARE = 20.0

# Gap at which switching the armband is worth flagging.
NOTABLE_GAP = 2.0

CEILING_COMPONENTS = ("goals", "assists", "bonus")


def score_candidate(
    *,
    player_id: int,
    web_name: str,
    name: str,
    team: str,
    position: str,
    xpts_next: float,
    components: Mapping,
    fixtures: Sequence[Mapping],
    ownership: float,
    availability: float,
    minutes_risk: str,
) -> dict:
    """One armband candidate, scored the way this module argues for.

    Shared so that every list of captain candidates is the same list. The public
    ranking used to sort on plain ``xpts_next`` and publish it doubled, which is
    a different answer to a different question: doubling is monotonic, so it
    changes no order and adds no information, and ranking on the mean is exactly
    the thing the module docstring says captaincy is not.
    """
    ceiling = sum(float(components.get(k, 0.0)) for k in CEILING_COMPONENTS)
    return {
        "id": player_id,
        "name": name,
        "web_name": web_name,
        "team": team,
        "position": position,
        "xpts": round(xpts_next, 2),
        # What the armband is actually worth. Kept because it answers "what do
        # I get", but never sorted on: doubling every number changes no order,
        # so a list ranked by it is a list ranked by plain xPts wearing a
        # bigger figure.
        "xpts_captained": round(xpts_next * 2, 2),
        "captain_score": round(xpts_next + CEILING_TILT * ceiling, 2),
        "ceiling": _ceiling_band(ceiling, xpts_next),
        "opponent": _opponent_str(fixtures),
        "fdr": fixtures[0]["fdr"] if fixtures else None,
        "blanking": not fixtures,
        "ownership": float(ownership or 0.0),
        "availability": availability,
        "minutes_risk": minutes_risk,
    }


def order_candidates(
    entries: list[dict], most_captained: int | None = None
) -> list[dict]:
    """Sort, rank and annotate in place.

    Blanking or unavailable players can never top the list, whatever their
    underlying numbers say.
    """
    entries.sort(
        key=lambda e: (
            e["blanking"] or e["availability"] <= 0.0,
            -e["captain_score"],
            e["id"],
        )
    )
    for rank, entry in enumerate(entries, start=1):
        entry["rank"] = rank
        entry["safety"] = _safety(entry, most_captained)
        entry["note"] = _note(entry)
    return entries


def rank_global(
    projections: Mapping[int, Mapping],
    elements: Mapping[int, Mapping],
    teams: Mapping[int, str],
    position_of,
    price_of,
    limit: int,
    most_captained: int | None = None,
) -> list[dict]:
    """The whole league ranked for the armband, for a visitor with no squad.

    Same scoring as the squad-scoped ranking. A public page that ordered players
    differently from the signed-in one would be recommending two captains for
    one gameweek and calling both of them best.
    """
    entries: list[dict] = []
    for player_id, projection in projections.items():
        element = elements.get(player_id)
        if element is None:
            continue
        xpts_next = float(projection.get("xpts_next") or 0.0)
        if xpts_next <= 0:
            continue
        first = (projection.get("per_gameweek") or [{}])[0]
        fixtures = first.get("fixtures") or []
        # A blank gameweek cannot be captained at all, so it is excluded here
        # rather than merely sunk: on a public list there is no squad forcing
        # the choice, and a name nobody can pick is not a candidate.
        if not fixtures:
            continue
        entries.append(
            score_candidate(
                player_id=player_id,
                web_name=element.get("web_name", ""),
                name=_name(element),
                team=teams.get(int(element.get("team", 0)), "UNK"),
                position=position_of(element),
                xpts_next=xpts_next,
                components=first.get("components") or {},
                fixtures=fixtures,
                ownership=element.get("selected_by_percent") or 0.0,
                availability=projection.get("availability", 1.0),
                minutes_risk=projection.get("minutes_risk", "medium"),
            )
            | {"price": price_of(element), "fixtures": fixtures}
        )

    return order_candidates(entries, most_captained)[:limit]


def rank_captains(
    squad: Sequence[Mapping],
    picks: Sequence[Mapping],
    projections: Mapping[int, Mapping],
    teams: Mapping[int, Mapping],
    gameweek: int,
    most_captained: int | None = None,
) -> dict:
    """Rank the manager's 15 for the armband, with warnings."""
    squad_by_id = {int(p["id"]): p for p in squad}
    current_captain = _pick_with(picks, "is_captain")
    current_vice = _pick_with(picks, "is_vice_captain")

    entries: list[dict] = []
    for player in squad:
        pid = int(player["id"])
        proj = projections.get(pid)
        if not proj:
            continue
        gw_entry = _gameweek_entry(proj, gameweek)
        entries.append(
            score_candidate(
                player_id=pid,
                web_name=player.get("web_name", ""),
                name=_name(player),
                team=_team_short(teams, player.get("team")),
                position=rules.position_of(player),
                xpts_next=float(gw_entry.get("xpts", 0.0)) if gw_entry else 0.0,
                components=(gw_entry or {}).get("components", {}),
                fixtures=(gw_entry or {}).get("fixtures", []),
                ownership=player.get("selected_by_percent") or 0.0,
                availability=proj.get("availability", 1.0),
                minutes_risk=proj.get("minutes_risk", "medium"),
            )
        )

    order_candidates(entries, most_captained)
    by_id = {e["id"]: e for e in entries}
    return {
        "gameweek": gameweek,
        "current_captain": by_id.get(current_captain),
        "current_vice": by_id.get(current_vice),
        "picks": entries,
        "warnings": _warnings(by_id, current_captain, current_vice, entries, squad_by_id),
    }


def _pick_with(picks: Sequence[Mapping], flag: str) -> int | None:
    for pick in picks:
        if pick.get(flag):
            return int(pick["element"])
    return None


def _gameweek_entry(proj: Mapping, gameweek: int) -> Mapping | None:
    for entry in proj.get("per_gameweek", ()):
        if entry["gameweek"] == gameweek:
            return entry
    return None


def _ceiling_band(ceiling: float, xpts: float) -> str:
    if xpts <= 0:
        return "none"
    share = ceiling / xpts
    if share >= 0.55:
        return "high"
    if share >= 0.3:
        return "medium"
    return "low"


def _safety(entry: Mapping, most_captained: int | None) -> str:
    if most_captained is not None and entry["id"] == most_captained:
        return "safe"
    if entry["ownership"] >= SAFE_CAPTAINCY_SHARE:
        return "safe"
    if entry["ownership"] >= 5.0:
        return "balanced"
    return "differential"


def _note(entry: Mapping) -> str:
    if entry["blanking"]:
        return "No fixture this gameweek."
    if entry["availability"] <= 0.0:
        return "Unavailable — do not captain."
    if entry["availability"] < 1.0:
        return f"Flagged: {entry['availability'] * 100:.0f}% chance of playing."
    if entry["minutes_risk"] == "high":
        return "Rotation risk — started a minority of games."
    if entry["ceiling"] == "high":
        return f"High ceiling vs {entry['opponent']}."
    return f"Faces {entry['opponent']}."


def _warnings(
    by_id: Mapping[int, Mapping],
    captain_id: int | None,
    vice_id: int | None,
    entries: Sequence[Mapping],
    squad_by_id: Mapping[int, Mapping],
) -> list[dict]:
    """The actual value of this feature: telling someone their armband is broken."""
    warnings: list[dict] = []
    captain = by_id.get(captain_id) if captain_id else None
    vice = by_id.get(vice_id) if vice_id else None

    if captain:
        if captain["availability"] <= 0.0:
            warnings.append(
                {
                    "severity": "critical",
                    "message": f"Your captain {captain['web_name']} is unavailable.",
                }
            )
        elif captain["blanking"]:
            warnings.append(
                {
                    "severity": "critical",
                    "message": f"Your captain {captain['web_name']} has no fixture this gameweek.",
                }
            )
        elif captain["availability"] < 0.75:
            warnings.append(
                {
                    "severity": "warning",
                    "message": (
                        f"Your captain {captain['web_name']} is flagged "
                        f"({captain['availability'] * 100:.0f}% chance of playing)."
                    ),
                }
            )

    if vice and (vice["availability"] <= 0.0 or vice["blanking"]):
        warnings.append(
            {
                "severity": "critical",
                "message": (
                    f"Your vice-captain {vice['web_name']} can't cover — "
                    "they're unavailable or blanking too."
                ),
            }
        )

    if entries and captain:
        top = entries[0]
        gap = top["xpts"] - captain["xpts"]
        if top["id"] != captain["id"] and gap > NOTABLE_GAP:
            warnings.append(
                {
                    "severity": "info",
                    "message": (
                        f"{top['web_name']} projects {gap:.1f} pts more than "
                        f"{captain['web_name']} this gameweek."
                    ),
                }
            )

    severity_order = {"critical": 0, "warning": 1, "info": 2}
    warnings.sort(key=lambda w: severity_order.get(w["severity"], 3))
    return warnings


def _name(player: Mapping) -> str:
    full = f"{player.get('first_name', '')} {player.get('second_name', '')}".strip()
    return full or str(player.get("web_name", ""))


def _team_short(teams: Mapping[int, Mapping], team_id) -> str:
    team = teams.get(int(team_id or 0))
    return str(team.get("short_name", "UNK")) if team else "UNK"


def _opponent_str(fixtures: Sequence[Mapping]) -> str:
    if not fixtures:
        return "—"
    return " + ".join(
        f"{fx['opponent']} ({'H' if fx['home'] else 'A'})" for fx in fixtures
    )
