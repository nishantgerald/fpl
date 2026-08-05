"""Data assembly for the public, server-rendered pages.

These pages exist for one reason: the Flutter bundle is invisible to a search
engine. Everything it shows is assembled by JavaScript in the visitor's browser
after the page loads, so a crawler fetching the site sees an empty document and
concludes we have no content. The app cannot be found, and a projection nobody
reads is worth nothing.

So this module produces the same numbers as the API, shaped for a template and
served as real HTML at a stable URL. It is a second *view* of one model, never a
second model: every figure here comes from :mod:`engine.service`, so a page and
the app can never quote different numbers for the same player.

Pure assembly — no Flask, no request context — so the whole surface is testable
without a browser.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Mapping, Sequence

from . import rules

# Enough players for a real content surface without generating 700 thin pages
# that compete with each other for the same queries.
INDEXABLE_PLAYER_COUNT = 120


def slugify(value: str) -> str:
    """URL-safe slug. Accents are folded rather than dropped, so
    ``Guéhi`` becomes ``guehi`` and not ``gu-hi``."""
    normalised = unicodedata.normalize("NFKD", value or "")
    ascii_only = normalised.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_only).strip("-").lower()
    return slug or "player"


def player_slug(element: Mapping) -> str:
    """Stable slug for one player, from their full name.

    Full name rather than ``web_name`` because web names collide — there are two
    Palmers in the current bootstrap, and a slug that points at either depending
    on iteration order is a URL that silently changes meaning.
    """
    full = f"{element.get('first_name', '')} {element.get('second_name', '')}".strip()
    return slugify(full or element.get("web_name", ""))


def build_slug_index(elements: Sequence[Mapping]) -> dict[str, int]:
    """``slug -> element id``, with collisions disambiguated by id.

    A collision keeps the first player on the bare slug and gives the second
    ``name-123``, so an existing URL never changes meaning when a namesake
    joins the league mid-season.
    """
    index: dict[str, int] = {}
    for element in sorted(elements, key=lambda e: int(e.get("id", 0))):
        slug = player_slug(element)
        element_id = int(element["id"])
        if slug in index:
            slug = f"{slug}-{element_id}"
        index[slug] = element_id
    return index


def _fixture_label(fixture: Mapping) -> str:
    return f"{fixture['opponent']} ({'H' if fixture['home'] else 'A'})"


def player_page(
    element: Mapping,
    projection: Mapping,
    team_short: str,
    rank_in_position: int,
    position_total: int,
    alternatives: Sequence[Mapping],
    horizon: int,
) -> dict:
    """Everything one player page renders."""
    position = rules.position_of(element)
    price = int(element.get("now_cost", 0)) / 10
    horizon_xpts = float(projection.get("horizon_xpts") or 0.0)

    return {
        "id": int(element["id"]),
        "slug": player_slug(element),
        "web_name": element.get("web_name", ""),
        "full_name": f"{element.get('first_name', '')} "
        f"{element.get('second_name', '')}".strip(),
        "team": team_short,
        "position": position,
        "price": price,
        "ownership": float(element.get("selected_by_percent") or 0.0),
        "status": element.get("status", "a"),
        "news": element.get("news", ""),
        "xpts_next": float(projection.get("xpts_next") or 0.0),
        "horizon_xpts": horizon_xpts,
        "horizon": horizon,
        "value": round(horizon_xpts / price, 2) if price else 0.0,
        "rank_in_position": rank_in_position,
        "position_total": position_total,
        "minutes_risk": projection.get("minutes_risk", "medium"),
        "per_gameweek": [
            {
                "gameweek": entry["gameweek"],
                "xpts": entry["xpts"],
                "label": " + ".join(_fixture_label(f) for f in entry["fixtures"])
                or "Blank",
                "fdr": (
                    round(
                        sum(f["fdr"] for f in entry["fixtures"])
                        / len(entry["fixtures"]),
                        1,
                    )
                    if entry["fixtures"]
                    else None
                ),
                "components": sorted(
                    (
                        {"name": name.replace("_", " "), "value": value}
                        for name, value in (entry.get("components") or {}).items()
                        if abs(value) >= 0.05
                    ),
                    key=lambda c: -c["value"],
                ),
            }
            for entry in projection.get("per_gameweek", [])
        ],
        "alternatives": list(alternatives),
    }


def verdict(page: Mapping) -> str:
    """One plain sentence answering "should I pick him?".

    Fixed phrasing assembled from the numbers, not generated. The judgement is
    the same for every reader, so paying a language model to rephrase it per
    visitor would buy variance in something that ought to be consistent — and
    would put a paraphrase between the reader and a figure they can check.
    """
    name = page["web_name"]
    rank = page["rank_in_position"]
    position = page["position"]
    horizon = page["horizon"]

    if page["status"] in ("i", "s", "u"):
        return (
            f"{name} is flagged unavailable, so he projects nothing over the "
            f"next {horizon} gameweeks whatever his underlying numbers say."
        )

    standing = (
        f"the highest-projected {position} in the game"
        if rank == 1
        else f"the number {rank} {position} of {page['position_total']}"
    )
    line = (
        f"At £{page['price']:.1f}m, {name} projects {page['horizon_xpts']:.1f} "
        f"points over the next {horizon} gameweeks — {standing}, and "
        f"{page['value']:.1f} points per £1m."
    )
    if page["minutes_risk"] == "high":
        line += " His minutes are the risk: he is not a guaranteed starter."
    elif page["ownership"] >= 30:
        line += (
            f" At {page['ownership']:.0f}% ownership he is close to essential — "
            "not owning him is itself a position."
        )
    return line + " Check team news before the deadline."
