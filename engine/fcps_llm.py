"""FCPS transfer advice — the LLM layer, restored and made reachable.

This is the original ``get_trade_recommendations()`` feature: rank the league by
:mod:`engine.fcps`, hand the manager's squad and an FCPS-ranked shortlist to a
language model, and get back a written transfer column in markdown.

Why it was broken, precisely:

1. The only ``GET /trade_recommendations`` did ``render_template(
   "trade_recommendations.html")``. That file never existed in the repo, so every
   GET raised ``TemplateNotFound`` and Flask returned **500**.
2. The working path was ``POST`` with a form body — and the Flutter client, which
   is the only shipped UI, never called it. So the feature was unreachable from
   the app even when the server was healthy.
3. Both failure modes were invisible: a missing ``OPENAI_API_KEY`` returned
   *200 OK* with the string ``"Error generating trade recommendations: ..."``
   sitting in the ``recommendation`` field, so the client had no way to tell
   advice from an exception.

All three are fixed here. This module has no templates and no Flask imports; it
returns data. The route is ``GET /api/fcps-recommendations``, which the client
calls. Failures raise :class:`FcpsUnavailable` with a machine-readable code, so a
missing key renders as "not configured" rather than as advice.

The prompt itself is kept close to the original — it is the feature — but it is
fed compact rows instead of two whole ``DataFrame.to_dict()`` dumps, which was
roughly 40 KB of JSON per call.
"""

from __future__ import annotations

import os
import time
from typing import Mapping, Sequence

from . import rules

DEFAULT_MODEL = "gpt-4o-mini"
TIMEOUT_SECONDS = 45
MAX_OUTPUT_TOKENS = 1600

# Responses are cached per (entry, gameweek, model). A transfer column doesn't
# change between two taps of the same button, and each call costs real money.
CACHE_TTL_SECONDS = 900
_CACHE: dict[tuple, tuple[float, dict]] = {}
_CACHE_MAX = 64


class FcpsUnavailable(Exception):
    """FCPS advice could not be produced. Carries a code the client can branch on.

    Codes:
        ``fcps_not_configured``  no API key on the server
        ``fcps_sdk_missing``     the openai package isn't installed
        ``fcps_upstream_error``  the model call failed or timed out
    """

    def __init__(self, code: str, message: str, status: int = 503):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status

    def to_dict(self) -> dict:
        return {"code": self.code, "error": self.message}


def is_configured() -> bool:
    """Whether the server can produce FCPS advice at all.

    Exposed so the client can hide or explain the feature *before* the user taps
    a button and waits 30 seconds for a 503.
    """
    return bool(os.getenv("OPENAI_API_KEY"))


def model_name() -> str:
    return os.getenv("FCPS_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


# ---------------------------------------------------------------- row shaping


def player_row(
    element: Mapping,
    fcps_entry: Mapping,
    team_short: str,
    in_squad: bool = False,
    starting: bool = False,
) -> dict:
    """One player as the prompt sees them.

    Eleven fields, not the ~40 of a bootstrap element. Everything the original
    prompt actually referenced is here; nothing else is.
    """
    row = {
        "id": int(element["id"]),
        "name": f"{element.get('first_name', '')} "
        f"{element.get('second_name', '')}".strip()
        or str(element.get("web_name", "")),
        "team": team_short,
        "position": rules.position_of(element),
        "price": round(int(element.get("now_cost", 0)) / 10, 1),
        "total_points": int(element.get("total_points", 0)),
        "form": _num(element.get("form")),
        "next_3_fdr": int(fcps_entry.get("next_3_fdr", 0)),
        "ict_index": _num(element.get("ict_index")),
        "fcps": round(float(fcps_entry.get("fcps", 0.0)), 1),
        "status": str(element.get("status", "a")),
    }
    if in_squad:
        row["starting_eleven"] = bool(starting)
    return row


def _num(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------- the prompt


SYSTEM_PROMPT = (
    "You are an expert Fantasy Premier League analyst. You write concise, "
    "concrete transfer columns. You never invent players, prices or statistics: "
    "every player you name and every number you quote must appear in the data "
    "you are given."
)


def build_prompt(
    squad_rows: Sequence[Mapping],
    shortlist_rows: Sequence[Mapping],
    gameweek: int,
    bank: int | None = None,
    free_transfers: int | None = None,
) -> str:
    """The original prompt, with the rules restated and the budget made explicit.

    The original told the model the 3-per-club cap and "ensure the trade is
    actually feasible in cost" without ever telling it the manager's bank
    balance, which was one field away in a response the app already parsed. It is
    passed now. That does not make the output verifiable — an LLM asked to
    respect an arithmetic constraint in English will still break it, which is why
    :mod:`engine.optimizer` exists and why this route is labelled as opinion in
    the UI — but withholding the number guaranteed failure.
    """
    budget_lines = []
    if bank is not None:
        budget_lines.append(f"- Money in the bank: {bank / 10:.1f}m")
    if free_transfers is not None:
        budget_lines.append(
            f"- Free transfers available: {free_transfers} "
            f"(each extra transfer costs 4 points)"
        )

    return f"""Gameweek {gameweek}.

FCPS (Fantasy Composite Player Score) is a 0-1000 composite of a player's total
points (20%), recent form (40%), the difficulty of their next 3 fixtures (25%)
and their ICT index (15%). Higher is better.

The manager's current 15-player squad:
{_table(squad_rows)}

The highest-FCPS available players by position:
{_table(shortlist_rows)}

{chr(10).join(budget_lines)}

Recommend transfers for this gameweek. Constraints you must respect:
- A squad may contain at most 3 players from the same club.
- Never suggest transferring IN a player who is already in the squad.
- Never suggest transferring OUT a player who is not in the squad.
- A transfer must be affordable: (price in) - (price out) must be covered by the
  bank. Say so explicitly when a move is tight.
- Transfers in and out must be the same position.
- If no transfer is clearly worth making, say so and recommend holding.

Structure your response as markdown, grouped by position, in this shape:

# Fantasy Premier League Transfer Recommendations

## 1. Goalkeeper
* **Out:** Name (GKP, TEAM) — 5.0
  Reason: one sentence.
* **In:** Name (GKP, TEAM) — 4.5
  Total points 79 · Form 4.8 · Next 3 FDR 7 · FCPS 426.0 · ICT 51.6

...repeat for Defenders, Midfielders and Forwards, omitting any position where
you don't recommend a change...

## Summary

| Out | In | Position | Price change | Note |
|-----|----|----------|--------------|------|

## Conclusion

Two or three sentences.
"""


def _table(rows: Sequence[Mapping]) -> str:
    """Rows as a compact markdown table.

    Costs about a fifth of the tokens of the original's ``to_dict(orient=
    'records')`` dump, and is markedly easier for the model to read across.
    """
    if not rows:
        return "_(none)_"

    header = (
        "| Name | Pos | Team | £ | Pts | Form | FDR3 | ICT | FCPS | Status |"
        "\n|---|---|---|---|---|---|---|---|---|---|"
    )
    # Status is spelled out rather than passed through as FPL's single letter.
    # "i" means nothing to a language model; "injured" does, and the whole point
    # of the column is that it shouldn't recommend an injured player.
    words = {
        "a": "fit",
        "d": "doubtful",
        "i": "injured",
        "s": "suspended",
        "u": "unavailable",
        "n": "not in squad",
    }

    lines = [header]
    for row in rows:
        status = str(row.get("status", "a"))
        label = words.get(status, status)
        if row.get("starting_eleven") is False:
            label += ", benched"
        lines.append(
            f"| {row['name']} | {row['position']} | {row['team']} | "
            f"{row['price']:.1f} | {row['total_points']} | {row['form']:.1f} | "
            f"{row['next_3_fdr']} | {row['ict_index']:.1f} | "
            f"{row['fcps']:.0f} | {label} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------- the call


def advise(
    squad_rows: Sequence[Mapping],
    shortlist_rows: Sequence[Mapping],
    gameweek: int,
    bank: int | None = None,
    free_transfers: int | None = None,
    cache_key: tuple | None = None,
    refresh: bool = False,
) -> dict:
    """Produce the FCPS transfer column.

    Raises :class:`FcpsUnavailable` rather than returning an error string in the
    success field — the original returned ``"Error generating trade
    recommendations: ..."`` with a 200, which the client rendered as advice.
    """
    if cache_key is not None and not refresh:
        cached = _cache_get(cache_key)
        if cached is not None:
            return {**cached, "cached": True}

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise FcpsUnavailable(
            "fcps_not_configured",
            "FCPS advice needs an OpenAI API key on the server. "
            "Set OPENAI_API_KEY to enable it.",
        )

    try:
        from openai import OpenAI
    except ImportError:  # pragma: no cover - depends on the install
        raise FcpsUnavailable(
            "fcps_sdk_missing",
            "The openai package isn't installed on the server.",
        ) from None

    prompt = build_prompt(squad_rows, shortlist_rows, gameweek, bank, free_transfers)

    try:
        client = OpenAI(api_key=api_key, timeout=TIMEOUT_SECONDS)
        completion = client.chat.completions.create(
            model=model_name(),
            temperature=0.2,
            max_tokens=MAX_OUTPUT_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        markdown = (completion.choices[0].message.content or "").strip()
    except Exception as error:  # the SDK raises a wide family of exceptions
        raise FcpsUnavailable(
            "fcps_upstream_error",
            f"The model call failed: {type(error).__name__}.",
            status=502,
        ) from error

    if not markdown:
        raise FcpsUnavailable(
            "fcps_upstream_error", "The model returned an empty response.", status=502
        )

    result = {
        "markdown": markdown,
        "model": model_name(),
        "gameweek": gameweek,
        "squad_size": len(squad_rows),
        "shortlist_size": len(shortlist_rows),
        "mentions": audit_mentions(markdown, squad_rows, shortlist_rows),
        "cached": False,
    }
    if cache_key is not None:
        _cache_put(cache_key, result)
    return result


def audit_mentions(
    markdown: str,
    squad_rows: Sequence[Mapping],
    shortlist_rows: Sequence[Mapping],
) -> dict:
    """Cheap check on whether the prose stayed inside the data it was given.

    Not a guardrail — nothing is rejected — but the counts are returned so the UI
    can carry an honest caveat, and so a regression in prompt quality is visible
    as a number rather than as a vibe. Surnames are matched because the model
    routinely shortens "Bruno Guimaraes" to "Guimaraes".
    """
    known: set[str] = set()
    for row in list(squad_rows) + list(shortlist_rows):
        name = str(row.get("name", ""))
        for part in name.split():
            if len(part) > 3:
                known.add(part.lower().strip("'-"))

    lower = markdown.lower()
    matched = sorted(n for n in known if n in lower)
    return {"known_players_named": len(matched), "data_rows": len(known)}


# ---------------------------------------------------------------- cache


def _cache_get(key: tuple) -> dict | None:
    entry = _CACHE.get(key)
    if entry is None:
        return None
    stored_at, value = entry
    if time.time() - stored_at > CACHE_TTL_SECONDS:
        _CACHE.pop(key, None)
        return None
    return value


def _cache_put(key: tuple, value: dict) -> None:
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.clear()
    _CACHE[key] = (time.time(), value)


def clear_cache() -> None:
    _CACHE.clear()
