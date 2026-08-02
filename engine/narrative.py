"""Optional LLM narration of an already-decided transfer plan.

The old design had the LLM *making* the decision: ~85 player records in a prompt,
FPL's rules asked for in English, markdown out. That's the wrong job for it —
unverifiable, untestable, non-deterministic, unparseable.

The instinct was right, though. "+6.4 xPts over 5 GWs" is correct and inert.
So the relationship is inverted: the optimiser decides and verifies, then the
model writes two sentences *about* a plan it cannot change, because it is never
shown an alternative to propose.

Guarantees:

* Off by default. No key, no flag, no call.
* Output lands in a separate ``narrative`` field. The computed ``reasons`` are
  never touched.
* Any failure — disabled, missing key, timeout, guardrail rejection — drops
  ``narrative`` and leaves the response otherwise byte-identical.
"""

from __future__ import annotations

import os
from typing import Mapping, Sequence

TIMEOUT_SECONDS = 6
MAX_CHARS = 400
MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You explain a Fantasy Premier League transfer that has already been decided "
    "and verified as legal. Explain only this transfer. Do not suggest "
    "alternatives, do not question the recommendation, do not invent statistics. "
    "Two or three sentences of plain English. No markdown, no headings, no lists."
)


def is_enabled() -> bool:
    """Whether narration should be attempted at all."""
    flag = os.getenv("ENABLE_LLM_NARRATIVE", "false").strip().lower()
    return flag in ("1", "true", "yes") and bool(os.getenv("OPENAI_API_KEY"))


def annotate(plans: Sequence[Mapping]) -> None:
    """Attach ``narrative`` to each plan, in place. Silent on any failure."""
    if not is_enabled() or not plans:
        return
    try:
        from openai import OpenAI
    except ImportError:
        return

    try:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=TIMEOUT_SECONDS)
    except Exception:
        return

    for plan in plans:
        text = _narrate(client, plan)
        if text:
            plan["narrative"] = text


def _narrate(client, plan: Mapping) -> str | None:
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            max_tokens=160,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _prompt(plan)},
            ],
        )
        text = (completion.choices[0].message.content or "").strip()
    except Exception:
        return None
    return text if _passes_guardrails(text, plan) else None


def _prompt(plan: Mapping) -> str:
    """The plan, and nothing but the plan — roughly 2 KB versus the old ~40 KB."""
    lines = [
        f"Transfers: {plan['n_transfers']}",
        f"Points hit: {plan.get('hit_cost', 0)}",
        f"Net gain over the horizon: {plan.get('net_gain', 0)} points",
        f"Spend: {plan.get('spend', 0)} tenths of a million",
        "",
    ]
    for transfer in plan.get("transfers", ()):
        out_ref, in_ref = transfer["out"], transfer["in"]
        lines.append(
            f"OUT {out_ref['web_name']} ({out_ref['position']}, "
            f"{out_ref.get('team', '?')}, sells for {out_ref.get('selling_price', '?')})"
        )
        lines.append(
            f"IN  {in_ref['web_name']} ({in_ref['position']}, "
            f"{in_ref.get('team', '?')}, costs {in_ref['now_cost']})"
        )
    lines.append("")
    lines.append("Computed reasons:")
    lines.extend(f"- {reason}" for reason in plan.get("reasons", ()))
    return "\n".join(lines)


def _passes_guardrails(text: str, plan: Mapping) -> bool:
    """Cheap mechanical checks. Degrading to the numbers is always acceptable."""
    if not text or len(text) > MAX_CHARS:
        return False

    allowed = set()
    for transfer in plan.get("transfers", ()):
        for side in ("out", "in"):
            ref = transfer[side]
            allowed.add(str(ref.get("web_name", "")).lower())
            for part in str(ref.get("name", "")).split():
                allowed.add(part.lower())
    allowed.discard("")

    # A capitalised word mid-sentence that isn't in the plan is very likely an
    # invented alternative, which is the one failure mode worth catching.
    words = text.replace(",", " ").replace(".", " ").split()
    for word in words[1:]:
        cleaned = word.strip("'\"()").lower()
        if word[:1].isupper() and cleaned.isalpha() and len(cleaned) > 3:
            if cleaned not in allowed and cleaned not in _SAFE_WORDS:
                return False
    return True


# Capitalised words that legitimately appear without being a player name.
_SAFE_WORDS = frozenset(
    {
        "gameweek", "gameweeks", "fpl", "fantasy", "premier", "league", "the",
        "this", "that", "their", "they", "his", "her", "your", "you", "with",
        "arsenal", "aston", "villa", "bournemouth", "brentford", "brighton",
        "burnley", "chelsea", "crystal", "palace", "everton", "fulham", "leeds",
        "liverpool", "luton", "manchester", "city", "united", "newcastle",
        "nottingham", "forest", "sheffield", "southampton", "tottenham",
        "hotspur", "spurs", "west", "ham", "wolves", "wolverhampton", "ipswich",
        "leicester", "sunderland",
    }
)
