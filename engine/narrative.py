"""Optional LLM narration of an already-decided transfer plan.

The old design had the LLM *making* the decision: ~85 player records in a prompt,
FPL's rules asked for in English, markdown out. That's the wrong job for it —
unverifiable, untestable, non-deterministic, unparseable.

The instinct was right, though. "+6.4 xPts over 5 GWs" is correct and inert.
So the relationship is inverted: the optimiser decides and verifies, then the
model writes two sentences *about* a plan it cannot change, because it is never
shown an alternative to propose.

Guarantees:

* Output lands in a separate ``narrative`` field. The computed ``reasons`` are
  never touched.
* Any failure — disabled, no CLI, timeout, budget exhausted, guardrail rejection
  — drops ``narrative`` and leaves the response otherwise byte-identical.

**Only the first plan is narrated.** This used to loop over every plan the
optimiser returned, up to five, on a route with no LLM cache — so a single
request was five model calls and the same request repeated was five more. On an
internet-facing route backed by a personal subscription that is an amplifier, not
a feature. The plans after the first are ranked lower and rarely read; one
sentence on the recommendation the user is actually going to take is the whole
value. See :mod:`engine.llm_budget`.

Results are cached on a hash of the plan's *content*, not on the requesting
entry, because two managers with the same squad problem get the same two
sentences and query parameters like ``horizon`` shouldn't mint a new call.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from typing import Mapping, Sequence

from . import fcps_llm, llm_budget

# Narration is two sentences about a decided plan; there is nothing to reason
# about, so the cheapest effort is the right one.
TIMEOUT_SECONDS = 90
EFFORT = "low"
MAX_CHARS = 400

CACHE_TTL_SECONDS = 86_400
_CACHE: dict[str, tuple[float, str]] = {}
_CACHE_MAX = 512

SYSTEM_PROMPT = (
    "You explain a Fantasy Premier League transfer that has already been decided "
    "and verified as legal. Explain only this transfer. Do not suggest "
    "alternatives, do not question the recommendation, do not invent statistics. "
    "Two or three sentences of plain English. No markdown, no headings, no lists."
)


def is_enabled() -> bool:
    """Whether narration should be attempted at all.

    Requires the flag *and* a reachable CLI. The flag alone was never enough —
    the old check paired it with an API key, and this is the same check against
    the transport that replaced it.
    """
    flag = os.getenv("ENABLE_LLM_NARRATIVE", "false").strip().lower()
    return flag in ("1", "true", "yes") and fcps_llm.is_configured()


def model_name() -> str:
    return os.getenv("NARRATIVE_MODEL", fcps_llm.DEFAULT_MODEL).strip() or (
        fcps_llm.DEFAULT_MODEL
    )


def would_call(plans: Sequence[Mapping]) -> bool:
    """Whether annotating these plans would actually spend a model call.

    The route meters callers against an hourly share, and that share must only
    be charged for work that costs something. Narration that is disabled, or
    already cached, costs nothing — billing it would throttle someone for
    reloading a page.
    """
    if not is_enabled() or not plans:
        return False
    return _cache_get(_key_for(plans[0])) is None


def annotate(plans: Sequence[Mapping]) -> None:
    """Attach ``narrative`` to the first plan, in place. Silent on any failure."""
    if not is_enabled() or not plans:
        return
    text = _narrate(plans[0])
    if text:
        plans[0]["narrative"] = text


def _key_for(plan: Mapping) -> str:
    return hashlib.sha256(_prompt(plan).encode("utf-8")).hexdigest()[:32]


def _narrate(plan: Mapping) -> str | None:
    prompt = _prompt(plan)
    key = _key_for(plan)

    cached = _cache_get(key)
    if cached is not None:
        return cached

    binary = fcps_llm.cli_path()
    if binary is None:
        return None

    command = [
        binary,
        "-p",
        "--model",
        model_name(),
        "--effort",
        EFFORT,
        "--output-format",
        "json",
        "--system-prompt",
        SYSTEM_PROMPT,
        "--disallowedTools",
        fcps_llm._DENIED_TOOLS,
    ]

    try:
        with llm_budget.reserve("narrative"), tempfile.TemporaryDirectory(
            prefix="narrative-"
        ) as scratch:
            completed = subprocess.run(
                command,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                cwd=scratch,
            )
        if completed.returncode != 0:
            return None
        payload = json.loads(completed.stdout)
        if payload.get("is_error"):
            return None
        text = str(payload.get("result", "")).strip()
    except (
        llm_budget.BudgetExhausted,
        llm_budget.TooBusy,
        subprocess.TimeoutExpired,
        OSError,
        ValueError,
    ):
        # Narration is strictly additive. Every failure mode, including a spent
        # budget, degrades to the computed reasons rather than to an error.
        return None

    if not text or not _passes_guardrails(text, plan):
        return None
    _cache_put(key, text)
    return text


def _cache_get(key: str) -> str | None:
    entry = _CACHE.get(key)
    if entry is None:
        return None
    stored_at, value = entry
    if time.time() - stored_at > CACHE_TTL_SECONDS:
        _CACHE.pop(key, None)
        return None
    return value


def _cache_put(key: str, value: str) -> None:
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.clear()
    _CACHE[key] = (time.time(), value)


def clear_cache() -> None:
    _CACHE.clear()


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
