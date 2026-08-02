"""Derive how many free transfers a manager has banked.

FPL does not publish this on any unauthenticated endpoint, but it is fully
determined by the public transfer history, so we reconstruct it.

The rule (2024/25 onward):

  * You start with 1 free transfer.
  * You gain 1 more each gameweek, capped at 5 banked.
  * Each transfer you make consumes one; beyond that you pay -4 points each.
  * Transfers made while playing a Wildcard or Free Hit are free and consume
    none, and any banked free transfers carry over that gameweek.

This is a derivation, not a fact we were told, so callers surface it with
``free_transfers_source: "derived"`` and always allow a user override.
"""

from __future__ import annotations

from typing import Mapping, Sequence

MAX_FREE_TRANSFERS = 5
MIN_FREE_TRANSFERS = 1

# Chips that make a gameweek's transfers free and preserve the bank.
UNLIMITED_TRANSFER_CHIPS = frozenset({"wildcard", "freehit"})


def derive_free_transfers(
    history_current: Sequence[Mapping] | None,
    chips: Sequence[Mapping] | None = None,
    max_free: int = MAX_FREE_TRANSFERS,
) -> int:
    """Free transfers available for the gameweek *after* the last one in history.

    ``history_current`` is ``entry/{id}/history/``'s ``current`` array, each
    entry carrying ``event`` and ``event_transfers``. ``chips`` is the same
    payload's ``chips`` array, each carrying ``name`` and ``event``.

    Returns a value in ``[MIN_FREE_TRANSFERS, max_free]``. With no history —
    preseason, or a brand new entry — returns 1, which is what FPL gives you.
    """
    if not history_current:
        return MIN_FREE_TRANSFERS

    chip_by_event = {
        int(c.get("event", 0)): str(c.get("name", "")).lower()
        for c in (chips or [])
        if c.get("event") is not None
    }

    ft = MIN_FREE_TRANSFERS
    for entry in sorted(history_current, key=lambda e: int(e.get("event", 0))):
        event = int(entry.get("event", 0))
        chip = chip_by_event.get(event, "")
        used = 0 if chip in UNLIMITED_TRANSFER_CHIPS else int(
            entry.get("event_transfers", 0) or 0
        )
        # Spend this gameweek's transfers, then accrue next gameweek's allowance.
        ft = _clamp(ft - used, MIN_FREE_TRANSFERS - 1, max_free)
        ft = _clamp(ft + 1, MIN_FREE_TRANSFERS, max_free)

    return _clamp(ft, MIN_FREE_TRANSFERS, max_free)


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))
