"""Rebuild a manager's *current* squad from public data alone.

The public picks endpoint is a photograph of the squad at the last deadline.
But transfers are not pending trades — they are instant and irreversible the
moment they're confirmed, and the public transfers endpoint records them. So
mid-week, the squad a manager actually owns is::

    picks(last deadline)  +  transfers with event > that gameweek

applied in the order they were made. The bank comes out exact, because every
transfer record carries the prices paid and received at transaction time.

What this cannot know is next gameweek's *choices* — captain, bench order,
chip — which stay private until the deadline. Ownership is fact; the lineup is
last week's, carried forward. Callers flag the result accordingly.

Pure: payloads in, payload out. The one deliberate impurity is absent — no
clock, no I/O — so a season's worth of edge cases can be tested in memory.
"""

from __future__ import annotations

from typing import Mapping, Sequence


def selling_price(purchase: int, now: int) -> int:
    """FPL's sell rule, in price tenths: half the profit, floored per 0.1m.

    A player bought at 8.0 now worth 8.3 sells for 8.1 — the site keeps the
    odd tenth. A player who fell sells at the fallen price; losses are not
    halved.
    """
    if now <= purchase:
        return now
    return purchase + (now - purchase) // 2


def purchase_prices(
    squad: Sequence[int],
    transfers: Sequence[Mapping],
    elements: Mapping[int, Mapping],
) -> dict[int, int]:
    """Best-knowledge purchase price (tenths) for each squad member.

    A player acquired by transfer cost what the (latest) transfer record says.
    Anyone else has been held since GW1, and every GW1 squad was bought at the
    season's starting price — which the bootstrap still carries as
    ``now_cost - cost_change_start``.
    """
    bought: dict[int, int] = {}
    # Upstream lists transfers newest-first; walk oldest-first so a player
    # bought, sold and re-bought ends at the most recent purchase price.
    for transfer in reversed(list(transfers or [])):
        bought[int(transfer["element_in"])] = int(transfer.get("element_in_cost") or 0)

    prices: dict[int, int] = {}
    for element_id in squad:
        if element_id in bought:
            prices[element_id] = bought[element_id]
        else:
            element = elements.get(element_id, {})
            now = int(element.get("now_cost") or 0)
            prices[element_id] = now - int(element.get("cost_change_start") or 0)
    return prices


def reconstruct(
    base: Mapping,
    transfers: Sequence[Mapping],
    gameweek: int,
) -> dict | None:
    """Apply post-deadline transfers to a picks payload.

    ``base`` is the picks payload the pending transfers act on; ``gameweek``
    is the *current* gameweek, which is the filter boundary — only transfers
    with a later event are pending. The two differ under a Free Hit, where the
    base is one gameweek older (see :func:`base_gameweek`) but the boundary
    stays current so the Free Hit's own reverted transfers are excluded.

    Returns a new picks-shaped payload, or ``None`` when nothing is pending —
    the caller keeps the official payload, which also keeps ``reconstructed``
    honest: it is only ever set when something was actually rebuilt.
    """
    pending = [
        t for t in (transfers or []) if int(t.get("event") or 0) > gameweek
    ]
    if not pending or not base or not base.get("picks"):
        return None

    # Oldest first: a player bought and then sold again mid-week must pass
    # through the squad, not miss it.
    pending.sort(key=lambda t: str(t.get("time") or ""))

    picks = [dict(p) for p in base["picks"]]
    slots = {int(p["element"]): p for p in picks}
    bank = int((base.get("entry_history") or {}).get("bank") or 0)

    for transfer in pending:
        out_id = int(transfer["element_out"])
        in_id = int(transfer["element_in"])
        bank += int(transfer.get("element_out_cost") or 0)
        bank -= int(transfer.get("element_in_cost") or 0)

        slot = slots.pop(out_id, None)
        if slot is None:
            # Upstream disagreement (e.g. a chip we haven't modelled). The
            # money math stands; the swap is dropped rather than guessed at.
            continue
        slot["element"] = in_id
        # The slot is inherited; the armband is not. A new signing has never
        # been anyone's captain.
        if slot.get("is_captain"):
            slot["is_captain"] = False
            if slot.get("multiplier"):
                slot["multiplier"] = 1
        if slot.get("is_vice_captain"):
            slot["is_vice_captain"] = False
        slots[in_id] = slot

    # If the captain was sold, the vice inherits — which is what the game
    # itself does when a captain is absent.
    if not any(p.get("is_captain") for p in picks):
        vice = next((p for p in picks if p.get("is_vice_captain")), None)
        if vice is not None:
            vice["is_captain"] = True
            if vice.get("multiplier"):
                vice["multiplier"] = 2

    return {
        "picks": picks,
        "entry_history": {
            "bank": bank,
            # Squad value is price-dependent and drifts nightly; the base
            # figure is the last authoritative one and refreshes at the next
            # deadline. Bank, by contrast, is exact.
            "value": (base.get("entry_history") or {}).get("value"),
        },
        "active_chip": None,
        "source": "reconstructed",
        "transfers_applied": len(pending),
    }


def base_gameweek(gameweek: int, chips: Sequence[Mapping]) -> int:
    """Which gameweek's picks the pending transfers act on.

    Normally the current one. But a Free Hit squad evaporates when its
    gameweek ends — the manager gets last week's team back — so transfers made
    after a Free Hit act on the gameweek *before* it. (Its transfers carry the
    Free Hit gameweek's event number, so the ``event > gameweek`` filter in
    :func:`reconstruct` already excludes them.)
    """
    for chip in chips or []:
        if chip.get("name") == "freehit" and int(chip.get("event") or 0) == gameweek:
            return gameweek - 1
    return gameweek
