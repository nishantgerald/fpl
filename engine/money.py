"""Money handling for FPL squads.

All money in this engine is an ``int`` in **tenths of a million** — FPL's own
``now_cost`` unit. Nothing here returns a float. Budget feasibility is the one
place where a floating-point rounding error would produce a recommendation the
user cannot actually execute, so the type system does the work instead: 10.3 is
not representable in binary, 103 is.

The hard problem is selling price. FPL's sell-on rule is::

    sell = purchase_price + floor((now_cost - purchase_price) / 2)   if risen
    sell = now_cost                                                  if fallen

and ``purchase_price`` only exists on the authenticated ``my-team/{id}/``
endpoint, which we deliberately do not use. :func:`estimate_selling_prices`
recovers it well enough from public data — see its docstring.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence


def tenths_to_str(tenths: int) -> str:
    """Render integer tenths as an FPL-style price string: ``85`` -> ``'8.5'``."""
    sign = "-" if tenths < 0 else ""
    tenths = abs(int(tenths))
    return f"{sign}{tenths // 10}.{tenths % 10}"


def estimate_selling_prices(
    squad: Sequence[Mapping],
    known_squad_selling_total: int | None,
) -> dict[int, int]:
    """Estimate each squad member's selling price, in tenths.

    ``squad`` is a sequence of bootstrap ``elements`` dicts (the manager's 15).
    ``known_squad_selling_total`` is the exact aggregate selling value of those
    15, derived by the caller as ``last_deadline_value - last_deadline_bank``
    from ``entry/{id}/`` — both fields are public.

    Method: the difference between what the squad costs now and what it sells
    for is the total sell-on tax ``D``. We know ``D`` exactly. We do not know how
    it splits across the 15, so we distribute it in proportion to each player's
    rise since the season start (``cost_change_start``), clamped per player to
    ``floor(cost_change_start / 2)`` — the most tax any player can possibly
    carry, since they cannot have been bought below the season-start price.

    Where the estimate is wrong it is wrong *conservatively*: a manager who
    bought a player mid-season above the start price gets over-taxed, which
    under-states their budget. Under-stating budget can only reject a legal
    transfer, never admit an illegal one. That is the correct direction to err.

    With no aggregate available (preseason, or an entry we couldn't read), falls
    back to ``now_cost`` for everyone, which is the true upper bound.
    """
    prices = {int(p["id"]): int(p.get("now_cost", 0)) for p in squad}
    if known_squad_selling_total is None or not prices:
        return prices

    total_now = sum(prices.values())
    tax_total = total_now - int(known_squad_selling_total)
    if tax_total <= 0:
        # Squad sells for at least what it costs — no sell-on tax to distribute.
        return prices

    # Per-player ceiling on tax, and the weights we split by.
    caps: dict[int, int] = {}
    for p in squad:
        pid = int(p["id"])
        rise = max(0, int(p.get("cost_change_start", 0)))
        caps[pid] = rise // 2

    cap_total = sum(caps.values())
    if cap_total <= 0:
        return prices

    # Cap the tax we try to distribute at what the caps can actually absorb;
    # any excess means our model of who rose is incomplete, and eating it is
    # better than distributing it onto players who provably owe none.
    distributable = min(tax_total, cap_total)

    assigned: dict[int, int] = {}
    remainder_order: list[tuple[float, int]] = []
    running = 0
    for pid, cap in caps.items():
        exact = distributable * cap / cap_total
        whole = int(exact)
        assigned[pid] = whole
        running += whole
        remainder_order.append((exact - whole, pid))

    # Hand out the rounding remainder deterministically: largest fractional part
    # first, ties broken by player id so the result never depends on dict order.
    remainder_order.sort(key=lambda t: (-t[0], t[1]))
    leftover = distributable - running
    for _, pid in remainder_order:
        if leftover <= 0:
            break
        if assigned[pid] < caps[pid]:
            assigned[pid] += 1
            leftover -= 1

    return {pid: prices[pid] - assigned.get(pid, 0) for pid in prices}


def selling_price_confidence(
    squad: Sequence[Mapping],
    known_squad_selling_total: int | None,
) -> str:
    """How much to trust :func:`estimate_selling_prices` — ``high``/``medium``/``low``.

    ``high`` when there is no tax to split at all (the estimate is then exact).
    ``low`` when we had no aggregate to anchor on. ``medium`` otherwise.
    """
    if known_squad_selling_total is None:
        return "low"
    total_now = sum(int(p.get("now_cost", 0)) for p in squad)
    tax_total = total_now - int(known_squad_selling_total)
    if tax_total <= 0:
        return "high"
    risers = sum(1 for p in squad if int(p.get("cost_change_start", 0)) > 0)
    return "medium" if risers else "low"


def transfer_spend(
    ins: Iterable[Mapping],
    outs: Iterable[Mapping],
    selling_prices: Mapping[int, int],
) -> int:
    """Net cost of a transfer set, in tenths. Negative means money freed up.

    Players in are bought at ``now_cost``; players out are sold at their
    (estimated) selling price.
    """
    cost_in = sum(int(p.get("now_cost", 0)) for p in ins)
    cost_out = sum(
        int(selling_prices.get(int(p["id"]), p.get("now_cost", 0))) for p in outs
    )
    return cost_in - cost_out


def is_affordable(spend: int, bank: int) -> bool:
    """Whether ``spend`` tenths can be covered by ``bank`` tenths. Integers only."""
    return int(spend) <= int(bank)
