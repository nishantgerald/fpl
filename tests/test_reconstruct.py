"""Rebuilding the current squad from last picks + public transfers.

The public picks payload freezes at the deadline, but transfers are instant
and irreversible. The properties worth pinning: money math is exact, the
armband never transfers to a new signing, a Free Hit's vanishing squad rolls
the base back a week, and when nothing is pending the caller keeps the
authoritative payload untouched.
"""

import pytest

from engine import reconstruct, service


def _base(picks=None, bank=5):
    return {
        "picks": picks
        or [
            {"element": 1, "position": 1, "multiplier": 1,
             "is_captain": False, "is_vice_captain": False},
            {"element": 2, "position": 2, "multiplier": 2,
             "is_captain": True, "is_vice_captain": False},
            {"element": 3, "position": 3, "multiplier": 1,
             "is_captain": False, "is_vice_captain": True},
            {"element": 4, "position": 12, "multiplier": 0,
             "is_captain": False, "is_vice_captain": False},
        ],
        "entry_history": {"bank": bank, "value": 1003},
    }


def _transfer(out_id, in_id, event, out_cost=74, in_cost=85, time="2026-09-01T10:00:00Z"):
    return {
        "element_out": out_id, "element_out_cost": out_cost,
        "element_in": in_id, "element_in_cost": in_cost,
        "event": event, "time": time,
    }


# ---------------------------------------------------------------- reconstruct


def test_nothing_pending_returns_none_so_the_official_payload_wins():
    already_applied = [_transfer(9, 1, event=3)]

    assert reconstruct.reconstruct(_base(), already_applied, gameweek=3) is None
    assert reconstruct.reconstruct(_base(), [], gameweek=3) is None


def test_a_pending_transfer_swaps_the_player_and_the_bank_exactly():
    result = reconstruct.reconstruct(
        _base(bank=5), [_transfer(1, 99, event=4, out_cost=74, in_cost=60)], gameweek=3
    )

    elements = [p["element"] for p in result["picks"]]
    assert 99 in elements and 1 not in elements
    assert result["entry_history"]["bank"] == 5 + 74 - 60
    assert result["source"] == "reconstructed"
    assert result["transfers_applied"] == 1


def test_chained_transfers_apply_in_the_order_they_were_made():
    """A buys B, then sells B for C — B passes through, C stays."""
    transfers = [
        # Upstream is newest-first; reconstruct must not depend on that.
        _transfer(99, 50, event=4, time="2026-09-02T10:00:00Z"),
        _transfer(1, 99, event=4, time="2026-09-01T10:00:00Z"),
    ]

    result = reconstruct.reconstruct(_base(), transfers, gameweek=3)

    elements = [p["element"] for p in result["picks"]]
    assert 50 in elements
    assert 99 not in elements and 1 not in elements


def test_a_new_signing_never_inherits_the_armband():
    """Sell the captain: the vice is promoted, the incoming player is neither."""
    result = reconstruct.reconstruct(
        _base(), [_transfer(2, 99, event=4)], gameweek=3
    )

    by_element = {p["element"]: p for p in result["picks"]}
    assert by_element[99]["is_captain"] is False
    assert by_element[99]["multiplier"] == 1
    assert by_element[3]["is_captain"] is True  # the vice, promoted
    assert by_element[3]["multiplier"] == 2


def test_an_unmatched_outgoing_player_keeps_the_money_but_skips_the_swap():
    result = reconstruct.reconstruct(
        _base(bank=0), [_transfer(777, 99, event=4, out_cost=50, in_cost=40)],
        gameweek=3,
    )

    assert result["entry_history"]["bank"] == 10
    assert 99 not in [p["element"] for p in result["picks"]]


# ---------------------------------------------------------------- money


def test_selling_price_halves_profit_and_floors_it():
    assert reconstruct.selling_price(80, 83) == 81   # +0.3 -> +0.1 kept
    assert reconstruct.selling_price(80, 84) == 82   # +0.4 -> +0.2
    assert reconstruct.selling_price(80, 80) == 80
    assert reconstruct.selling_price(80, 76) == 76   # losses are not halved


def test_purchase_prices_prefer_the_transfer_record_over_the_start_price():
    elements = {
        1: {"now_cost": 83, "cost_change_start": 3},   # held since GW1: 8.0
        99: {"now_cost": 90, "cost_change_start": 5},  # bought at 8.5
    }
    transfers = [_transfer(1, 99, event=4, in_cost=85)]

    prices = reconstruct.purchase_prices([1, 99], transfers, elements)

    assert prices[1] == 80
    assert prices[99] == 85


# ---------------------------------------------------------------- free hit


def test_free_hit_rolls_the_base_back_and_its_transfers_are_excluded():
    chips = [{"name": "freehit", "event": 7}]

    assert reconstruct.base_gameweek(7, chips) == 6
    assert reconstruct.base_gameweek(8, chips) == 8  # only its own week

    # The Free Hit's transfers carry event 7; with gameweek=7 as the boundary
    # they are not pending, so the reverted squad stays reverted.
    fh_transfers = [_transfer(1, 99, event=7)]
    assert reconstruct.reconstruct(_base(), fh_transfers, gameweek=7) is None


def test_a_wildcard_does_not_touch_the_base():
    assert reconstruct.base_gameweek(7, [{"name": "wildcard", "event": 7}]) == 7


# ---------------------------------------------------------------- service


def test_current_picks_skips_all_io_when_no_transfer_is_pending(monkeypatch):
    """The common case must cost one request, not four."""
    monkeypatch.setattr(
        service.fpl_client, "transfers", lambda entry_id: [_transfer(9, 1, event=3)]
    )
    fetched = []
    monkeypatch.setattr(
        service.fpl_client, "history", lambda entry_id: fetched.append("history")
    )

    assert service.current_picks(7, gameweek=3) is None
    assert fetched == []


def test_current_picks_folds_pending_transfers_into_the_last_locked_squad(
    monkeypatch,
):
    monkeypatch.setattr(
        service.fpl_client, "transfers", lambda entry_id: [_transfer(1, 99, event=4)]
    )
    monkeypatch.setattr(service.fpl_client, "history", lambda entry_id: {"chips": []})
    monkeypatch.setattr(
        service.fpl_client, "picks", lambda entry_id, gw: _base() if gw == 3 else None
    )

    result = service.current_picks(7, gameweek=3)

    assert 99 in [p["element"] for p in result["picks"]]
    assert result["source"] == "reconstructed"
