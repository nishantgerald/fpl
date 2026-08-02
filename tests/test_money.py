"""Money: integer tenths, and the selling-price estimate.

Budget is the one place a float rounding error turns into a recommendation the
user physically cannot execute, so these tests assert on integers throughout.
"""

import pytest

from engine import money
from tests.conftest import make_element


def test_tenths_to_str():
    assert money.tenths_to_str(85) == "8.5"
    assert money.tenths_to_str(100) == "10.0"
    assert money.tenths_to_str(4) == "0.4"
    assert money.tenths_to_str(-3) == "-0.3"


def test_no_aggregate_falls_back_to_current_price():
    squad = [make_element(i, 3, 1, now_cost=50 + i) for i in range(1, 16)]
    prices = money.estimate_selling_prices(squad, None)
    assert prices == {p["id"]: p["now_cost"] for p in squad}


def test_squad_with_no_risers_sells_at_current_price():
    squad = [make_element(i, 3, 1, now_cost=50, cost_change_start=0) for i in range(1, 16)]
    total = sum(p["now_cost"] for p in squad)
    prices = money.estimate_selling_prices(squad, total)
    assert set(prices.values()) == {50}
    assert money.selling_price_confidence(squad, total) == "high"


def test_tax_is_distributed_and_the_aggregate_is_exact():
    # Five players have risen 4 (so owe up to 2 tax each); the rest are flat.
    squad = [
        make_element(i, 3, 1, now_cost=54, cost_change_start=4) for i in range(1, 6)
    ] + [
        make_element(i, 3, 1, now_cost=50, cost_change_start=0) for i in range(6, 16)
    ]
    total_now = sum(p["now_cost"] for p in squad)
    # Manager's real selling total is 6 tenths below cost.
    selling_total = total_now - 6

    prices = money.estimate_selling_prices(squad, selling_total)

    assert sum(prices.values()) == selling_total, "aggregate must be exact"
    for player in squad:
        pid, now_cost = player["id"], player["now_cost"]
        assert prices[pid] <= now_cost, "never sells above current price"
        if player["cost_change_start"] == 0:
            assert prices[pid] == now_cost, "a player who never rose owes no tax"


def test_per_player_tax_never_exceeds_the_rule_maximum():
    squad = [
        make_element(i, 3, 1, now_cost=60, cost_change_start=10) for i in range(1, 16)
    ]
    total_now = sum(p["now_cost"] for p in squad)
    # An absurd claimed tax — far more than the rules permit.
    prices = money.estimate_selling_prices(squad, total_now - 500)
    for player in squad:
        max_tax = player["cost_change_start"] // 2
        assert player["now_cost"] - prices[player["id"]] <= max_tax


def test_estimate_is_deterministic():
    squad = [
        make_element(i, 3, 1, now_cost=50 + i, cost_change_start=i % 5)
        for i in range(1, 16)
    ]
    total_now = sum(p["now_cost"] for p in squad)
    first = money.estimate_selling_prices(squad, total_now - 4)
    for _ in range(20):
        assert money.estimate_selling_prices(squad, total_now - 4) == first


def test_transfer_spend_uses_selling_price_not_current_price():
    out_player = make_element(1, 3, 1, now_cost=80)
    in_player = make_element(2, 3, 2, now_cost=85)
    # Sells for 78, not 80 — the 0.2m sell-on tax is real money.
    spend = money.transfer_spend([in_player], [out_player], {1: 78})
    assert spend == 7
    assert isinstance(spend, int)


def test_transfer_spend_can_be_negative_when_downgrading():
    out_player = make_element(1, 3, 1, now_cost=100)
    in_player = make_element(2, 3, 2, now_cost=60)
    assert money.transfer_spend([in_player], [out_player], {1: 98}) == -38


@pytest.mark.parametrize(
    "spend,bank,ok",
    [(0, 0, True), (5, 5, True), (6, 5, False), (-10, 0, True), (1, 0, False)],
)
def test_affordability_is_integer_arithmetic(spend, bank, ok):
    assert money.is_affordable(spend, bank) is ok


def test_confidence_reports_low_without_an_aggregate():
    squad = [make_element(i, 3, 1) for i in range(1, 16)]
    assert money.selling_price_confidence(squad, None) == "low"
