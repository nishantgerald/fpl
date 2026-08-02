"""Free-transfer derivation.

FPL doesn't publish this on any unauthenticated endpoint, so we reconstruct it
from transfer history. These tests pin the rule as of 2024/25: 1 to start, +1 a
week, capped at 5, and wildcard/free-hit weeks consume none.
"""

from engine.free_transfers import derive_free_transfers


def _history(transfers_per_gw, start=1):
    return [
        {"event": start + i, "event_transfers": n}
        for i, n in enumerate(transfers_per_gw)
    ]


def test_no_history_means_one_free_transfer():
    assert derive_free_transfers(None) == 1
    assert derive_free_transfers([]) == 1


def test_one_transfer_a_week_stays_at_one():
    assert derive_free_transfers(_history([1, 1, 1, 1, 1])) == 1


def test_saving_accrues_one_a_week():
    assert derive_free_transfers(_history([0])) == 2
    assert derive_free_transfers(_history([0, 0])) == 3
    assert derive_free_transfers(_history([0, 0, 0])) == 4


def test_accrual_caps_at_five():
    assert derive_free_transfers(_history([0] * 12)) == 5


def test_hits_do_not_push_the_bank_negative():
    # Four transfers on one free transfer is a -12 hit, but you still get 1 back.
    assert derive_free_transfers(_history([4])) == 1


def test_spending_a_saved_bank_leaves_the_remainder():
    # Save to 3, then spend 2 -> 1 left, +1 accrual = 2.
    assert derive_free_transfers(_history([0, 0, 2])) == 2


def test_wildcard_transfers_consume_nothing_and_the_bank_carries_over():
    with_chip = derive_free_transfers(
        _history([0, 0, 9]), chips=[{"name": "wildcard", "event": 3}]
    )
    without_chip = derive_free_transfers(_history([0, 0, 9]))
    assert with_chip == 4, "saved transfers carry over a wildcard week"
    assert without_chip == 1
    assert with_chip > without_chip


def test_free_hit_behaves_like_wildcard():
    assert derive_free_transfers(
        _history([0, 5]), chips=[{"name": "freehit", "event": 2}]
    ) == 3


def test_bench_boost_does_not_exempt_transfers():
    assert derive_free_transfers(
        _history([0, 3]), chips=[{"name": "bboost", "event": 2}]
    ) == 1


def test_history_order_does_not_matter():
    forwards = _history([0, 0, 1])
    assert derive_free_transfers(forwards) == derive_free_transfers(
        list(reversed(forwards))
    )


def test_result_is_always_in_range():
    for pattern in ([0] * 40, [9] * 20, [1, 0, 3, 0, 0, 2], []):
        result = derive_free_transfers(_history(pattern))
        assert 1 <= result <= 5
