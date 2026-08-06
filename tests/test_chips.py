"""Chip timing.

A chip is worth what it adds over doing nothing, and that is computable rather
than a matter of taste — so every property here is about the arithmetic being
honest and the windows being read from FPL rather than invented.

The window handling carries the most risk: two of each chip exist per season,
one per half, with different legal ranges. Getting that wrong means telling a
manager to play something that expired a fortnight ago, or that they have
spent something they still hold.
"""

import pytest

from engine import chips


BOOTSTRAP_CHIPS = [
    {"name": "wildcard", "start_event": 2, "stop_event": 19, "chip_type": "transfer"},
    {"name": "wildcard", "start_event": 20, "stop_event": 38, "chip_type": "transfer"},
    {"name": "freehit", "start_event": 2, "stop_event": 19, "chip_type": "transfer"},
    {"name": "freehit", "start_event": 20, "stop_event": 38, "chip_type": "transfer"},
    {"name": "bboost", "start_event": 1, "stop_event": 19, "chip_type": "team"},
    {"name": "bboost", "start_event": 20, "stop_event": 38, "chip_type": "team"},
    {"name": "3xc", "start_event": 1, "stop_event": 19, "chip_type": "team"},
    {"name": "3xc", "start_event": 20, "stop_event": 38, "chip_type": "team"},
    # Not a chip we advise on; must be ignored rather than crash.
    {"name": "manager", "start_event": 1, "stop_event": 38, "chip_type": "team"},
]


def _player(name, points):
    return {"web_name": name, "per_gameweek": [{"xpts": points}]}


# ---------------------------------------------------------------- windows


def test_windows_come_from_the_bootstrap_not_from_us():
    """Hardcoding the legal ranges is how an app recommends an expired chip."""
    windows = chips.windows(BOOTSTRAP_CHIPS)

    assert len(windows) == 8  # four chips, two halves
    wildcards = [w for w in windows if w["name"] == "wildcard"]
    assert [(w["start"], w["stop"]) for w in wildcards] == [(2, 19), (20, 38)]
    # Wildcard and Free Hit cannot be played in GW1; the team chips can.
    assert min(w["start"] for w in windows if w["name"] == "freehit") == 2
    assert min(w["start"] for w in windows if w["name"] == "bboost") == 1


def test_unknown_chips_are_ignored_rather_than_guessed_at():
    assert all(w["name"] != "manager" for w in chips.windows(BOOTSTRAP_CHIPS))


def test_spending_a_first_half_chip_leaves_the_second_half_copy():
    """The part managers most often get wrong."""
    windows = chips.windows(BOOTSTRAP_CHIPS)
    used = [{"name": "wildcard", "event": 8}]

    remaining = chips.available(windows, used, gameweek=25)
    wildcards = [c for c in remaining if c["name"] == "wildcard"]

    assert len(wildcards) == 1
    assert wildcards[0]["start"] == 20


def test_expired_chips_disappear_entirely():
    windows = chips.windows(BOOTSTRAP_CHIPS)

    remaining = chips.available(windows, [], gameweek=25)

    # Nothing from the first half survives past GW19.
    assert all(c["stop"] >= 25 for c in remaining)


def test_a_chip_not_yet_open_is_listed_but_not_playable():
    windows = chips.windows(BOOTSTRAP_CHIPS)

    remaining = chips.available(windows, [], gameweek=10)
    second_half = [c for c in remaining if c["start"] == 20]

    assert second_half
    assert all(c["playable_now"] is False for c in second_half)


# ---------------------------------------------------------------- values


def test_triple_captain_adds_one_captain_not_two():
    """The chip turns a double into a triple, so it is worth one more score."""
    squad = [_player("Haaland", 8.0), _player("Salah", 6.0)]

    value, name = chips.triple_captain_value(squad)

    assert value == 8.0
    assert name == "Haaland"


def test_bench_boost_is_worth_exactly_the_bench():
    bench = [_player("A", 3.0), _player("B", 2.5), _player("C", 1.0),
             _player("D", 0.5)]

    assert chips.bench_boost_value(bench) == 7.0


def test_free_hit_is_the_gap_to_the_best_possible_eleven():
    assert chips.free_hit_value(xi_points=30.0, best_possible_xi=52.0) == 22.0
    # Never negative: a squad already at the ceiling gains nothing.
    assert chips.free_hit_value(xi_points=55.0, best_possible_xi=52.0) == 0.0


def test_wildcard_value_discounts_the_transfers_you_would_make_anyway():
    """Charging free transfers against it is the difference between advice and
    salesmanship."""
    with_none = chips.wildcard_value(200.0, 240.0, free_transfers=0)
    with_two = chips.wildcard_value(200.0, 240.0, free_transfers=2)

    assert with_none == 40.0
    assert with_two == 32.0  # two transfers at ~4 points each
    assert chips.wildcard_value(240.0, 240.0, free_transfers=5) == 0.0


# ---------------------------------------------------------------- verdicts


def _chip(name, playable=True, expires_in=20):
    return {
        "name": name, "label": chips.CHIP_LABELS[name],
        "start": 1, "stop": 19, "playable_now": playable,
        "expires_in": expires_in,
    }


def test_a_good_week_says_play_it():
    verdict = chips.recommend(_chip("bboost"), value=20.0, gameweek=5)

    assert verdict["verdict"] == "play"
    assert "20 points" in verdict["action"]


def test_a_thin_week_says_hold_and_names_the_bar():
    """'Not yet' and 'no' are different advice, so the threshold is stated."""
    verdict = chips.recommend(_chip("bboost"), value=5.0, gameweek=5)

    assert verdict["verdict"] == "hold"
    assert "Hold" in verdict["action"]
    assert "12" in verdict["action"]  # the bar it did not clear


def test_an_expiring_chip_says_take_it_rather_than_lose_it():
    """A mediocre return beats an unused chip once the window closes."""
    verdict = chips.recommend(_chip("bboost", expires_in=2), value=6.0, gameweek=17)

    assert verdict["verdict"] == "expiring"
    assert verdict["urgent"] is True
    assert "rather than lose it" in verdict["action"]


def test_a_locked_chip_says_when_it_opens_and_is_never_urgent():
    verdict = chips.recommend(
        _chip("wildcard", playable=False, expires_in=1), value=40.0, gameweek=10
    )

    assert verdict["verdict"] == "locked"
    assert verdict["urgent"] is False
    assert "Unlocks" in verdict["action"]
