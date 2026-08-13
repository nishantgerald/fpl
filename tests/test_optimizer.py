"""The optimiser.

These are the tests that matter most: every recommendation must be legal, and
"legal" has to be re-checked from scratch, not taken on trust from the search.
"""

import pytest

from engine import money, optimizer, rules, xpts
from tests.conftest import make_element


@pytest.fixture
def projections(elements, fixtures, teams, events):
    return xpts.project_all(elements, fixtures, teams, events, 13, 5)


@pytest.fixture
def gameweeks():
    return [13, 14, 15, 16, 17]


def _optimise(squad, elements, projections, gameweeks, **kwargs):
    defaults = dict(
        bank=20,
        free_transfers=1,
        selling_prices={int(p["id"]): int(p["now_cost"]) for p in squad},
        quotas=rules.DEFAULT_SQUAD_QUOTAS,
        club_limit=3,
        max_transfers=2,
    )
    defaults.update(kwargs)
    return optimizer.optimise(
        squad=squad,
        all_elements=elements,
        projections=projections,
        gameweeks=gameweeks,
        **defaults,
    )


# ------------------------------------------------------------------ legality


def test_every_plan_is_a_legal_squad(squad, elements, projections, gameweeks):
    result = _optimise(squad, elements, projections, gameweeks)
    squad_ids = {int(p["id"]) for p in squad}
    by_id = {int(p["id"]): p for p in elements}

    for plan in result["plans"]:
        assert plan["legality"]["all_ok"] is True

        # Rebuild the squad independently of anything the engine reported.
        out_ids = {t["out"]["id"] for t in plan["transfers"]}
        in_ids = {t["in"]["id"] for t in plan["transfers"]}
        new_squad = [by_id[pid] for pid in (squad_ids - out_ids) | in_ids]

        verdict = rules.check_squad(new_squad, rules.DEFAULT_SQUAD_QUOTAS, 3)
        assert verdict["all_ok"] is True, plan["transfers"]
        assert len(new_squad) == 15


def test_never_sells_a_player_we_do_not_own(squad, elements, projections, gameweeks):
    squad_ids = {int(p["id"]) for p in squad}
    result = _optimise(squad, elements, projections, gameweeks)
    for plan in result["plans"]:
        for transfer in plan["transfers"]:
            assert transfer["out"]["id"] in squad_ids


def test_never_buys_a_player_we_already_own(squad, elements, projections, gameweeks):
    squad_ids = {int(p["id"]) for p in squad}
    result = _optimise(squad, elements, projections, gameweeks)
    for plan in result["plans"]:
        for transfer in plan["transfers"]:
            assert transfer["in"]["id"] not in squad_ids


def test_transfers_are_position_preserving(squad, elements, projections, gameweeks):
    result = _optimise(squad, elements, projections, gameweeks)
    for plan in result["plans"]:
        for transfer in plan["transfers"]:
            assert transfer["out"]["position"] == transfer["in"]["position"]


def test_never_buys_an_unavailable_player(squad, elements, projections, gameweeks):
    result = _optimise(squad, elements, projections, gameweeks)
    for plan in result["plans"]:
        for transfer in plan["transfers"]:
            assert transfer["in"]["status"] not in rules.UNAVAILABLE_STATUSES


def test_respects_the_club_limit_even_when_a_club_is_already_full(
    elements, projections, gameweeks
):
    """Build a squad already holding 3 from club 1, then check we never add a 4th."""
    by_team = {}
    for element in elements:
        by_team.setdefault(element["team"], []).append(element)

    # 3 midfielders from club 1, then fill legally from other clubs.
    squad = [p for p in by_team[1] if p["element_type"] == 3][:3]
    counts = {1: 2, 2: 5, 3: 5, 4: 3}
    counts[3] -= 3
    club_used = {1: 3}
    for element in elements:
        if element["team"] == 1:
            continue
        et = element["element_type"]
        if counts.get(et, 0) <= 0:
            continue
        if club_used.get(element["team"], 0) >= 3:
            continue
        squad.append(element)
        counts[et] -= 1
        club_used[element["team"]] = club_used.get(element["team"], 0) + 1

    assert rules.check_squad(squad, rules.DEFAULT_SQUAD_QUOTAS, 3)["all_ok"]

    result = _optimise(squad, elements, projections, gameweeks, bank=100)
    squad_ids = {int(p["id"]) for p in squad}
    by_id = {int(p["id"]): p for p in elements}
    for plan in result["plans"]:
        out_ids = {t["out"]["id"] for t in plan["transfers"]}
        in_ids = {t["in"]["id"] for t in plan["transfers"]}
        new_squad = [by_id[pid] for pid in (squad_ids - out_ids) | in_ids]
        assert max(rules.club_counts(new_squad).values()) <= 3


# ------------------------------------------------------------------ budget


def test_never_spends_more_than_the_bank(squad, elements, projections, gameweeks):
    for bank in (0, 3, 10, 50):
        result = _optimise(squad, elements, projections, gameweeks, bank=bank)
        for plan in result["plans"]:
            assert plan["spend"] <= bank
            assert plan["bank_after"] == bank - plan["spend"]
            assert plan["bank_after"] >= 0


def test_zero_bank_only_allows_cost_neutral_or_downgrade_moves(
    squad, elements, projections, gameweeks
):
    result = _optimise(squad, elements, projections, gameweeks, bank=0)
    for plan in result["plans"]:
        assert plan["spend"] <= 0


def test_all_money_is_integer(squad, elements, projections, gameweeks):
    result = _optimise(squad, elements, projections, gameweeks)
    for plan in result["plans"]:
        assert isinstance(plan["spend"], int)
        assert isinstance(plan["bank_after"], int)
        for transfer in plan["transfers"]:
            assert isinstance(transfer["in"]["now_cost"], int)
            assert isinstance(transfer["out"]["selling_price"], int)


def test_selling_price_not_current_price_drives_affordability(
    squad, elements, projections, gameweeks
):
    """A squad taxed to zero sale value can't afford anything."""
    broke = {int(p["id"]): 0 for p in squad}
    result = _optimise(
        squad, elements, projections, gameweeks, bank=0, selling_prices=broke
    )
    assert result["plans"] == []


# ------------------------------------------------------------------ hits


def test_hit_cost_is_exactly_four_per_extra_transfer(
    squad, elements, projections, gameweeks
):
    for free in (0, 1, 2):
        result = _optimise(
            squad, elements, projections, gameweeks, free_transfers=free, bank=100
        )
        for plan in result["plans"]:
            expected = 4 * max(0, plan["n_transfers"] - free)
            assert plan["hit_cost"] == expected
            assert plan["net_gain"] == pytest.approx(
                plan["gross_gain"] - expected, abs=0.01
            )


def test_include_hits_false_never_exceeds_free_transfers(
    squad, elements, projections, gameweeks
):
    result = _optimise(
        squad, elements, projections, gameweeks,
        free_transfers=1, include_hits=False, bank=100,
    )
    for plan in result["plans"]:
        assert plan["n_transfers"] <= 1
        assert plan["hit_cost"] == 0


def test_only_positive_net_gain_plans_are_returned(
    squad, elements, projections, gameweeks
):
    result = _optimise(squad, elements, projections, gameweeks)
    for plan in result["plans"]:
        assert plan["net_gain"] > 0


def test_plans_are_ranked_by_net_gain(squad, elements, projections, gameweeks):
    result = _optimise(squad, elements, projections, gameweeks, bank=100)
    gains = [p["net_gain"] for p in result["plans"]]
    assert gains == sorted(gains, reverse=True)
    assert [p["rank"] for p in result["plans"]] == list(range(1, len(gains) + 1))


# ------------------------------------------------------------------ hold


def test_hold_is_always_populated(squad, elements, projections, gameweeks):
    result = _optimise(squad, elements, projections, gameweeks)
    assert "hold" in result
    assert result["hold"]["reason"]
    assert result["hold"]["baseline_xpts"] == result["baseline_xpts"]


def test_max_transfers_zero_returns_hold(squad, elements, projections, gameweeks):
    result = _optimise(squad, elements, projections, gameweeks, max_transfers=0)
    assert result["plans"] == []
    assert result["hold"]["reason"]


def test_hold_cites_the_best_rejected_move_when_hits_kill_it(
    squad, elements, projections, gameweeks
):
    # Zero free transfers makes even a single move cost 4 points.
    result = _optimise(
        squad, elements, projections, gameweeks,
        free_transfers=0, max_transfers=1, bank=100,
    )
    if not result["plans"]:
        rejected = result["hold"]["best_rejected"]
        assert rejected is not None
        assert rejected["hit_cost"] == 4
        assert rejected["net_gain"] <= 0
        assert "Bank the transfer" in result["hold"]["reason"]


def test_hold_does_not_deny_the_plans_shown_beside_it(
    squad, elements, projections, gameweeks
):
    """The client renders the hold card under the plans, not instead of them.

    Before this, `best_rejected` was None whenever plans existed, so the reason
    fell through to "No transfer improves your projected points" — printed
    directly beneath a card recommending a +14 pt move.
    """
    result = _optimise(squad, elements, projections, gameweeks, bank=100)
    assert result["plans"], "fixture must produce plans for this to mean anything"

    reason = result["hold"]["reason"]
    assert "No transfer improves" not in reason
    # It has to name what holding costs, which is the top plan's gain.
    assert f"{result['plans'][0]['net_gain']:.1f} pts" in reason


def test_hold_still_says_nothing_improves_when_there_is_nothing_to_cite():
    """The original wording is correct in the case it was written for.

    That case is narrower than "no plans": with no plans there is usually still
    a rejected move worth naming. This is the genuinely empty one.
    """
    hold = optimizer._hold(
        baseline=50.0, detail=[], best_rejected=None, free_transfers=1,
        top_plan=None,
    )
    assert "No transfer improves" in hold["reason"]


# ------------------------------------------------------------------ determinism


def test_identical_inputs_produce_identical_output(
    squad, elements, projections, gameweeks
):
    first = _optimise(squad, elements, projections, gameweeks)
    for _ in range(5):
        assert _optimise(squad, elements, projections, gameweeks) == first


def test_squad_order_does_not_change_the_answer(
    squad, elements, projections, gameweeks
):
    forwards = _optimise(squad, elements, projections, gameweeks)
    backwards = _optimise(list(reversed(squad)), elements, projections, gameweeks)
    assert [p["transfers"] for p in forwards["plans"]] == [
        p["transfers"] for p in backwards["plans"]
    ]


# ------------------------------------------------------------------ quality


def test_candidate_pool_spans_the_price_range(elements, projections):
    """The old FCPS-truncated top-70 pool was all premiums by construction.

    A pool with no cheap options can never find the enabler that makes a
    premium affordable, so we require the pool to reach below the position's
    median price.
    """
    pool = optimizer._candidate_pool(elements, projections, {})
    for position in ("GKP", "DEF", "MID", "FWD"):
        universe = sorted(
            int(p["now_cost"])
            for p in elements
            if rules.position_of(p) == position
        )
        median = universe[len(universe) // 2]
        cheapest_in_pool = min(int(p["now_cost"]) for p in pool[position])
        assert cheapest_in_pool <= median, (
            f"{position} pool bottoms out at {cheapest_in_pool}, "
            f"median is {median} — no budget enablers available"
        )


def test_injured_squad_player_gets_transferred_out(
    elements, projections, gameweeks, teams, events, fixtures
):
    """An unavailable player projects zero, so the optimiser should move him on."""
    by_team = {}
    for element in elements:
        by_team.setdefault(element["team"], []).append(element)

    squad = []
    counts = {1: 2, 2: 5, 3: 5, 4: 3}
    club_used = {}
    for element in elements:
        et = element["element_type"]
        if counts.get(et, 0) <= 0 or club_used.get(element["team"], 0) >= 3:
            continue
        squad.append(dict(element))
        counts[et] -= 1
        club_used[element["team"]] = club_used.get(element["team"], 0) + 1

    casualty = next(p for p in squad if p["element_type"] == 3)
    casualty["status"] = "i"

    injured_projections = xpts.project_all(
        [p for p in elements if p["id"] != casualty["id"]] + [casualty],
        fixtures, teams, events, 13, 5,
    )
    result = _optimise(
        squad, elements, injured_projections, gameweeks, bank=100, max_transfers=1
    )
    assert result["plans"], "an injured player should always be worth moving"
    # Replacing the injured player and replacing the weakest starter at the same
    # position can tie on points, so both are legitimate top plans — but moving
    # the casualty on must at least be among the options offered.
    moved_out = {
        transfer["out"]["id"]
        for plan in result["plans"]
        for transfer in plan["transfers"]
    }
    assert casualty["id"] in moved_out


def test_every_plan_carries_reasons(squad, elements, projections, gameweeks):
    result = _optimise(squad, elements, projections, gameweeks, bank=100)
    for plan in result["plans"]:
        assert plan["reasons"]
        assert all(isinstance(reason, str) and reason for reason in plan["reasons"])


def test_per_gameweek_detail_covers_the_horizon(
    squad, elements, projections, gameweeks
):
    result = _optimise(squad, elements, projections, gameweeks, bank=100)
    for plan in result["plans"]:
        assert len(plan["per_gameweek"]) == len(gameweeks)
        for entry in plan["per_gameweek"]:
            assert entry["delta"] == pytest.approx(
                entry["after"] - entry["before"], abs=0.01
            )


def test_internal_keys_never_leak_into_the_response(
    squad, elements, projections, gameweeks
):
    result = _optimise(squad, elements, projections, gameweeks, bank=100)
    for plan in result["plans"]:
        assert not any(key.startswith("_") for key in plan)


def test_no_llm_in_the_decision_path():
    """The engine must not be able to reach a language model, even by accident."""
    import inspect

    source = inspect.getsource(optimizer) + inspect.getsource(rules)
    source += inspect.getsource(money) + inspect.getsource(xpts)
    for forbidden in ("openai", "OpenAI", "requests", "random.", "time."):
        assert forbidden not in source, f"{forbidden} must not appear in the engine"
