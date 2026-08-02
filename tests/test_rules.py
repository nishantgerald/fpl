"""Squad rules — the constraints the old LLM prompt only asked for in English."""

import pytest

from engine import rules
from tests.conftest import make_element


def test_legal_formations_are_the_eight_real_shapes():
    assert len(rules.LEGAL_FORMATIONS) == 8
    for d, m, f in rules.LEGAL_FORMATIONS:
        assert d + m + f == 10
        assert 3 <= d <= 5 and 2 <= m <= 5 and 1 <= f <= 3
    assert (3, 4, 3) in rules.LEGAL_FORMATIONS
    assert (5, 4, 1) in rules.LEGAL_FORMATIONS
    # 2 at the back and 6 in midfield are both illegal.
    assert (2, 5, 3) not in rules.LEGAL_FORMATIONS
    assert (3, 6, 1) not in rules.LEGAL_FORMATIONS


def test_quotas_come_from_bootstrap_not_hard_coding(element_types):
    assert rules.squad_quotas(element_types) == {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}


def test_quotas_fall_back_when_bootstrap_is_unreadable():
    assert rules.squad_quotas(None) == rules.DEFAULT_SQUAD_QUOTAS
    assert rules.squad_quotas([]) == rules.DEFAULT_SQUAD_QUOTAS


def test_club_limit_comes_from_game_settings():
    assert rules.max_per_club({"squad_team_limit": 3}) == 3
    assert rules.max_per_club(None) == 3


def test_valid_squad_passes_every_check(squad, element_types):
    verdict = rules.check_squad(squad, rules.squad_quotas(element_types), 3)
    assert verdict["all_ok"] is True
    assert all(verdict[k] for k in verdict)


def test_wrong_squad_size_is_rejected(squad, element_types):
    verdict = rules.check_squad(squad[:14], rules.squad_quotas(element_types), 3)
    assert verdict["squad_size_ok"] is False
    assert verdict["all_ok"] is False


def test_position_quota_violation_is_rejected(squad, element_types):
    # Swap a defender for a fourth forward.
    broken = [p for p in squad if p["id"] != squad[2]["id"]]
    broken.append(make_element(9001, 4, 19, now_cost=60))
    verdict = rules.check_squad(broken, rules.squad_quotas(element_types), 3)
    assert verdict["position_quotas_ok"] is False
    assert verdict["all_ok"] is False


def test_fourth_player_from_one_club_is_rejected(squad, element_types):
    quotas = rules.squad_quotas(element_types)
    target_team = squad[0]["team"]
    broken = list(squad)
    # Force a fourth from the same club, keeping the position counts intact.
    replaced = next(p for p in broken if p["team"] != target_team)
    broken = [p for p in broken if p["id"] != replaced["id"]]
    broken.append(
        make_element(9002, replaced["element_type"], target_team, now_cost=50)
    )
    verdict = rules.check_squad(broken, quotas, 3)
    assert verdict["club_limit_ok"] is False
    assert verdict["all_ok"] is False


def test_duplicate_player_is_rejected(squad, element_types):
    broken = squad[:-1] + [squad[0]]
    verdict = rules.check_squad(broken, rules.squad_quotas(element_types), 3)
    assert verdict["unique_players_ok"] is False


@pytest.mark.parametrize(
    "status,chance,expected",
    [
        ("a", None, True),
        ("d", 100, True),
        ("d", 75, True),
        ("d", 50, False),
        ("d", None, False),
        ("i", None, False),
        ("s", None, False),
        ("u", None, False),
        ("n", None, False),
    ],
)
def test_availability_gate_for_transfers_in(status, chance, expected):
    element = make_element(1, 3, 1, status=status, chance=chance)
    assert rules.is_available(element) is expected


def test_best_xi_respects_formation_limits(squad):
    # Make every forward worthless so the optimiser wants to bench them — it
    # still has to start at least one.
    scores = {int(p["id"]): 5.0 for p in squad}
    for p in squad:
        if rules.position_of(p) == "FWD":
            scores[int(p["id"])] = 0.0

    xi, shape, total = rules.best_xi(squad, scores)
    assert len(xi) == 11
    assert shape[2] >= 1, "must field at least one forward"
    assert shape[0] >= 3, "must field at least three defenders"
    assert sum(shape) == 10

    positions = [rules.position_of(next(p for p in squad if int(p["id"]) == pid))
                 for pid in xi]
    assert positions.count("GKP") == 1
    assert total == pytest.approx(50.0)


def test_best_xi_picks_the_highest_scorers_available(squad):
    scores = {int(p["id"]): float(i) for i, p in enumerate(squad)}
    xi, _shape, total = rules.best_xi(squad, scores)
    # Whatever the shape, we should never leave a startable higher scorer out
    # in favour of a lower one at the same position.
    for pid in xi:
        player = next(p for p in squad if int(p["id"]) == pid)
        position = rules.position_of(player)
        benched = [
            p for p in squad
            if int(p["id"]) not in xi and rules.position_of(p) == position
        ]
        for other in benched:
            assert scores[int(other["id"])] <= scores[pid]
    assert total > 0


def test_best_xi_is_deterministic(squad):
    scores = {int(p["id"]): 4.0 for p in squad}
    first = rules.best_xi(squad, scores)
    for _ in range(10):
        assert rules.best_xi(squad, scores) == first


def test_captain_doubles_the_top_scorer(squad):
    scores = {int(p["id"]): 1.0 for p in squad}
    star = next(p for p in squad if rules.position_of(p) == "MID")
    scores[int(star["id"])] = 12.0

    _plain_xi, _plain_shape, plain = rules.best_xi(squad, scores)
    xi, _shape, with_captain, captain = rules.best_xi_with_captain(squad, scores)
    assert captain == int(star["id"])
    assert with_captain == pytest.approx(plain + 12.0)
    assert int(star["id"]) in xi


def test_can_field_legal_xi():
    assert rules.can_field_legal_xi({"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3})
    assert not rules.can_field_legal_xi({"GKP": 0, "DEF": 5, "MID": 5, "FWD": 3})
    assert not rules.can_field_legal_xi({"GKP": 2, "DEF": 2, "MID": 5, "FWD": 3})
    assert not rules.can_field_legal_xi({"GKP": 2, "DEF": 5, "MID": 5, "FWD": 0})


@pytest.mark.parametrize(
    "transfers,free,expected",
    [(0, 1, 0), (1, 1, 0), (2, 1, 4), (3, 1, 8), (2, 2, 0), (5, 5, 0), (6, 5, 4)],
)
def test_hit_cost(transfers, free, expected):
    assert rules.hit_cost(transfers, free) == expected
