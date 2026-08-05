"""Draft squad: legality, the seam it fills, and graceful degradation."""
import pytest
from engine import draft


def _rows(n=60):
    out = []
    quotas = [("GKP", 6), ("DEF", 18), ("MID", 18), ("FWD", 12)]
    pid = 1
    for pos, count in quotas:
        for i in range(count):
            out.append({
                "id": pid, "web_name": f"{pos}{i}", "name": f"{pos} {i}",
                "position": pos, "team_id": (pid % 20) + 1, "team": f"T{(pid%20)+1}",
                "price_tenths": 40 + (i % 8) * 10, "price": (40 + (i % 8) * 10) / 10,
                "value": 10.0 + (count - i), "xpts_next": 2.0,
                "total_points": 50 + i, "selected_by_percent": 1.0, "status": "a",
            })
            pid += 1
    return out


def test_the_squad_is_legal():
    built = draft.build(_rows())
    assert built is not None
    from collections import Counter
    squad = built["squad"]
    assert len(squad) == 15
    assert Counter(p["position"] for p in squad) == {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    assert max(Counter(p["team_id"] for p in squad).values()) <= draft.MAX_PER_CLUB
    assert sum(p["price_tenths"] for p in squad) <= draft.BUDGET_TENTHS


def test_the_starting_xi_is_a_legal_formation():
    built = draft.build(_rows())
    from collections import Counter
    counts = Counter(p["position"] for p in built["starting_xi"])
    assert sum(counts.values()) == 11
    assert counts["GKP"] == 1
    assert 3 <= counts["DEF"] <= 5
    assert 2 <= counts["MID"] <= 5
    assert 1 <= counts["FWD"] <= 3


def test_the_bench_is_the_squad_minus_the_xi():
    built = draft.build(_rows())
    ids = {p["id"] for p in built["squad"]}
    assert len(built["bench"]) == 4
    assert {p["id"] for p in built["starting_xi"]} | {p["id"] for p in built["bench"]} == ids


def test_the_objective_is_the_xi_not_the_fifteen():
    """Spending on the bench starves the pitch; the bench must be cheap."""
    built = draft.build(_rows())
    bench_cost = sum(p["price_tenths"] for p in built["bench"])
    xi_cost = sum(p["price_tenths"] for p in built["starting_xi"])
    assert bench_cost < xi_cost


def test_keepers_sit_last_on_the_bench():
    """An outfield substitution is far likelier than a keeper one."""
    built = draft.build(_rows())
    positions = [p["position"] for p in built["bench"]]
    if "GKP" in positions:
        assert positions[-1] == "GKP"


def test_a_pinned_player_always_appears():
    rows = _rows()
    # Something the optimiser would never take on value alone.
    rows[0]["value"] = 0.01
    rows[0]["price_tenths"] = 120
    built = draft.build(rows, pinned=[rows[0]["web_name"]])
    assert rows[0]["id"] in {p["id"] for p in built["squad"]}


def test_unavailable_players_are_never_candidates():
    elements = [
        {"id": 1, "web_name": "Fit", "element_type": 3, "team": 1, "now_cost": 50,
         "status": "a", "total_points": 100},
        {"id": 2, "web_name": "Hurt", "element_type": 3, "team": 1, "now_cost": 50,
         "status": "i", "total_points": 100},
    ]
    projections = {1: {"horizon_xpts": 20.0}, 2: {"horizon_xpts": 99.0}}
    rows = draft.candidates(elements, projections, {1: "AAA"})
    assert [r["web_name"] for r in rows] == ["Fit"]


def test_zero_projection_players_are_dropped():
    """Between seasons xpts scores a player with no minutes at zero."""
    elements = [{"id": 1, "web_name": "Ghost", "element_type": 3, "team": 1,
                 "now_cost": 50, "status": "a", "total_points": 0}]
    rows = draft.candidates(elements, {1: {"horizon_xpts": 0.0}}, {1: "AAA"})
    assert rows == []


def test_the_greedy_fallback_still_returns_a_legal_squad():
    """The web process is not required to carry SciPy."""
    from collections import Counter
    indices = draft._solve_greedy(_rows(), draft.BUDGET_TENTHS, set())
    assert indices is not None
    squad = [_rows()[i] for i in indices]
    assert len(squad) == 15
    assert Counter(p["position"] for p in squad) == {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}


def test_an_impossible_budget_returns_none_rather_than_an_illegal_squad():
    assert draft.build(_rows(), budget_tenths=100) is None
