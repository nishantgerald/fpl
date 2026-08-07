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


@pytest.fixture
def rows():
    """A pool with enough spread for the strategies to genuinely disagree.

    The flat `_rows()` helper gives every player the same ownership and a
    narrow price band, under which a differential squad and a max-points squad
    are the same fifteen — and a test that cannot tell them apart proves
    nothing about a selector whose whole job is that they differ.
    """
    out = []
    quotas = [("GKP", 8), ("DEF", 24), ("MID", 24), ("FWD", 16)]
    pid = 1
    for pos, count in quotas:
        for i in range(count):
            # Price and points rise together, so the expensive players really
            # are the best ones — as in the real data.
            price = 40 + (i % 10) * 15
            out.append({
                "id": pid,
                "web_name": f"{pos}{i}",
                "name": f"{pos} {i}",
                "position": pos,
                "team_id": (pid % 20) + 1,
                "team": f"T{(pid % 20) + 1}",
                "price_tenths": price,
                "price": price / 10,
                "value": 4.0 + (i % 10) * 2.5,
                "xpts_next": 2.0,
                "total_points": 50 + i,
                # The expensive ones are also the popular ones, which is what
                # makes "differential" a real constraint rather than a no-op.
                "selected_by_percent": 0.5 + (i % 10) * 4.0,
                "status": "a",
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


# ------------------------------------------------- the solver must actually ship


def test_the_exact_solver_is_reachable_in_this_environment():
    """scipy was installed locally and missing from requirements.txt.

    So every production build silently took the greedy fallback and served a
    squad worth 205 xPts that left £14m unspent, against a true optimum of 233
    that spends the lot. Nothing failed and nothing logged — the answer was
    simply 28 points worse, which is invisible unless you know what to compare
    it against.

    This asserts the dependency is present wherever the suite runs, so the
    absence is a red test rather than a quietly worse recommendation.
    """
    from engine import draft

    assert draft._solver_available(), (
        "scipy is missing, so draft.build falls back to the greedy pass"
    )


def test_the_optimum_spends_essentially_the_whole_budget(rows):
    """The tell that separates the optimum from the fallback.

    A greedy points-per-million pass buys cheap efficiency and banks the rest;
    an optimiser maximising points has no reason to leave money unspent. Any
    large remainder means the fallback ran.
    """
    from engine import draft

    built = draft.build(rows)

    assert built is not None
    assert built["remaining"] <= 1.0, (
        f"£{built['remaining']}m left unspent — that is the greedy fallback"
    )


def test_the_optimum_beats_the_fallback(monkeypatch):
    """Not merely different: at least as good, or the fallback would be fine.

    Uses the flat pool rather than the spread one, because greedy takes
    best-value-first and never backtracks — given a pool where the best value
    is also the most expensive, it spends into a corner and returns nothing at
    all. That is a real property of the fallback, but it is not what this test
    is about.
    """
    from engine import draft

    pool = _rows()
    exact = draft.build(pool)
    monkeypatch.setattr(draft, "_solve_exact", lambda *a, **k: None)
    greedy = draft.build(pool)

    assert greedy is not None
    assert exact["xi_projected"] >= greedy["xi_projected"]


# ---------------------------------------------------------------- strategies


def test_every_strategy_returns_a_legal_squad(rows):
    from engine import draft, rules

    for name in draft.STRATEGIES:
        built = draft.build(rows, strategy=name)
        assert built is not None, f"{name} produced no squad"
        assert len(built["squad"]) == 15, name
        assert len(built["starting_xi"]) == 11, name
        assert built["cost"] <= 100.0, name
        counts = {}
        for player in built["squad"]:
            counts[player["position"]] = counts.get(player["position"], 0) + 1
        assert counts == draft.SQUAD_QUOTAS, f"{name} broke the quotas: {counts}"


def test_the_strategies_actually_disagree(rows):
    """If they returned the same fifteen the selector would be decoration —
    the same complaint the Rebuild button earned."""
    from engine import draft

    squads = {
        name: frozenset(p["id"] for p in draft.build(rows, strategy=name)["squad"])
        for name in draft.STRATEGIES
    }

    assert len(set(squads.values())) > 1, "every strategy returned the same squad"


def test_max_points_is_the_highest_scoring_strategy(rows):
    """The others trade points for something — money kept back, low ownership,
    no premiums. None of them should beat the one maximising points."""
    from engine import draft

    best = draft.build(rows, strategy="max_points")["xi_projected"]
    for name in draft.STRATEGIES:
        if name == "max_points":
            continue
        assert draft.build(rows, strategy=name)["xi_projected"] <= best + 0.01, name


def test_value_deliberately_banks_money(rows):
    """It optimises points per million, so unspent budget is the intended
    outcome rather than the bug it is under max_points."""
    from engine import draft

    assert draft.build(rows, strategy="value")["remaining"] > 0


def test_differential_only_picks_low_ownership_players(rows):
    from engine import draft

    built = draft.build(rows, strategy="differential")
    for player in built["squad"]:
        assert player["selected_by_percent"] < draft.DIFFERENTIAL_MAX_OWNERSHIP


def test_no_premiums_keeps_every_player_under_the_cap(rows):
    from engine import draft

    built = draft.build(rows, strategy="balanced")
    for player in built["squad"]:
        assert player["price_tenths"] <= draft.BALANCED_MAX_PRICE_TENTHS


def test_an_unknown_strategy_falls_back_rather_than_raising(rows):
    """It arrives from a query string, so it is user input."""
    from engine import draft

    built = draft.build(rows, strategy="nonsense")

    assert built is not None
    assert built["strategy"] == draft.DEFAULT_STRATEGY
