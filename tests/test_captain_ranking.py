"""Who the app puts the armband on, and why.

The armband is a *ceiling* decision. Doubling a score doubles its spread, so the
question is not "who scores most on average" but "who can win me the week" —
which is why the engine tilts toward the components that explode (goals,
assists, bonus) and away from the ones that cap out (appearance, clean sheet).

The public ranking did not do any of that. It sorted on plain next-gameweek
xPts and published it doubled, which looks like a captaincy metric and is not
one: doubling is monotonic, so it moves nobody. These tests exist because that
list and the signed-in list disagreed about the same gameweek.
"""

from engine import captain


def _projection(xpts, components, fixtures=None, availability=1.0, risk="low"):
    return {
        "xpts_next": xpts,
        "availability": availability,
        "minutes_risk": risk,
        "per_gameweek": [
            {
                "gameweek": 1,
                "xpts": xpts,
                "components": components,
                "fixtures": fixtures
                if fixtures is not None
                else [{"opponent": "COV", "home": True, "fdr": 2}],
            }
        ],
    }


def _element(element_id, name, element_type=4, owned=10.0, cost=80):
    return {
        "id": element_id,
        "web_name": name,
        "first_name": name,
        "second_name": "",
        "element_type": element_type,
        "team": 1,
        "now_cost": cost,
        "selected_by_percent": owned,
    }


TEAMS = {1: "ARS"}


def _rank(projections, elements, limit=10):
    return captain.rank_global(
        projections,
        elements,
        TEAMS,
        lambda e: {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}[e["element_type"]],
        lambda e: e["now_cost"] / 10,
        limit,
    )


def test_the_ceiling_beats_a_higher_mean():
    """The whole argument for a separate captain ranking.

    A defender on 5.2 whose points are a clean sheet cannot return 15; a striker
    on 5.0 who scores them can. Ranked on the mean the defender wins, and the
    manager is handed the armband choice that cannot win a gameweek.
    """
    projections = {
        1: _projection(5.2, {"clean_sheet": 3.6, "appearance": 1.6}),
        2: _projection(5.0, {"goals": 2.8, "assists": 0.9, "bonus": 0.6}),
    }
    elements = {
        1: _element(1, "Defender", element_type=2),
        2: _element(2, "Striker", element_type=4),
    }

    ranked = _rank(projections, elements)

    assert [r["web_name"] for r in ranked] == ["Striker", "Defender"]
    assert ranked[0]["ceiling"] == "high"
    assert ranked[1]["ceiling"] == "low"


def test_the_published_number_is_not_what_the_order_is_built_on():
    """Doubling is monotonic, so a list sorted by it is sorted by plain xPts.

    The figure is kept because "what do I get" is a real question; it is the
    ranking that must not use it.
    """
    projections = {
        1: _projection(5.2, {"clean_sheet": 3.6, "appearance": 1.6}),
        2: _projection(5.0, {"goals": 2.8, "assists": 0.9, "bonus": 0.6}),
    }
    elements = {
        1: _element(1, "Defender", element_type=2),
        2: _element(2, "Striker", element_type=4),
    }

    ranked = _rank(projections, elements)

    assert ranked[0]["xpts_captained"] == 10.0
    assert ranked[1]["xpts_captained"] == 10.4
    # The leader publishes the *smaller* doubled figure, which is only possible
    # because the order does not come from it.
    assert ranked[0]["xpts_captained"] < ranked[1]["xpts_captained"]


def test_the_public_ranking_agrees_with_the_signed_in_one():
    """Two lists recommending different captains for one gameweek, each calling
    its own pick the best, is the bug this shares code to prevent."""
    projections = {
        1: _projection(5.2, {"clean_sheet": 3.6, "appearance": 1.6}),
        2: _projection(5.0, {"goals": 2.8, "assists": 0.9, "bonus": 0.6}),
    }
    elements = {
        1: _element(1, "Defender", element_type=2),
        2: _element(2, "Striker", element_type=4),
    }
    squad = [
        {**elements[1], "selected_by_percent": 10.0},
        {**elements[2], "selected_by_percent": 10.0},
    ]

    public = _rank(projections, elements)
    personal = captain.rank_captains(
        squad, [], projections, {1: {"short_name": "ARS"}}, gameweek=1
    )

    assert [r["web_name"] for r in public] == [
        p["web_name"] for p in personal["picks"]
    ]


def test_a_blank_is_not_a_candidate_on_a_public_list():
    """Nothing forces the choice here, so a name nobody can captain is noise."""
    projections = {
        1: _projection(6.0, {"goals": 4.0}, fixtures=[]),
        2: _projection(4.0, {"goals": 2.0}),
    }
    elements = {1: _element(1, "Blanking"), 2: _element(2, "Playing")}

    assert [r["web_name"] for r in _rank(projections, elements)] == ["Playing"]


def test_an_unavailable_player_cannot_top_the_list():
    projections = {
        1: _projection(9.0, {"goals": 7.0}, availability=0.0),
        2: _projection(4.0, {"goals": 2.0}),
    }
    elements = {1: _element(1, "Injured"), 2: _element(2, "Fit")}

    assert _rank(projections, elements)[0]["web_name"] == "Fit"


def test_every_candidate_says_how_crowded_the_pick_is():
    """Captaining the template is a shield and captaining a differential is a
    swing. Both are defensible; which one you are doing is not optional."""
    projections = {
        1: _projection(5.0, {"goals": 3.0}),
        2: _projection(5.0, {"goals": 3.0}),
        3: _projection(5.0, {"goals": 3.0}),
    }
    elements = {
        1: _element(1, "Template", owned=55.0),
        2: _element(2, "Balanced", owned=9.0),
        3: _element(3, "Punt", owned=1.2),
    }

    bands = {r["web_name"]: r["safety"] for r in _rank(projections, elements)}

    assert bands == {
        "Template": "safe",
        "Balanced": "balanced",
        "Punt": "differential",
    }


def test_the_limit_is_applied_after_ranking_not_before():
    projections = {i: _projection(float(i), {"goals": float(i)}) for i in range(1, 9)}
    elements = {i: _element(i, f"P{i}") for i in range(1, 9)}

    ranked = _rank(projections, elements, limit=3)

    assert [r["web_name"] for r in ranked] == ["P8", "P7", "P6"]
