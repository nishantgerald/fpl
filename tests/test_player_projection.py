"""The per-gameweek breakdown route.

`horizon_xpts` is a sum over several gameweeks, which is exactly the sort of
number a user cannot check. This route is how they check it, so the contract
that matters is that the parts it hands back actually reconstruct the total.
"""

import pytest

import app as flask_app
from engine import service


@pytest.fixture
def client():
    flask_app.app.config.update(TESTING=True)
    return flask_app.app.test_client()


def _projection_set():
    per_gameweek = [
        {
            "gameweek": 1,
            "xpts": 4.66,
            "fixtures": [{"opponent": "COV", "home": True, "fdr": 2}],
            "components": {"appearance": 1.24, "goals": 3.42},
        },
        {
            "gameweek": 2,
            "xpts": 5.97,
            "fixtures": [{"opponent": "AVL", "home": False, "fdr": 4}],
            "components": {"appearance": 1.76, "goals": 4.21},
        },
    ]
    projections = {
        4: {
            "player_id": 4,
            "horizon_xpts": 10.63,
            "xpts_next": 4.66,
            "per_gameweek": per_gameweek,
        }
    }
    data = {"elements": [{"id": 4, "web_name": "Gabriel", "team": 1}], "teams": []}
    return service.Projection(projections, [1, 2], data, "xpts", "xpts")


@pytest.fixture
def _stub_projections(monkeypatch):
    monkeypatch.setattr(
        flask_app.service,
        "projections_for",
        lambda horizon, engine: _projection_set(),
    )


def test_the_breakdown_sums_to_the_headline_number(client, _stub_projections):
    """The whole point of the route: the parts must add up to the total shown."""
    body = client.get("/api/player/4/projection").get_json()

    assert body["horizon_xpts"] == pytest.approx(
        sum(g["xpts"] for g in body["per_gameweek"]), abs=0.01
    )


def test_every_gameweek_carries_its_fixture_and_components(client, _stub_projections):
    body = client.get("/api/player/4/projection").get_json()

    assert [g["gameweek"] for g in body["per_gameweek"]] == [1, 2]
    for gameweek in body["per_gameweek"]:
        assert gameweek["components"]
        # A blank is a legitimate empty list, but the key must always be there
        # so the client never has to guess between "blank" and "missing".
        assert "fixtures" in gameweek


def test_an_unknown_player_is_a_json_404_not_an_html_one(client, _stub_projections):
    """The client parses the body on failure; Flask's HTML 404 would break it."""
    response = client.get("/api/player/999999/projection")

    assert response.status_code == 404
    assert response.get_json()["code"] == "player_not_found"


def test_the_horizon_is_clamped_to_the_supported_range(client, monkeypatch):
    seen = []

    def _capture(horizon, engine):
        seen.append(horizon)
        return _projection_set()

    monkeypatch.setattr(flask_app.service, "projections_for", _capture)

    client.get("/api/player/4/projection?horizon=999")
    client.get("/api/player/4/projection?horizon=0")

    assert seen == [service.MAX_HORIZON, 1]


# ------------------------------------------------- the fixture run on a player


def _fixtures_for(projection):
    """`_player_payload` builds the photo URL from `request.host_url`, so it
    needs a request context even though nothing here is about routing."""
    with flask_app.app.test_request_context():
        return flask_app._player_payload(
            {"id": 1, "web_name": "Test", "status": "a", "element_type": 3},
            "ARS",
            projection,
        )["next_fixtures"]


def test_next_fixtures_carries_the_whole_run_not_one_gameweek():
    """The field is named for a run and every client treats it as one.

    It was built from `per_gameweek[0]["fixtures"]` — the fixtures of a single
    gameweek, which is normally exactly one match. The players list takes four
    of them and the comparison screen takes five, so both rendered a single
    chip and the fixture context those screens exist to give was simply absent.
    Nothing errored; there was just one chip where there should have been five.
    """
    projection = {
        "per_gameweek": [
            {"gameweek": 1, "fixtures": [{"opponent": "COV", "home": True, "fdr": 2}]},
            {"gameweek": 2, "fixtures": [{"opponent": "AVL", "home": False, "fdr": 4}]},
            {"gameweek": 3, "fixtures": [{"opponent": "CHE", "home": True, "fdr": 4}]},
            {"gameweek": 4, "fixtures": [{"opponent": "SUN", "home": False, "fdr": 3}]},
            {"gameweek": 5, "fixtures": [{"opponent": "BHA", "home": False, "fdr": 3}]},
        ],
    }

    fixtures = _fixtures_for(projection)

    assert [f["opponent"] for f in fixtures] == [
        "COV", "AVL", "CHE", "SUN", "BHA",
    ]


def test_a_double_gameweek_contributes_both_matches():
    """Flattening in gameweek order is what makes a double visible at all —
    the old code would have shown both of GW1's and none of anything else."""
    projection = {
        "per_gameweek": [
            {
                "gameweek": 1,
                "fixtures": [
                    {"opponent": "COV", "home": True, "fdr": 2},
                    {"opponent": "WOL", "home": False, "fdr": 2},
                ],
            },
            {"gameweek": 2, "fixtures": [{"opponent": "AVL", "home": False, "fdr": 4}]},
        ],
    }

    fixtures = _fixtures_for(projection)

    assert len(fixtures) == 3


def test_a_blank_gameweek_simply_contributes_nothing():
    """A team with no match that week drops out of the run rather than
    appearing as a placeholder — which is what a reader scanning it wants."""
    projection = {
        "per_gameweek": [
            {"gameweek": 1, "fixtures": [{"opponent": "COV", "home": True, "fdr": 2}]},
            {"gameweek": 2, "fixtures": []},
            {"gameweek": 3, "fixtures": [{"opponent": "CHE", "home": True, "fdr": 4}]},
        ],
    }

    fixtures = _fixtures_for(projection)

    assert [f["opponent"] for f in fixtures] == ["COV", "CHE"]


def test_each_fixture_says_which_gameweek_it_is():
    """Position in the run is not the gameweek, and a client that assumes it is
    will lay one player's GW3 beside another's GW4.

    A blank contributes nothing, so everything after it shifts left by one. Two
    players compared side by side then have their runs misaligned by exactly the
    number of blanks between them — silently, because both rows are full of
    plausible three-letter clubs."""
    projection = {
        "per_gameweek": [
            {"gameweek": 1, "fixtures": [{"opponent": "COV", "home": True, "fdr": 2}]},
            {"gameweek": 2, "fixtures": []},
            {"gameweek": 3, "fixtures": [{"opponent": "CHE", "home": True, "fdr": 4}]},
        ],
    }

    fixtures = _fixtures_for(projection)

    assert [f["gameweek"] for f in fixtures] == [1, 3]


def test_both_halves_of_a_double_carry_the_same_gameweek():
    """So a column can hold two chips rather than borrowing the next week's."""
    projection = {
        "per_gameweek": [
            {
                "gameweek": 7,
                "fixtures": [
                    {"opponent": "COV", "home": True, "fdr": 2},
                    {"opponent": "WOL", "home": False, "fdr": 2},
                ],
            },
        ],
    }

    assert [f["gameweek"] for f in _fixtures_for(projection)] == [7, 7]


def test_no_scheduled_matches_gives_an_empty_run_not_an_error():
    """Which is what lets the client say "No games scheduled" rather than
    rendering an empty strip that reads as a loading failure."""
    fixtures = _fixtures_for({"per_gameweek": []})

    assert fixtures == []
