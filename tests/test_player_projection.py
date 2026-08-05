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
