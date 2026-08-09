"""Deep links into the client app.

The client routes on the path, so /app/players is a page to a reader and a
missing file to this server. It answered with Flask's 404 — meaning a bookmark,
a refresh, or a link someone was sent all landed on "Not Found", and the app
only held together if you entered at /app/ and never reloaded.

The fix has one edge worth pinning: a *missing asset* must still 404. Handing
back index.html for a dead .js turns a clean miss into a syntax error thrown
somewhere else entirely, which is a far worse thing to debug than a 404.
"""

import pytest

import app as flask_app


@pytest.fixture
def client():
    flask_app.app.config.update(TESTING=True)
    return flask_app.app.test_client()


def _is_app_shell(response) -> bool:
    body = response.get_data(as_text=True)
    return "<html" in body.lower() and "flutter" in body.lower()


@pytest.mark.parametrize(
    "path",
    [
        "/app/players",
        "/app/team",
        "/app/transfers",
        "/app/actions",
        "/app/leagues",
        "/app/analytics",
    ],
)
def test_a_deep_link_serves_the_app_rather_than_a_404(client, path):
    response = client.get(path)

    assert response.status_code == 200, f"{path} must not 404 on a refresh"
    assert _is_app_shell(response), f"{path} must return the app shell"


def test_a_nested_route_works_too(client):
    """Player detail is /app/players/<id>, which is two segments deep."""
    response = client.get("/app/players/427")

    assert response.status_code == 200
    assert _is_app_shell(response)


def test_the_front_door_is_unchanged(client):
    response = client.get("/app/")

    assert response.status_code == 200
    assert _is_app_shell(response)


def test_a_real_asset_is_still_served_as_itself(client):
    """The fallback must not shadow the bundle."""
    response = client.get("/app/main.dart.js")

    assert response.status_code == 200
    assert not _is_app_shell(response)
    assert "javascript" in response.headers["Content-Type"].lower()


@pytest.mark.parametrize(
    "path",
    [
        "/app/main.dart.js.map",
        "/app/assets/does-not-exist.png",
        "/app/nope.js",
    ],
)
def test_a_missing_asset_still_404s(client, path):
    """An extension means a file was meant, and a missing file is a 404."""
    response = client.get(path)

    assert response.status_code == 404, (
        f"{path} looks like an asset; answering it with HTML hides the miss"
    )
