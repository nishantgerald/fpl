"""The public, server-rendered pages.

These exist because the Flutter bundle is invisible to a search engine — a
crawler fetching /app/ gets eleven characters of text. So the property under
test is not that the routes return 200, but that the HTML they return actually
*contains the projection*, in the markup, without JavaScript.

They must also never become a second model. Every figure is asserted against
`engine.service`, so a page and the app cannot quote different numbers for the
same player.
"""

import re

import pytest

import app as flask_app
from engine import content


@pytest.fixture
def client():
    flask_app.app.config.update(TESTING=True)
    return flask_app.app.test_client()


def _text(html: str) -> str:
    """Visible text only — what a crawler indexes."""
    stripped = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.S)
    return " ".join(re.sub(r"<[^>]+>", " ", stripped).split())


# ---------------------------------------------------------------- slugs


def test_accents_fold_rather_than_vanish():
    """`Guéhi` must not become `gu-hi`."""
    assert content.slugify("Marc Guéhi") == "marc-guehi"
    assert content.slugify("João Pedro") == "joao-pedro"
    assert content.slugify("Gabriel dos Santos Magalhães") == (
        "gabriel-dos-santos-magalhaes"
    )


def test_slugs_come_from_full_names_because_web_names_collide():
    """There are two Palmers in the bootstrap. A slug that points at whichever
    one iteration reached first is a URL that silently changes meaning."""
    palmer_a = {"id": 1, "first_name": "Cole", "second_name": "Palmer",
                "web_name": "Palmer"}
    palmer_b = {"id": 2, "first_name": "Tyrick", "second_name": "Palmer",
                "web_name": "Palmer"}

    index = content.build_slug_index([palmer_a, palmer_b])

    assert index["cole-palmer"] == 1
    assert index["tyrick-palmer"] == 2


def test_a_genuine_collision_keeps_the_incumbent_url_stable():
    """A namesake joining mid-season must not repoint an existing page."""
    first = {"id": 5, "first_name": "John", "second_name": "Smith"}
    later = {"id": 9, "first_name": "John", "second_name": "Smith"}

    index = content.build_slug_index([later, first])

    # Lower id keeps the bare slug regardless of input order.
    assert index["john-smith"] == 5
    assert index["john-smith-9"] == 9


# ---------------------------------------------------------------- verdict


def test_the_verdict_states_the_numbers_it_is_based_on():
    page = {
        "web_name": "Gabriel", "position": "DEF", "price": 8.0,
        "horizon_xpts": 23.4, "horizon": 5, "value": 2.9,
        "rank_in_position": 1, "position_total": 200, "ownership": 26.0,
        "minutes_risk": "low", "status": "a",
    }

    line = content.verdict(page)

    assert "23.4" in line and "£8.0m" in line
    assert "highest-projected DEF" in line
    assert "Check team news" in line


def test_an_unavailable_player_is_not_recommended_on_his_numbers():
    page = {
        "web_name": "Someone", "position": "FWD", "price": 9.0,
        "horizon_xpts": 0.0, "horizon": 5, "value": 0.0,
        "rank_in_position": 400, "position_total": 400, "ownership": 1.0,
        "minutes_risk": "high", "status": "i",
    }

    assert "unavailable" in content.verdict(page)


def test_rotation_risk_is_stated_rather_than_buried():
    page = {
        "web_name": "Someone", "position": "MID", "price": 5.0,
        "horizon_xpts": 18.0, "horizon": 5, "value": 3.6,
        "rank_in_position": 12, "position_total": 200, "ownership": 2.0,
        "minutes_risk": "high", "status": "a",
    }

    assert "not a guaranteed starter" in content.verdict(page)


# ---------------------------------------------------------------- pages


def test_every_public_page_renders(client):
    for path in ("/", "/projections", "/players", "/captain", "/fixtures"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert len(_text(response.get_data(as_text=True))) > 400, path


def test_a_player_page_contains_the_projection_in_the_markup(client):
    """The point of the exercise: no JavaScript required to read the numbers."""
    listing = client.get("/players").get_data(as_text=True)
    slug = re.search(r'/players/([a-z0-9-]+)"', listing).group(1)

    body = _text(client.get(f"/players/{slug}").get_data(as_text=True))

    # A gameweek table, with fixtures and figures, present as text.
    assert "gameweek by gameweek" in body.lower()
    assert re.search(r"GW\d+", body)
    assert "Where the points come from" in body


def test_pages_quote_the_same_numbers_as_the_api(client):
    """A page and the app must never disagree about one player."""
    listing = client.get("/players").get_data(as_text=True)
    slug = re.search(r'/players/([a-z0-9-]+)"', listing).group(1)
    page = client.get(f"/players/{slug}").get_data(as_text=True)

    element_id = flask_app._page_context()["slugs"][slug]
    api = client.get(f"/api/player/{element_id}/projection").get_json()

    assert f"{api['horizon_xpts']:.1f}" in _text(page)


def test_an_unknown_player_is_a_404_not_a_blank_page(client):
    assert client.get("/players/no-such-person").status_code == 404


def test_every_page_declares_a_canonical_url_and_a_description(client):
    for path in ("/", "/projections", "/captain", "/fixtures", "/players"):
        html = client.get(path).get_data(as_text=True)
        assert 'rel="canonical"' in html, path
        assert 'name="description"' in html, path
        # An empty description is worse than none: it tells a crawler the page
        # is about nothing.
        description = re.search(r'name="description" content="([^"]*)"', html)
        assert description and len(description.group(1)) > 40, path


def test_titles_are_distinct_so_pages_do_not_compete_with_each_other(client):
    titles = set()
    for path in ("/", "/projections", "/captain", "/fixtures", "/players"):
        html = client.get(path).get_data(as_text=True)
        titles.add(re.search(r"<title>(.*?)</title>", html, re.S).group(1))
    assert len(titles) == 5


# ---------------------------------------------------------------- discovery


def test_robots_allows_crawling_and_points_at_the_sitemap(client):
    body = client.get("/robots.txt").get_data(as_text=True)

    assert "Allow: /" in body
    assert "Disallow: /api/" in body
    assert "/sitemap.xml" in body


def test_the_sitemap_lists_the_tools_and_the_player_pages(client):
    body = client.get("/sitemap.xml").get_data(as_text=True)

    for path in ("/projections", "/captain", "/fixtures", "/players"):
        assert f"{path}<" in body, path
    assert body.count("<url>") > 20
    # Capped deliberately: 700 near-identical pages compete rather than rank.
    assert body.count("/players/") <= content.INDEXABLE_PLAYER_COUNT + 1


def test_the_app_is_still_served_and_is_not_indexed(client):
    """The SPA keeps working; it is simply not the thing crawlers read."""
    assert client.get("/app/").status_code == 200
