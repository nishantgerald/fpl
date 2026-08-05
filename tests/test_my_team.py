"""Reading the manager's squad before the first deadline.

The public picks endpoint 404s until the season starts, so `/api/team` is a 503
for the whole pre-season — the window in which a manager is actually building
their squad. `/my-team/` returns it, but needs a session cookie.

Two things have to hold. Without a cookie the app behaves exactly as it did
before, because most deployments will never have one. With an *expired* cookie
it must say so rather than quietly serving a cached squad forever.
"""

import pytest
import requests

import app as flask_app
from engine import fpl_client


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """No ambient cookie, no cross-test cache bleed."""
    monkeypatch.delenv("FPL_SESSION_COOKIE", raising=False)
    monkeypatch.setattr(fpl_client, "SESSION_FILE", tmp_path / "absent")
    fpl_client.clear_caches()
    yield
    fpl_client.clear_caches()


@pytest.fixture
def client():
    flask_app.app.config.update(TESTING=True)
    return flask_app.app.test_client()


def _my_team_payload():
    return {
        "picks": [
            {"element": 1, "position": 1, "is_captain": False,
             "is_vice_captain": False, "selling_price": 45, "purchase_price": 45},
            {"element": 2, "position": 2, "is_captain": True,
             "is_vice_captain": False, "selling_price": 80, "purchase_price": 80},
            {"element": 3, "position": 13, "is_captain": False,
             "is_vice_captain": False, "selling_price": 40, "purchase_price": 40},
        ],
        "chips": [
            {"name": "wildcard", "status_for_entry": "available"},
            {"name": "bboost", "status_for_entry": "active"},
        ],
        "transfers": {"bank": 5, "value": 1000, "limit": 1, "cost": 4, "made": 0},
    }


# ------------------------------------------------------------------ cookie


def test_no_cookie_is_not_an_error_it_is_just_absent():
    assert fpl_client.has_session() is False
    with pytest.raises(fpl_client.NotAuthenticated):
        fpl_client.my_team(1)


def test_a_cookie_file_is_read_fresh_so_replacing_it_needs_no_restart(
    monkeypatch, tmp_path
):
    path = tmp_path / "fpl-session"
    monkeypatch.setattr(fpl_client, "SESSION_FILE", path)

    assert fpl_client.has_session() is False
    path.write_text("sessionid=abc123\n")
    assert fpl_client.has_session() is True


def test_a_pasted_cookie_header_is_accepted_verbatim(monkeypatch, tmp_path):
    """Copy-as-cURL yields `Cookie: ...`; requiring hand-editing invites typos."""
    path = tmp_path / "fpl-session"
    path.write_text("Cookie: sessionid=abc123; pl_profile=xyz\n")
    monkeypatch.setattr(fpl_client, "SESSION_FILE", path)

    sent = {}

    def _capture(url, headers=None, timeout=None):
        sent["cookie"] = (headers or {}).get("Cookie")
        return _FakeResponse(_my_team_payload())

    monkeypatch.setattr(requests, "get", _capture)
    fpl_client.my_team(7)

    assert sent["cookie"] == "sessionid=abc123; pl_profile=xyz"


def test_the_env_var_wins_over_the_file(monkeypatch, tmp_path):
    path = tmp_path / "fpl-session"
    path.write_text("from-file")
    monkeypatch.setattr(fpl_client, "SESSION_FILE", path)
    monkeypatch.setenv("FPL_SESSION_COOKIE", "from-env")

    assert fpl_client._session_cookie() == "from-env"


# ------------------------------------------------------------- expiry


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)


def test_an_expired_cookie_raises_rather_than_serving_a_stale_squad(
    monkeypatch, tmp_path
):
    """The failure mode worth guarding: a squad frozen at whenever it expired."""
    path = tmp_path / "fpl-session"
    path.write_text("sessionid=expired")
    monkeypatch.setattr(fpl_client, "SESSION_FILE", path)
    monkeypatch.setattr(
        requests, "get", lambda *a, **k: _FakeResponse({"detail": "..."}, 403)
    )

    with pytest.raises(fpl_client.NotAuthenticated):
        fpl_client.my_team(7)


# ------------------------------------------------------------- reshaping


def test_bank_and_value_are_moved_to_where_the_route_looks_for_them():
    """my-team files them under `transfers`, picks under `entry_history`."""
    reshaped = fpl_client._as_picks_payload(_my_team_payload())

    assert reshaped["entry_history"]["bank"] == 5
    assert reshaped["entry_history"]["value"] == 1000
    assert reshaped["free_transfers"] == 1


def test_the_starting_eleven_is_derived_when_multiplier_is_absent():
    """Defaulting the missing field to 0 would bench the entire squad."""
    reshaped = fpl_client._as_picks_payload(_my_team_payload())
    by_element = {p["element"]: p for p in reshaped["picks"]}

    assert by_element[1]["multiplier"] == 1
    assert by_element[2]["multiplier"] == 2  # captain
    assert by_element[3]["multiplier"] == 0  # position 13 is the bench


def test_an_active_chip_is_found_among_the_available_ones():
    reshaped = fpl_client._as_picks_payload(_my_team_payload())

    assert reshaped["active_chip"] == "bboost"


def test_an_empty_squad_reshapes_to_none_rather_than_a_hollow_payload():
    assert fpl_client._as_picks_payload({"picks": []}) is None
    assert fpl_client._as_picks_payload(None) is None


# ------------------------------------------------------------------ route


def test_without_a_cookie_the_route_still_reports_season_not_started(
    monkeypatch, client
):
    """The existing behaviour must be untouched for deployments with no cookie."""
    monkeypatch.setattr(
        flask_app.fpl_client,
        "season_state",
        lambda: {"started": False, "gameweek": 1, "gameweek_name": "Gameweek 1",
                 "deadline": "2026-08-21T17:30:00Z"},
    )
    monkeypatch.setattr(flask_app.fpl_client, "entry", lambda entry_id: {"id": entry_id})

    response = client.get("/api/team?user_id=1")

    assert response.status_code == 503
    assert response.get_json()["code"] == "season_not_started"


def test_the_squad_is_flagged_provisional_before_the_deadline(monkeypatch, client):
    """A pre-deadline draft is still editable; presenting it as settled misleads."""
    monkeypatch.setattr(
        flask_app.fpl_client,
        "season_state",
        lambda: {"started": False, "gameweek": 1, "gameweek_name": "Gameweek 1",
                 "deadline": "2026-08-21T17:30:00Z"},
    )
    monkeypatch.setattr(flask_app.fpl_client, "entry", lambda entry_id: {"id": entry_id})
    monkeypatch.setattr(
        flask_app.fpl_client,
        "my_team",
        lambda entry_id, cookie=None: fpl_client._as_picks_payload(_my_team_payload()),
    )

    body = client.get("/api/team?user_id=1").get_json()

    assert body["provisional"] is True
    assert body["bank"] == 5


# ------------------------------------------------------- session health


def test_session_status_reports_absent_without_a_cookie():
    status = fpl_client.session_status(7)

    assert status["state"] == "absent"
    assert status["ok"] is False


def test_session_status_distinguishes_expiry_from_an_outage(monkeypatch, tmp_path):
    """Crying wolf on an FPL outage would train the alert to be ignored."""
    path = tmp_path / "fpl-session"
    path.write_text("sessionid=whatever")
    monkeypatch.setattr(fpl_client, "SESSION_FILE", path)

    monkeypatch.setattr(
        requests, "get", lambda *a, **k: _FakeResponse({"detail": "..."}, 403)
    )
    assert fpl_client.session_status(7)["state"] == "expired"

    def _boom(*a, **k):
        raise requests.ConnectionError("upstream down")

    monkeypatch.setattr(requests, "get", _boom)
    assert fpl_client.session_status(7)["state"] == "unknown"


def test_session_status_never_leaks_any_part_of_the_cookie(monkeypatch, tmp_path):
    secret = "sessionid=super-secret-value-123"
    path = tmp_path / "fpl-session"
    path.write_text(secret)
    monkeypatch.setattr(fpl_client, "SESSION_FILE", path)
    monkeypatch.setattr(
        requests, "get", lambda *a, **k: _FakeResponse(_my_team_payload())
    )

    rendered = repr(fpl_client.session_status(7))

    assert "super-secret" not in rendered
    assert "sessionid" not in rendered


def test_session_status_is_not_served_from_the_cache(monkeypatch, tmp_path):
    """A cached 'valid' would keep saying so for a minute after it stopped being."""
    path = tmp_path / "fpl-session"
    path.write_text("sessionid=whatever")
    monkeypatch.setattr(fpl_client, "SESSION_FILE", path)

    calls = []

    def _count(*a, **k):
        calls.append(1)
        return _FakeResponse(_my_team_payload())

    monkeypatch.setattr(requests, "get", _count)
    fpl_client.session_status(7)
    fpl_client.session_status(7)

    assert len(calls) == 2


def test_the_session_route_is_a_200_even_when_the_cookie_is_missing(client):
    """`ok: false` is a truthful report, not a failed request."""
    response = client.get("/api/session?user_id=7")

    assert response.status_code == 200
    assert response.get_json()["state"] == "absent"
