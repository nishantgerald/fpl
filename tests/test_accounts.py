"""App accounts: registration, sessions, FPL linking, and the cookie vault.

The properties that matter are the ones a breach or a bug would exploit:
passwords must never be recoverable from what we store, login failures must not
reveal which emails exist, session tokens must actually die on logout, and a
stored FPL cookie must never travel back out through any response.
"""

import pytest
import requests

import app as flask_app
from engine import accounts, fpl_client, llm_budget


@pytest.fixture(autouse=True)
def _fresh_db(monkeypatch, tmp_path):
    monkeypatch.setattr(accounts, "DB_PATH", tmp_path / "app.db")
    monkeypatch.setattr(accounts, "KEY_PATH", tmp_path / "vault.key")
    monkeypatch.delenv("FPL_APP_VAULT_KEY", raising=False)
    accounts.init_db()
    # Auth throttling is per-process state; a full window from one test must
    # not lock the next one out.
    # The throttle windows moved into llm_budget so they are shared across
    # gunicorn workers; this is the store now.
    llm_budget._local_windows.clear()
    fpl_client.clear_caches()
    yield
    fpl_client.clear_caches()


@pytest.fixture
def client():
    flask_app.app.config.update(TESTING=True)
    return flask_app.app.test_client()


def _register(client, email="nish@example.com", password="a-decent-password"):
    return client.post(
        "/api/auth/register", json={"email": email, "password": password}
    )


def _auth_header(response):
    return {"Authorization": f"Bearer {response.get_json()['token']}"}


# ---------------------------------------------------------------- passwords


def test_the_stored_password_is_a_hash_not_the_password():
    accounts.register("a@b.co", "hunter2hunter2")

    import sqlite3

    with sqlite3.connect(accounts.DB_PATH) as conn:
        stored = conn.execute("SELECT password_hash FROM users").fetchone()[0]

    assert "hunter2" not in stored
    assert stored.startswith(("scrypt:", "pbkdf2:"))


def test_login_failure_does_not_reveal_whether_the_email_exists(client):
    _register(client, email="real@example.com")

    wrong_password = client.post(
        "/api/auth/login",
        json={"email": "real@example.com", "password": "wrong-password"},
    )
    unknown_email = client.post(
        "/api/auth/login",
        json={"email": "ghost@example.com", "password": "wrong-password"},
    )

    assert wrong_password.status_code == unknown_email.status_code == 401
    assert wrong_password.get_json() == unknown_email.get_json()


def test_a_short_password_is_rejected(client):
    response = _register(client, password="short")

    assert response.status_code == 400
    assert response.get_json()["code"] == "weak_password"


def test_registering_the_same_email_twice_fails_cleanly(client):
    assert _register(client).status_code == 201
    response = _register(client)

    assert response.status_code == 400
    assert response.get_json()["code"] == "email_taken"


# ---------------------------------------------------------------- sessions


def test_a_token_works_until_logout_and_not_after(client):
    headers = _auth_header(_register(client))

    assert client.get("/api/auth/me", headers=headers).status_code == 200
    client.post("/api/auth/logout", headers=headers)
    assert client.get("/api/auth/me", headers=headers).status_code == 401


def test_an_expired_token_is_rejected_and_reaped(monkeypatch):
    accounts.register("a@b.co", "a-decent-password")
    token = accounts.authenticate("a@b.co", "a-decent-password")
    monkeypatch.setattr(accounts, "SESSION_TTL_SECONDS", -1)

    fresh = accounts.authenticate("a@b.co", "a-decent-password")

    assert accounts.user_for_token(fresh) is None  # born expired
    assert accounts.user_for_token(token) is not None  # minted under old TTL


def test_repeated_login_attempts_are_throttled(client):
    for _ in range(flask_app._AUTH_ATTEMPT_LIMIT):
        client.post(
            "/api/auth/login", json={"email": "x@y.co", "password": "nope-nope"}
        )

    response = client.post(
        "/api/auth/login", json={"email": "x@y.co", "password": "nope-nope"}
    )

    assert response.status_code == 429


# ---------------------------------------------------------------- FPL link


def test_linking_verifies_the_entry_actually_exists(monkeypatch, client):
    headers = _auth_header(_register(client))
    monkeypatch.setattr(flask_app.fpl_client, "entry", lambda entry_id: None)

    response = client.post(
        "/api/me/fpl", json={"entry_id": 999999999}, headers=headers
    )

    assert response.status_code == 404


def test_a_linked_entry_appears_on_me(monkeypatch, client):
    headers = _auth_header(_register(client))
    monkeypatch.setattr(
        flask_app.fpl_client,
        "entry",
        lambda entry_id: {
            "id": entry_id,
            "name": "__here_to_win__",
            "player_first_name": "Nishant",
            "player_last_name": "Gerald",
        },
    )

    client.post("/api/me/fpl", json={"entry_id": 2670555}, headers=headers)
    body = client.get("/api/auth/me", headers=headers).get_json()

    assert body["fpl"]["entry_id"] == 2670555
    assert body["fpl"]["team_name"] == "__here_to_win__"
    assert body["fpl_connected"] is False


# ---------------------------------------------------------------- cookie vault


def test_the_cookie_is_encrypted_at_rest():
    accounts.register("a@b.co", "a-decent-password")
    accounts.store_cookie(1, "sessionid=super-secret")

    import sqlite3

    with sqlite3.connect(accounts.DB_PATH) as conn:
        blob = conn.execute("SELECT ciphertext FROM fpl_cookies").fetchone()[0]

    assert b"super-secret" not in blob
    assert accounts.cookie_for(1) == "sessionid=super-secret"


def test_a_rotated_vault_key_reads_as_absent_not_a_crash(tmp_path):
    accounts.register("a@b.co", "a-decent-password")
    accounts.store_cookie(1, "sessionid=old")

    accounts.KEY_PATH.unlink()  # rotates: next _fernet() call generates fresh

    assert accounts.cookie_for(1) is None


def test_a_bad_cookie_is_rejected_at_connect_time_not_stored(monkeypatch, client):
    headers = _auth_header(_register(client))
    monkeypatch.setattr(
        flask_app.fpl_client, "entry", lambda entry_id: {"id": entry_id, "name": "t"}
    )
    client.post("/api/me/fpl", json={"entry_id": 7}, headers=headers)

    def _reject(entry_id, cookie=None):
        raise fpl_client.NotAuthenticated("no")

    monkeypatch.setattr(flask_app.fpl_client, "my_team", _reject)
    response = client.post(
        "/api/me/fpl-cookie", json={"cookie": "sessionid=bad"}, headers=headers
    )

    assert response.status_code == 400
    assert accounts.has_cookie(1) is False


def test_no_response_ever_contains_the_stored_cookie(monkeypatch, client):
    """The vault is write-only from the outside; this is the seal on it."""
    headers = _auth_header(_register(client))
    monkeypatch.setattr(
        flask_app.fpl_client, "entry", lambda entry_id: {"id": entry_id, "name": "t"}
    )
    client.post("/api/me/fpl", json={"entry_id": 7}, headers=headers)
    monkeypatch.setattr(
        flask_app.fpl_client,
        "my_team",
        lambda entry_id, cookie=None: {"picks": [{"element": 1, "position": 1}],
                                       "source": "my_team"},
    )
    client.post(
        "/api/me/fpl-cookie",
        json={"cookie": "sessionid=super-secret-value"},
        headers=headers,
    )

    for path in ("/api/auth/me", "/api/me/team"):
        assert "super-secret-value" not in client.get(
            path, headers=headers
        ).get_data(as_text=True)


# ---------------------------------------------------------------- /api/me/*


def test_me_routes_refuse_anonymous_callers(client):
    for path in ("/api/me/team", "/api/me/transfers", "/api/me/recommendations"):
        assert client.get(path).status_code == 401


def test_me_team_uses_the_linked_entry_and_the_users_own_cookie(
    monkeypatch, client
):
    headers = _auth_header(_register(client))
    monkeypatch.setattr(
        flask_app.fpl_client, "entry", lambda entry_id: {"id": entry_id, "name": "t"}
    )
    client.post("/api/me/fpl", json={"entry_id": 2670555}, headers=headers)
    accounts.store_cookie(1, "sessionid=mine")

    seen = {}

    def _capture(user_id, horizon, cookie=None):
        seen.update(user_id=user_id, cookie=cookie)
        return {"ok": True}, 200

    monkeypatch.setattr(flask_app, "_team_response", _capture)
    client.get("/api/me/team", headers=headers)

    assert seen["user_id"] == 2670555
    assert seen["cookie"] == "sessionid=mine"


def test_me_transfers_resolves_player_names(monkeypatch, client):
    headers = _auth_header(_register(client))
    monkeypatch.setattr(
        flask_app.fpl_client, "entry", lambda entry_id: {"id": entry_id, "name": "t"}
    )
    client.post("/api/me/fpl", json={"entry_id": 7}, headers=headers)

    monkeypatch.setattr(
        flask_app.fpl_client,
        "transfers",
        lambda entry_id: [
            {"event": 3, "element_in": 10, "element_in_cost": 85,
             "element_out": 20, "element_out_cost": 74, "time": "2026-09-01T10:00:00Z"}
        ],
    )
    monkeypatch.setattr(
        flask_app.fpl_client,
        "bootstrap",
        lambda: {"elements": [{"id": 10, "web_name": "Saka"},
                              {"id": 20, "web_name": "Gordon"}]},
    )

    body = client.get("/api/me/transfers", headers=headers).get_json()

    assert body["transfers"] == [
        {"gameweek": 3, "in": "Saka", "in_cost": 8.5,
         "out": "Gordon", "out_cost": 7.4, "time": "2026-09-01T10:00:00Z"}
    ]
