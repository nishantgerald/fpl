"""Google sign-in and account consolidation.

One email, one record, both login methods — with the merge only ever
happening in the direction where ownership is proven. Google's verified email
may claim a password account (Google vouched for the address); a cold
password registration may NOT claim a Google account (nobody vouched for
anything). That asymmetry is the security property under test.
"""

import base64
import json
import time

import pytest
import requests

import app as flask_app
from engine import accounts


@pytest.fixture(autouse=True)
def _fresh_db(monkeypatch, tmp_path):
    monkeypatch.setattr(accounts, "DB_PATH", tmp_path / "app.db")
    monkeypatch.setattr(accounts, "KEY_PATH", tmp_path / "vault.key")
    accounts.init_db()
    flask_app._auth_attempts.clear()


@pytest.fixture
def client():
    flask_app.app.config.update(TESTING=True)
    return flask_app.app.test_client()


@pytest.fixture
def _google_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_SECRET", "test-secret")


def _fake_id_token(**overrides):
    claims = {
        "aud": "test-client-id",
        "sub": "google-sub-1",
        "email": "nish@example.com",
        "email_verified": True,
        **overrides,
    }
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"header.{payload}.signature"


class _FakeExchange:
    def __init__(self, id_token, status_code=200):
        self._id_token = id_token
        self.status_code = status_code

    def json(self):
        return {"id_token": self._id_token}


# ---------------------------------------------------------- consolidation


def test_a_new_google_user_gets_a_passwordless_record():
    token, _ = accounts.login_google("sub-1", "new@example.com")
    user = accounts.user_for_token(token)

    assert user["email"] == "new@example.com"
    assert accounts.login_methods(user["id"]) == {"password": False, "google": True}


def test_google_login_consolidates_into_an_existing_password_account():
    """Same email, one record afterwards — not a duplicate."""
    original = accounts.register("nish@example.com", "a-decent-password")

    token, _ = accounts.login_google("sub-1", "nish@example.com")
    via_google = accounts.user_for_token(token)

    assert via_google["id"] == original["id"]
    assert accounts.login_methods(original["id"]) == {
        "password": True,
        "google": True,
    }
    # And the original method still works on the same record.
    assert accounts.user_for_token(
        accounts.authenticate("nish@example.com", "a-decent-password")
    )["id"] == original["id"]


def test_a_returning_google_user_matches_by_sub_not_email():
    """Changing the email on the Google account must not fork the record."""
    first = accounts.user_for_token(accounts.login_google("sub-1", "old@example.com")[0])
    second = accounts.user_for_token(accounts.login_google("sub-1", "new@example.com")[0])

    assert first["id"] == second["id"]


def test_registering_over_a_google_only_account_is_refused_with_directions(client):
    """The unsafe merge direction: nobody has proven this registrant owns the
    email, so attaching their password would hand them the account."""
    accounts.login_google("sub-1", "nish@example.com")

    response = client.post(
        "/api/auth/register",
        json={"email": "nish@example.com", "password": "a-decent-password"},
    )

    assert response.status_code == 400
    body = response.get_json()
    assert body["code"] == "email_taken"
    assert "Google" in body["error"]
    # And the record is untouched.
    user = accounts.user_for_token(accounts.login_google("sub-1", "nish@example.com")[0])
    assert accounts.login_methods(user["id"])["password"] is False


def test_password_login_on_a_google_only_account_fails_generically():
    accounts.login_google("sub-1", "nish@example.com")

    with pytest.raises(accounts.AccountError) as excinfo:
        accounts.authenticate("nish@example.com", "any-guess-at-all")

    assert excinfo.value.code == "bad_credentials"


def test_set_password_is_the_safe_path_to_both_methods(client):
    """Google-first user adds a password from inside their session."""
    token, _ = accounts.login_google("sub-1", "nish@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    response = client.post(
        "/api/me/password", json={"password": "a-decent-password"}, headers=headers
    )

    assert response.status_code == 200
    assert response.get_json()["methods"] == {"password": True, "google": True}
    fresh = accounts.authenticate("nish@example.com", "a-decent-password")
    assert accounts.user_for_token(fresh)["email"] == "nish@example.com"


# ---------------------------------------------------------------- the flow


def test_unconfigured_deployments_say_so(monkeypatch, client):
    # Explicit, not ambient: `load_dotenv()` puts the developer's real
    # credentials in the environment, so a test about the *absence* of config
    # has to remove them or it silently starts asserting the opposite.
    monkeypatch.delenv("GOOGLE_OAUTH_CLIENT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_OAUTH_CLIENT_SECRET", raising=False)

    assert client.get("/api/auth/config").get_json() == {"google": False}
    assert client.get("/api/auth/google/start").status_code == 503


def test_the_full_redirect_flow_mints_a_working_session(
    monkeypatch, client, _google_env
):
    start = client.get("/api/auth/google/start")
    assert start.status_code == 302
    assert "accounts.google.com" in start.headers["Location"]
    state = dict(
        pair.split("=") for pair in start.headers["Location"].split("?")[1].split("&")
    )["state"]

    monkeypatch.setattr(
        flask_app.requests,
        "post",
        lambda *a, **k: _FakeExchange(_fake_id_token()),
    )
    callback = client.get(f"/api/auth/google/callback?state={state}&code=one-time")

    assert callback.status_code == 302
    location = callback.headers["Location"]
    assert "auth_error" not in location
    token = location.split("token=")[1]
    me = client.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {token}"}
    ).get_json()
    assert me["user"]["email"] == "nish@example.com"
    assert me["methods"] == {"password": False, "google": True}


def test_a_forged_state_is_rejected(client, _google_env):
    # With a legitimate cookie in place, so the rejection is provably about the
    # forged signature rather than the missing cookie.
    client.set_cookie(flask_app._GOOGLE_NONCE_COOKIE, "n", path="/api/auth/google")

    response = client.get("/api/auth/google/callback?state=made-up&code=x")

    assert response.status_code == 302
    assert "auth_error=state_mismatch" in response.headers["Location"]


# ------------------------------------------------- state across worker processes


def test_a_state_verifies_without_the_worker_that_issued_it(
    monkeypatch, client, _google_env
):
    """The bug that broke production: under gunicorn the callback lands on a
    different worker than /start, so anything held in process memory is gone.

    Nothing here calls /start — the state and its cookie are built from the
    client secret alone, which is what every worker shares. If verifying ever
    needs a record of the issuing request again, this fails.
    """
    nonce = "nonce-from-another-process"
    state = flask_app._issue_google_state(nonce)
    client.set_cookie(flask_app._GOOGLE_NONCE_COOKIE, nonce, path="/api/auth/google")
    monkeypatch.setattr(
        flask_app.requests, "post", lambda *a, **k: _FakeExchange(_fake_id_token())
    )

    response = client.get(f"/api/auth/google/callback?state={state}&code=x")

    assert "auth_error" not in response.headers["Location"]
    assert "token=" in response.headers["Location"]


def test_a_state_is_useless_in_a_browser_that_did_not_ask_for_it(client, _google_env):
    """Binding to the cookie is what keeps this a CSRF defence. Without it a
    signed state would be a bearer token anyone could paste."""
    state = flask_app._issue_google_state("the-real-browsers-nonce")
    client.set_cookie(
        flask_app._GOOGLE_NONCE_COOKIE, "some-other-browser", path="/api/auth/google"
    )

    response = client.get(f"/api/auth/google/callback?state={state}&code=x")

    assert "auth_error=state_mismatch" in response.headers["Location"]


def test_a_missing_cookie_is_reported_as_a_blocked_cookie(client, _google_env):
    """Distinct from a mismatch: nothing the user retries will fix it, so the
    message has to send them somewhere else."""
    state = flask_app._issue_google_state("whatever")

    response = client.get(f"/api/auth/google/callback?state={state}&code=x")

    assert "auth_error=cookie_blocked" in response.headers["Location"]


def test_a_stale_state_expires(monkeypatch, client, _google_env):
    state = flask_app._issue_google_state("n")
    client.set_cookie(flask_app._GOOGLE_NONCE_COOKIE, "n", path="/api/auth/google")
    # Capture the real clock first — flask_app.time *is* this module's `time`,
    # so patching it with a lambda that calls time.time() recurses forever.
    now = time.time()
    monkeypatch.setattr(
        flask_app.time, "time", lambda: now + flask_app._GOOGLE_STATE_TTL + 30
    )

    response = client.get(f"/api/auth/google/callback?state={state}&code=x")

    assert "auth_error=state_expired" in response.headers["Location"]


def test_the_nonce_cookie_is_cleared_after_the_callback(
    monkeypatch, client, _google_env
):
    """One redirect back is all it's good for; a live cookie would let the
    same state be replayed until it expired."""
    nonce = "single-use"
    state = flask_app._issue_google_state(nonce)
    client.set_cookie(flask_app._GOOGLE_NONCE_COOKIE, nonce, path="/api/auth/google")
    monkeypatch.setattr(
        flask_app.requests, "post", lambda *a, **k: _FakeExchange(_fake_id_token())
    )

    first = client.get(f"/api/auth/google/callback?state={state}&code=x")
    assert "auth_error" not in first.headers["Location"]

    # The test client honours Set-Cookie, so the deletion carries into the retry.
    replay = client.get(f"/api/auth/google/callback?state={state}&code=x")
    assert "auth_error=cookie_blocked" in replay.headers["Location"]


def test_start_sets_a_nonce_cookie_that_survives_googles_redirect(client, _google_env):
    """SameSite=Strict would strip the cookie on the way back from Google and
    break sign-in for everyone — the failure this whole mechanism exists to fix."""
    cookie = client.get("/api/auth/google/start").headers["Set-Cookie"]

    assert flask_app._GOOGLE_NONCE_COOKIE in cookie
    assert "SameSite=Lax" in cookie
    assert "HttpOnly" in cookie


@pytest.mark.parametrize(
    "claims, expected",
    [
        ({"email_verified": False}, "unverified_email"),
        ({"aud": "someone-elses-client"}, "bad_audience"),
    ],
)
def test_bad_identities_never_reach_consolidation(
    monkeypatch, client, _google_env, claims, expected
):
    start = client.get("/api/auth/google/start")
    state = dict(
        pair.split("=") for pair in start.headers["Location"].split("?")[1].split("&")
    )["state"]
    monkeypatch.setattr(
        flask_app.requests,
        "post",
        lambda *a, **k: _FakeExchange(_fake_id_token(**claims)),
    )

    response = client.get(f"/api/auth/google/callback?state={state}&code=x")

    assert f"auth_error={expected}" in response.headers["Location"]


def test_migration_adds_the_column_to_a_pre_google_database(tmp_path, monkeypatch):
    """init_db must upgrade in place; existing users keep working."""
    import sqlite3

    db = tmp_path / "old.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT NOT NULL UNIQUE,"
            " password_hash TEXT NOT NULL, created_at REAL NOT NULL)"
        )
        conn.execute(
            "INSERT INTO users (email, password_hash, created_at)"
            " VALUES ('old@example.com', 'x', 0)"
        )
    monkeypatch.setattr(accounts, "DB_PATH", db)

    accounts.init_db()
    token, _ = accounts.login_google("sub-1", "old@example.com")

    assert accounts.user_for_token(token)["email"] == "old@example.com"


def test_a_first_google_sign_in_reports_that_it_created_the_account():
    """The caller needs to know, because a welcome email is owed exactly once."""
    _, created_first = accounts.login_google("sub-1", "new@example.com")
    _, created_again = accounts.login_google("sub-1", "new@example.com")

    assert created_first is True
    assert created_again is False


def test_google_consolidation_into_an_existing_account_is_not_a_creation():
    """No welcome email for someone who already has an account."""
    accounts.register("nish@example.com", "a-decent-password")

    _, created = accounts.login_google("sub-1", "nish@example.com")

    assert created is False
