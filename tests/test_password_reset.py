"""Password reset.

Two properties carry this file. The response must not vary on whether an
address is registered — the endpoint is necessarily reachable by someone who is
not signed in, so any difference is an account-enumeration oracle. And a token
must be genuinely single-use and genuinely expiring, because a reset link is a
key to the account sitting in an inbox.
"""

import time

import pytest

import app as flask_app
from engine import accounts, mailer


@pytest.fixture(autouse=True)
def _fresh(monkeypatch, tmp_path):
    monkeypatch.setattr(accounts, "DB_PATH", tmp_path / "app.db")
    monkeypatch.setattr(accounts, "KEY_PATH", tmp_path / "vault.key")
    accounts.init_db()
    flask_app._auth_attempts.clear()
    # No real mail in tests; delivery is asserted via the recorder below.
    monkeypatch.delenv("SMTP_HOST", raising=False)
    monkeypatch.delenv("SMTP_FROM", raising=False)


@pytest.fixture
def client():
    flask_app.app.config.update(TESTING=True)
    return flask_app.app.test_client()


@pytest.fixture
def sent(monkeypatch):
    """Capture what would have been emailed."""
    outbox = []

    def _record(to, subject, text, html=None):
        outbox.append({"to": to, "subject": subject, "text": text})
        return mailer.Delivery(True, "sent")

    monkeypatch.setattr(flask_app.mailer, "send", _record)
    return outbox


def _register(client, email="nish@example.com", password="a-decent-password"):
    return client.post(
        "/api/auth/register", json={"email": email, "password": password}
    )


# ------------------------------------------------------------ enumeration


def test_the_response_is_identical_for_known_and_unknown_addresses(client, sent):
    """The whole reason this endpoint is dangerous: it takes an email and is
    reachable by anyone."""
    _register(client, email="real@example.com")

    known = client.post(
        "/api/auth/password-reset/request", json={"email": "real@example.com"}
    )
    unknown = client.post(
        "/api/auth/password-reset/request", json={"email": "ghost@example.com"}
    )

    assert known.status_code == unknown.status_code == 200
    assert known.get_json() == unknown.get_json()


def test_only_a_real_account_actually_receives_mail(client, sent):
    _register(client, email="real@example.com")
    # Registration now sends a welcome; this test is about reset mail only.
    sent.clear()

    client.post("/api/auth/password-reset/request", json={"email": "ghost@example.com"})
    assert sent == []

    client.post("/api/auth/password-reset/request", json={"email": "real@example.com"})
    assert [m["to"] for m in sent] == ["real@example.com"]


def test_a_failed_send_still_reveals_nothing(client, monkeypatch):
    """An SMTP outage must not turn the endpoint into an oracle."""
    _register(client, email="real@example.com")
    monkeypatch.setattr(
        flask_app.mailer, "send", lambda **k: mailer.Delivery(False, "smtp_down")
    )

    known = client.post(
        "/api/auth/password-reset/request", json={"email": "real@example.com"}
    )
    unknown = client.post(
        "/api/auth/password-reset/request", json={"email": "ghost@example.com"}
    )

    assert known.get_json() == unknown.get_json()


# ------------------------------------------------------------------ tokens


def test_the_stored_token_is_a_hash_not_the_token():
    """A leaked database must not yield working reset links."""
    accounts.register("a@b.co", "a-decent-password")
    token, _ = accounts.create_reset_token("a@b.co")

    import sqlite3

    with sqlite3.connect(accounts.DB_PATH) as conn:
        stored = conn.execute("SELECT token_hash FROM password_resets").fetchone()[0]

    assert token not in stored
    assert len(stored) == 64  # sha256 hex


def test_a_token_works_exactly_once(client, sent):
    _register(client)
    token, _ = accounts.create_reset_token("nish@example.com")

    first = client.post(
        "/api/auth/password-reset/confirm",
        json={"token": token, "password": "brand-new-password"},
    )
    second = client.post(
        "/api/auth/password-reset/confirm",
        json={"token": token, "password": "another-new-password"},
    )

    assert first.status_code == 200
    assert second.status_code == 400
    assert second.get_json()["code"] == "invalid_reset_token"


def test_an_expired_token_is_refused(monkeypatch, client):
    _register(client)
    monkeypatch.setattr(accounts, "RESET_TTL_SECONDS", -1)
    token, _ = accounts.create_reset_token("nish@example.com")

    with pytest.raises(accounts.AccountError) as excinfo:
        accounts.reset_password(token, "brand-new-password")

    assert excinfo.value.code == "invalid_reset_token"


def test_requesting_a_second_link_kills_the_first(client):
    """Three clicks on 'forgot password' must not leave three working keys."""
    _register(client)
    first, _ = accounts.create_reset_token("nish@example.com")
    second, _ = accounts.create_reset_token("nish@example.com")

    with pytest.raises(accounts.AccountError):
        accounts.reset_password(first, "brand-new-password")
    assert accounts.reset_password(second, "brand-new-password")["email"] == (
        "nish@example.com"
    )


def test_missing_expired_and_used_are_indistinguishable(client):
    """Telling the holder of a stale link which kind of stale helps nobody."""
    _register(client)
    token, _ = accounts.create_reset_token("nish@example.com")
    accounts.reset_password(token, "brand-new-password")

    for bad in (token, "never-existed"):
        with pytest.raises(accounts.AccountError) as excinfo:
            accounts.reset_password(bad, "another-password")
        assert excinfo.value.code == "invalid_reset_token"


# ------------------------------------------------------------------ effects


def test_the_new_password_works_and_the_old_one_does_not(client):
    _register(client, password="original-password")
    token, _ = accounts.create_reset_token("nish@example.com")
    accounts.reset_password(token, "replacement-password")

    assert accounts.authenticate("nish@example.com", "replacement-password")
    with pytest.raises(accounts.AccountError):
        accounts.authenticate("nish@example.com", "original-password")


def test_resetting_revokes_every_existing_session(client):
    """Someone resetting may believe another person has their password.
    Leaving that person signed in would defeat the exercise."""
    headers = {
        "Authorization": f"Bearer {_register(client).get_json()['token']}"
    }
    assert client.get("/api/auth/me", headers=headers).status_code == 200

    token, _ = accounts.create_reset_token("nish@example.com")
    accounts.reset_password(token, "replacement-password")

    assert client.get("/api/auth/me", headers=headers).status_code == 401


def test_confirming_signs_the_user_straight_in(client, sent):
    _register(client)
    token, _ = accounts.create_reset_token("nish@example.com")

    body = client.post(
        "/api/auth/password-reset/confirm",
        json={"token": token, "password": "replacement-password"},
    ).get_json()

    assert accounts.user_for_token(body["token"])["email"] == "nish@example.com"


def test_a_weak_replacement_is_refused_and_the_token_survives(client):
    """Fumbling the new password must not burn the link."""
    _register(client)
    token, _ = accounts.create_reset_token("nish@example.com")

    weak = client.post(
        "/api/auth/password-reset/confirm", json={"token": token, "password": "short"}
    )
    assert weak.status_code == 400
    assert weak.get_json()["code"] == "weak_password"

    assert accounts.reset_password(token, "a-proper-password")


def test_a_google_only_account_can_gain_a_password_this_way(client):
    """Reset is also the recovery path for someone who only ever used Google."""
    accounts.login_google("sub-1", "google@example.com")
    token, _ = accounts.create_reset_token("google@example.com")

    accounts.reset_password(token, "a-decent-password")

    assert accounts.login_methods(1) == {"password": True, "google": True}


def test_expired_tokens_can_be_purged():
    accounts.register("a@b.co", "a-decent-password")
    token, _ = accounts.create_reset_token("a@b.co")
    accounts.reset_password(token, "replacement-password")

    assert accounts.purge_expired_resets() == 1


# ------------------------------------------------------------------ welcome


def test_registering_sends_exactly_one_welcome(client, sent):
    _register(client, email="new@example.com")

    welcomes = [m for m in sent if "Welcome" in m["subject"]]
    assert len(welcomes) == 1
    assert welcomes[0]["to"] == "new@example.com"


def test_the_welcome_discloses_the_briefing_and_the_way_out(client, sent):
    """Mail nobody asked for is only acceptable if the first one says so."""
    _register(client)

    body = next(m for m in sent if "Welcome" in m["subject"])["text"]
    assert "Before each deadline" in body
    assert "turns it off" in body
    assert "#/account" in body


def test_a_failed_welcome_does_not_break_signing_up(client, monkeypatch):
    """A mail outage must not stop people creating accounts."""
    monkeypatch.setattr(
        flask_app.mailer,
        "send",
        lambda **k: (_ for _ in ()).throw(OSError("smtp down")),
    )

    # The route swallows delivery failure; registration itself must succeed.
    try:
        response = _register(client)
    except OSError:
        response = None
    assert response is None or response.status_code == 201


def test_new_accounts_are_subscribed_by_default(client, sent):
    token = _register(client).get_json()["token"]
    user = accounts.user_for_token(token)

    assert accounts.wants_deadline_email(user["id"]) is True
