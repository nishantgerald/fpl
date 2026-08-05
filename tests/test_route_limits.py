"""The limits must bind at the route, not merely inside the budget module.

A correct limiter wired to the wrong branch protects nothing, and the live
endpoints can't be exercised for this until the season is under way — they
short-circuit on `season_not_started` long before reaching a model. So the
wiring is tested here against Flask's test client with the service layer
stubbed.
"""

import pytest

import app as flask_app
from engine import llm_budget, narrative


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_budget, "STATE_DIR", tmp_path / "budget")
    flask_app.app.config.update(TESTING=True)
    return flask_app.app.test_client()


def _plan():
    return {
        "n_transfers": 1,
        "hit_cost": 0,
        "net_gain": 6.4,
        "spend": 11,
        "transfers": [
            {
                "out": {"id": 1, "web_name": "Gordon", "name": "Anthony Gordon",
                        "position": "MID", "team": "NEW", "selling_price": 74},
                "in": {"id": 2, "web_name": "Saka", "name": "Bukayo Saka",
                       "position": "MID", "team": "ARS", "now_cost": 85},
            }
        ],
        "reasons": ["Gains 6.4 pts over 5 GWs"],
    }


@pytest.fixture
def _stub_recommendations(monkeypatch):
    monkeypatch.setattr(
        flask_app.service,
        "recommendations",
        lambda **kwargs: {"plans": [_plan()], "recommendation": "transfer"},
    )


def test_a_throttled_caller_still_gets_recommendations(
    monkeypatch, client, _stub_recommendations
):
    """Narration is additive: losing it must not cost the caller the answer."""
    monkeypatch.setattr(narrative, "would_call", lambda plans: True)
    monkeypatch.setattr(llm_budget, "CLIENT_CALLS_PER_HOUR", 0)

    response = client.get("/api/recommendations?user_id=1")

    assert response.status_code == 200
    assert response.get_json()["plans"]
    assert "narrative" not in response.get_json()["plans"][0]


def test_narration_is_not_charged_when_it_would_not_call(
    monkeypatch, client, _stub_recommendations
):
    """A disabled or cached narration must not consume the caller's share."""
    monkeypatch.setattr(narrative, "would_call", lambda plans: False)

    charged = []
    monkeypatch.setattr(llm_budget, "check_client", lambda cid: charged.append(cid))

    client.get("/api/recommendations?user_id=1")
    assert charged == []


def test_the_forwarded_header_is_ignored_by_default(monkeypatch, client):
    """Otherwise a caller mints a fresh identity per request and walks past."""
    monkeypatch.delenv("TRUST_PROXY_HEADER", raising=False)

    with flask_app.app.test_request_context(
        "/api/recommendations", headers={"X-Forwarded-For": "1.2.3.4"}
    ):
        assert flask_app._client_id() != "1.2.3.4"


def test_the_forwarded_header_is_honoured_when_explicitly_trusted(monkeypatch, client):
    monkeypatch.setenv("TRUST_PROXY_HEADER", "true")

    with flask_app.app.test_request_context(
        "/api/recommendations", headers={"X-Forwarded-For": "1.2.3.4, 10.0.0.1"}
    ):
        assert flask_app._client_id() == "1.2.3.4"


def test_engines_reports_the_budget_without_identifying_callers(client):
    body = client.get("/api/engines").get_json()

    assert "daily_ceiling" in body["budget"]
    assert "remaining_today" in body["budget"]
    # Counts only. Nothing that could identify who called.
    assert "clients" not in body["budget"]
    assert "by_kind" not in body["budget"]
