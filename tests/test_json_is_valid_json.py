"""Responses must be JSON that a JSON parser accepts.

Python's json module emits bare `NaN` and `Infinity`. Those are a Python
extension, not JSON, and every strict parser rejects them — including the two
that matter here, `JSON.parse` and Dart's `jsonDecode`.

This was not hypothetical. The trained model's metadata carries NaN for metrics
that were never computed, `/api/engines` embedded it, and the Flutter client's
`fetchEngines` swallowed the parse failure and returned `EngineInfo.unknown`.
The visible result was a Transfers tab with no engine picker and no FCPS
column — two features present on the server and unreachable in the client, with
nothing logged anywhere to say why.
"""

import json

import pytest

import app as flask_app


@pytest.fixture
def client():
    flask_app.app.config.update(TESTING=True)
    return flask_app.app.test_client()


def test_non_finite_floats_serialise_as_null(client):
    payload = {
        "nan": float("nan"),
        "inf": float("inf"),
        "ninf": float("-inf"),
        "nested": {"xs": [1.0, float("nan"), 3.0]},
        "ok": 1.5,
    }
    body = flask_app.app.json.dumps(payload)

    # The point of the exercise: a strict parser has to accept it.
    parsed = json.loads(body)

    assert parsed["nan"] is None
    assert parsed["inf"] is None
    assert parsed["ninf"] is None
    assert parsed["nested"]["xs"] == [1.0, None, 3.0]
    assert parsed["ok"] == 1.5


def test_no_bare_nan_token_survives_serialisation():
    """`json.loads` is lenient about these, so assert on the bytes too."""
    body = flask_app.app.json.dumps({"x": float("nan"), "y": float("inf")})
    assert "NaN" not in body
    assert "Infinity" not in body


def test_engines_endpoint_is_strictly_parseable(client):
    """The endpoint that actually shipped the bug."""
    response = client.get("/api/engines")
    assert response.status_code == 200

    body = response.get_data(as_text=True)
    assert "NaN" not in body
    json.loads(body)  # raises if the response is not JSON
