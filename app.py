#!/usr/bin/env python3
"""FPL companion API.

Routes are thin: they parse query params, call into :mod:`engine`, and serialise.
Everything that decides anything lives in the engine package, where it's pure and
testable without a network.

Three scoring systems are exposed, and the client can see and choose between
them via ``GET /api/engines``:

``?engine=xpts``   the hand-built component model (default; no dependencies)
``?engine=ml``     the trained model in :mod:`ml`, falling back to ``xpts``
``?engine=blend``  the mean of the two
``/api/fcps-recommendations``  FCPS plus a written column from the language model

Every engine feeds the same constrained optimiser, so a recommendation is legal
under FPL's rules regardless of which one produced its numbers.

See ``PRDs/`` in the Flutter repo for the specifications behind each endpoint,
and ``PRDs/ml-methodology.md`` for how the trained model was built and validated.
"""

import os

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, redirect, request, send_from_directory
from flask_cors import CORS

from engine import captain as captain_engine
from engine import fcps_llm, fpl_client, ml_scorer, narrative, prices, rules, service, ticker

# Once, at import — not inside a request handler, which is where the old code
# called it.
load_dotenv()

app = Flask(__name__)

# Restrict CORS to the API surface and the origins that actually use it. The old
# `CORS(app)` opened every route to every origin.
_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS",
        "https://fpl.nishantgerald.com,http://localhost:*,http://127.0.0.1:*",
    ).split(",")
    if o.strip()
]
CORS(app, resources={r"/api/*": {"origins": _ALLOWED_ORIGINS}})

STATUS_MAP = {
    "a": "Available",
    "i": "Injured",
    "d": "Doubtful",
    "s": "Suspended",
    "u": "Unavailable",
    "n": "Unavailable",
}


@app.before_request
def _reset_freshness():
    """Track how fresh this request's data is, for the `meta` block."""
    fpl_client.begin_request()


@app.errorhandler(service.ServiceError)
def _handle_service_error(error):
    return jsonify(error.to_dict()), error.status


@app.errorhandler(fcps_llm.FcpsUnavailable)
def _handle_fcps_unavailable(error):
    """FCPS advice couldn't be produced. The client shows why, not a spinner.

    The route this replaces returned 500 from a missing Jinja template, and its
    working path returned 200 with an exception string in the success field.
    """
    return jsonify(error.to_dict()), error.status


@app.errorhandler(fpl_client.UpstreamUnavailable)
def _handle_upstream_unavailable(error):
    """FPL is down and we have nothing cached — say so, don't leak a traceback."""
    return jsonify(
        {
            "code": "upstream_unavailable",
            "error": "The Fantasy Premier League API isn't responding right now.",
        }
    ), 503


# ------------------------------------------------------------------ helpers


def _status_class(status_code: str) -> str:
    if status_code == "d":
        return "doubtful"
    if status_code in ("i", "s", "u", "n"):
        return "injured"
    return "available"


def _int_arg(name, default=None, low=None, high=None):
    value = request.args.get(name, type=int)
    if value is None:
        return default
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return value


def _bool_arg(name, default=True):
    raw = request.args.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes")


def _engine_arg():
    """Which scoring engine to project with. Unknown values fall back silently."""
    requested = (request.args.get("engine") or "").strip().lower()
    return requested if requested in ml_scorer.ENGINES else ml_scorer.DEFAULT_ENGINE


def _player_payload(element, team_short, projection, fcps_entry=None):
    """One player, as the client consumes them.

    Carries both scores. They answer different questions — xPts is in points and
    prices a hit, FCPS is a 0-1000 composite that ranks — and the client labels
    them as such rather than presenting two numbers that look interchangeable.
    """
    status_code = str(element.get("status", "a"))
    next_fixtures = (
        projection["per_gameweek"][0]["fixtures"] if projection.get("per_gameweek") else []
    )
    fcps_entry = fcps_entry or {}
    return {
        "id": int(element["id"]),
        "name": f"{element.get('first_name', '')} {element.get('second_name', '')}".strip(),
        "web_name": element.get("web_name", ""),
        "team": team_short,
        "team_id": int(element.get("team", 0)),
        "position": rules.position_of(element),
        "price": int(element.get("now_cost", 0)) / 10,
        "total_points": int(element.get("total_points", 0)),
        "form": _safe_float(element.get("form")),
        "selected_by_percent": _safe_float(element.get("selected_by_percent")),
        "status": STATUS_MAP.get(status_code, "Unknown"),
        "status_class": _status_class(status_code),
        "news": element.get("news", ""),
        "ict_index": _safe_float(element.get("ict_index")),
        "xpts": projection.get("xpts_next", 0.0),
        "xpts_horizon": projection.get("horizon_xpts", 0.0),
        "xpts_per_million": projection.get("xpts_per_million", 0.0),
        "minutes_risk": projection.get("minutes_risk", "medium"),
        "availability": projection.get("availability", 1.0),
        "next_3_fdr": fcps_entry.get("next_3_fdr", _next_n_fdr(projection, 3)),
        "next_fixtures": next_fixtures,
        "photo": f"{request.host_url}api/photo/{element.get('code', 0)}",
        # FCPS, restored. `fcps_fixtures` says how many fixtures the FDR term is
        # built on — the score divides by three fixtures' worth of difficulty
        # whether or not three were scheduled.
        "fcps": fcps_entry.get("fcps", 0.0),
        "fcps_fixtures": fcps_entry.get("fixtures_counted", 0),
        "fcps_components": fcps_entry.get("components", {}),
        "engine": projection.get("engine", "xpts"),
    }


def _next_n_fdr(projection, count):
    """Sum of the FDRs of the next `count` fixtures actually scheduled.

    Blanks contribute nothing rather than flattering the total, which is the bug
    the old fixed divisor of 15 hid.
    """
    difficulties = [
        fixture["fdr"]
        for entry in projection.get("per_gameweek", ())
        for fixture in entry["fixtures"]
    ][:count]
    return sum(difficulties)


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------------ routes


@app.route("/")
def home():
    return redirect("/app/", code=302)


@app.route("/api/players", methods=["GET"])
def players_data():
    """All players, projected. Optional `player_id`, `position`, `max_price`, `engine`."""
    horizon = _int_arg("horizon", 5, 1, service.MAX_HORIZON)
    projection_set = service.projections_for(horizon, _engine_arg())
    projections, data = projection_set.projections, projection_set.data
    fcps_scores, _data = service.fcps_for()
    teams = {int(t["id"]): t.get("short_name", "UNK") for t in data.get("teams", [])}

    meta = fpl_client.meta({"engine": projection_set.engine})

    player_id = request.args.get("player_id", type=int)
    if player_id is not None:
        element = next(
            (p for p in data.get("elements", []) if int(p["id"]) == player_id), None
        )
        if element is None or player_id not in projections:
            return jsonify({"code": "player_not_found", "error": "Player not found"}), 404
        payload = _player_payload(
            element,
            teams.get(int(element.get("team", 0)), "UNK"),
            projections[player_id],
            fcps_scores.get(player_id),
        )
        return jsonify({"data": [payload], "meta": meta})

    position = request.args.get("position")
    max_price = request.args.get("max_price", type=float)

    payloads = []
    for element in data.get("elements", []):
        pid = int(element["id"])
        projection = projections.get(pid)
        if projection is None:
            continue
        if position and rules.position_of(element) != position.upper():
            continue
        if max_price is not None and int(element.get("now_cost", 0)) / 10 > max_price:
            continue
        payloads.append(
            _player_payload(
                element,
                teams.get(int(element.get("team", 0)), "UNK"),
                projection,
                fcps_scores.get(pid),
            )
        )

    payloads.sort(key=lambda p: (-p["xpts_horizon"], p["id"]))
    return jsonify({"data": payloads, "meta": meta})


@app.route("/api/team")
def team_data():
    """The user's current squad, with projections."""
    user_id = _int_arg("user_id", 3022850)
    horizon = _int_arg("horizon", 5, 1, service.MAX_HORIZON)

    state = fpl_client.season_state()
    if not state:
        return jsonify(
            {"code": "upstream_unavailable", "error": "Failed to fetch gameweek state"}
        ), 503

    if not state["started"]:
        # No picks exist for anyone yet, so a stale ID would otherwise stay
        # hidden until the deadline passes. Surface it now.
        if fpl_client.entry(user_id) is None:
            return jsonify(
                {
                    "code": "entry_not_found",
                    "error": f"No FPL team found with ID {user_id} this season.",
                }
            ), 404
        return jsonify(
            {
                "code": "season_not_started",
                "error": "The season hasn't kicked off yet.",
                "gameweek": state["gameweek"],
                "gameweek_name": state["gameweek_name"],
                "deadline": state["deadline"],
            }
        ), 503

    picks_payload = fpl_client.picks(user_id, state["gameweek"])
    if not picks_payload or not picks_payload.get("picks"):
        return jsonify(
            {
                "code": "entry_not_found",
                "error": "Failed to fetch gameweek picks. Check the user ID.",
            }
        ), 404

    projection_set = service.projections_for(horizon, _engine_arg())
    projections, data = projection_set.projections, projection_set.data
    fcps_scores, _data = service.fcps_for()
    elements = {int(p["id"]): p for p in data.get("elements", [])}
    teams = {int(t["id"]): t.get("short_name", "UNK") for t in data.get("teams", [])}

    lineup = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
    bench = []
    for pick in picks_payload["picks"]:
        element = elements.get(int(pick["element"]))
        if element is None:
            continue
        projection = projections.get(int(element["id"]), {})
        payload = _player_payload(
            element,
            teams.get(int(element.get("team", 0)), "UNK"),
            projection,
            fcps_scores.get(int(element["id"])),
        )
        payload.update(
            {
                "is_captain": bool(pick.get("is_captain")),
                "is_vice_captain": bool(pick.get("is_vice_captain")),
                "starting_eleven": int(pick.get("multiplier", 0)) > 0,
            }
        )
        if payload["starting_eleven"] and payload["position"] in lineup:
            lineup[payload["position"]].append(payload)
        else:
            bench.append(payload)

    history = picks_payload.get("entry_history") or {}
    return jsonify(
        {
            "gameweek": state["gameweek"],
            "deadline": state["deadline"],
            "user_id": user_id,
            "lineup": lineup,
            "bench": bench,
            "bank": history.get("bank"),
            "squad_value": history.get("value"),
            "active_chip": picks_payload.get("active_chip"),
            "engine": projection_set.engine,
            "meta": fpl_client.meta(
                {"gameweek": state["gameweek"], "engine": projection_set.engine}
            ),
        }
    )


@app.route("/api/recommendations")
def recommendations():
    """Rules-legal transfer advice. `engine=xpts|ml|blend` picks the scorer."""
    user_id = request.args.get("user_id", type=int)
    if user_id is None:
        return jsonify({"code": "missing_user_id", "error": "user_id is required"}), 400

    result = service.recommendations(
        entry_id=user_id,
        horizon=_int_arg("horizon", service.DEFAULT_HORIZON, 1, service.MAX_HORIZON),
        max_transfers=_int_arg("max_transfers", service.MAX_SEARCH_TRANSFERS, 0, 3),
        free_transfers_override=_int_arg("free_transfers", None, 0, 5),
        bank_override=_int_arg("bank", None, 0),
        include_hits=_bool_arg("include_hits", True),
        engine=_engine_arg(),
    )

    # Strictly additive: a failure here leaves the response unchanged.
    narrative.annotate(result["plans"])
    return jsonify(result)


@app.route("/api/fcps-recommendations", methods=["GET", "POST"])
def fcps_recommendations():
    """FCPS transfer advice, written by the language model.

    This is the route that used to be ``/trade_recommendations``. Its GET handler
    rendered a template that was never committed, so every GET was a 500, and its
    POST handler was never called by the shipped client. It is JSON on both verbs
    now, and the Flutter app calls it.

    ``user_id`` may arrive as a query parameter or a form field, so the old POST
    call shape still works.
    """
    user_id = request.args.get("user_id", type=int)
    if user_id is None and request.form:
        try:
            user_id = int(request.form.get("user_id", ""))
        except (TypeError, ValueError):
            user_id = None
    if user_id is None:
        return jsonify({"code": "missing_user_id", "error": "user_id is required"}), 400

    return jsonify(
        service.fcps_advice(entry_id=user_id, refresh=_bool_arg("refresh", False))
    )


@app.route("/api/engines")
def engines():
    """What this deployment can actually do, so the client stops guessing.

    The engine picker, the FCPS button and the "trained on" caption are all
    driven by this. Without it the client either hides working features or offers
    broken ones, and both were happening.
    """
    return jsonify(
        {
            "scoring": ml_scorer.describe(),
            "fcps": {
                "available": fcps_llm.is_configured(),
                "model": fcps_llm.model_name(),
                "reason": None
                if fcps_llm.is_configured()
                else "No OpenAI API key is configured on this server.",
            },
            "narrative": {"available": narrative.is_enabled()},
            "meta": fpl_client.meta(),
        }
    )


@app.route("/api/captain")
def captain_picks():
    """Rank the manager's squad for the armband."""
    user_id = request.args.get("user_id", type=int)
    if user_id is None:
        return jsonify({"code": "missing_user_id", "error": "user_id is required"}), 400

    state = fpl_client.season_state()
    if not state or not state["started"]:
        return jsonify(
            {
                "code": "season_not_started",
                "error": "Captain picks start once the season is under way.",
            }
        ), 503

    squad_state = service.load_squad_state(user_id, state["gameweek"])
    projection_set = service.projections_for(1, _engine_arg())
    projections, data = projection_set.projections, projection_set.data
    teams = {int(t["id"]): t for t in data.get("teams", [])}
    event = next(
        (e for e in data.get("events", []) if int(e["id"]) == state["gameweek"]), {}
    )

    result = captain_engine.rank_captains(
        squad=squad_state["squad"],
        picks=squad_state["picks"],
        projections=projections,
        teams=teams,
        gameweek=state["gameweek"],
        most_captained=event.get("most_captained"),
    )
    result["engine"] = projection_set.engine
    result["meta"] = fpl_client.meta(
        {"gameweek": state["gameweek"], "engine": projection_set.engine}
    )
    return jsonify(result)


@app.route("/api/price-changes")
def price_changes():
    """Tonight's likely risers and fallers, framed against the user's squad."""
    data = fpl_client.bootstrap()
    total_players = int(data.get("total_players", 0) or 0)

    squad_ids = None
    user_id = request.args.get("user_id", type=int)
    if user_id is not None:
        state = fpl_client.season_state()
        if state and state["started"]:
            picks_payload = fpl_client.picks(user_id, state["gameweek"])
            if picks_payload and picks_payload.get("picks"):
                squad_ids = [int(p["element"]) for p in picks_payload["picks"]]

    result = prices.predict_all(
        data.get("elements", []), total_players, squad_ids=squad_ids
    )
    result["meta"] = fpl_client.meta()
    return jsonify(result)


@app.route("/api/fixture-ticker")
def fixture_ticker():
    """Team x gameweek difficulty grid, with named fixture swings."""
    state = fpl_client.season_state()
    if not state:
        return jsonify(
            {"code": "upstream_unavailable", "error": "Failed to fetch gameweek state"}
        ), 503

    start = _int_arg("start", state["gameweek"], 1, 38)
    count = _int_arg("count", 6, 1, 12)

    data = fpl_client.bootstrap()
    result = ticker.build_ticker(
        fpl_client.fixtures() or [], data.get("teams", []), start, count
    )
    result["meta"] = fpl_client.meta({"gameweek": state["gameweek"]})
    return jsonify(result)


@app.route("/api/fixtures")
def fixtures_data():
    """Raw fixture list. Kept for compatibility; prefer /api/fixture-ticker."""
    fixtures = fpl_client.fixtures()
    data = fpl_client.bootstrap()
    teams = {int(t["id"]): t.get("short_name", "UNK") for t in data.get("teams", [])}
    return jsonify(
        [
            {
                "gameweek": fixture.get("event"),
                "home_team": teams.get(int(fixture["team_h"]), "UNK"),
                "away_team": teams.get(int(fixture["team_a"]), "UNK"),
                "team_h_difficulty": fixture.get("team_h_difficulty"),
                "team_a_difficulty": fixture.get("team_a_difficulty"),
                "kickoff_time": fixture.get("kickoff_time"),
                "finished": fixture.get("finished", False),
            }
            for fixture in (fixtures or [])
        ]
    )


@app.route("/api/entry/<int:entry_id>")
def get_entry(entry_id):
    """Manager name / team / rank, for first-run confirmation."""
    data = fpl_client.entry(entry_id)
    if data is None:
        return jsonify({"code": "entry_not_found", "error": "Entry not found"}), 404
    return jsonify(
        {
            "id": data["id"],
            "manager_name": f"{data.get('player_first_name', '')} "
            f"{data.get('player_last_name', '')}".strip(),
            "team_name": data.get("name", ""),
            "region": data.get("player_region_name", ""),
            "overall_points": data.get("summary_overall_points", 0),
            "overall_rank": data.get("summary_overall_rank", 0),
            "bank": data.get("last_deadline_bank"),
            "squad_value": data.get("last_deadline_value"),
        }
    )


@app.route("/api/photo/<int:code>")
def player_photo(code):
    """Proxy player photos through our domain.

    Bypasses Safari ITP and other cross-origin restrictions on the client side.
    """
    content = fpl_client.photo(code)
    if content is None:
        return "", 404
    return Response(
        content,
        mimetype="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


# ------------------------------------------------------------------ Flutter app

FLUTTER_BUILD_DIR = os.path.join(os.path.dirname(__file__), "flutter_web")


@app.route("/app")
@app.route("/app/")
def flutter_index():
    return send_from_directory(FLUTTER_BUILD_DIR, "index.html")


@app.route("/app/<path:filename>")
def flutter_static(filename):
    return send_from_directory(FLUTTER_BUILD_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))
