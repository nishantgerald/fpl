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

import base64
import json
import os
import secrets
import time
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
)
from flask_cors import CORS

from engine import captain as captain_engine
from engine import (
    accounts,
    content,
    digest,
    fcps_llm,
    fpl_client,
    leagues,
    llm_budget,
    mailer,
    ml_scorer,
    narrative,
    prices,
    research,
    rules,
    service,
    squad_import,
    ticker,
    vision,
)

# Once, at import — not inside a request handler, which is where the old code
# called it.
load_dotenv()
accounts.init_db()

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


def _team_games_played() -> int | None:
    """League games played so far, for reporting whether the ML engine is live.

    Best-effort: this decorates a status response, so an upstream hiccup should
    leave the field absent rather than fail the request.
    """
    from engine import xpts

    try:
        data = fpl_client.bootstrap()
        return xpts.team_games_played(data.get("teams", []), data.get("events", []))
    except Exception:
        return None


def _client_id() -> str:
    """Best available identifier for the caller, for per-client throttling.

    ``X-Forwarded-For`` is attacker-controlled unless a trusted proxy is known to
    overwrite it, so it is ignored by default: honouring it on a directly-exposed
    app would let anyone mint a fresh identity per request and walk straight past
    the limit. Set ``TRUST_PROXY_HEADER=true`` only when a reverse proxy in front
    of this app rewrites the header.
    """
    trust = os.getenv("TRUST_PROXY_HEADER", "false").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if trust:
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            # Left-most entry is the originating client per the convention.
            return forwarded.split(",")[0].strip()
    return request.remote_addr or ""


# Auth endpoints attract password-guessing the moment they exist. A sliding
# in-memory window per client is the right weight for one process; it resets on
# restart, which for brute-force protection is acceptable — the attacker's
# progress resets with it.
_AUTH_ATTEMPT_LIMIT = int(os.getenv("AUTH_ATTEMPTS_PER_WINDOW", 10))
_AUTH_WINDOW_SECONDS = int(os.getenv("AUTH_WINDOW_SECONDS", 900))
_auth_attempts: dict[str, list[float]] = {}


def _auth_throttled(client: str) -> bool:
    import time as _time

    now = _time.time()
    window = [t for t in _auth_attempts.get(client, []) if now - t < _AUTH_WINDOW_SECONDS]
    if len(window) >= _AUTH_ATTEMPT_LIMIT:
        _auth_attempts[client] = window
        return True
    window.append(now)
    _auth_attempts[client] = window
    return False


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


# ------------------------------------------------- public, indexable pages
#
# The Flutter bundle is invisible to a search engine: everything it shows is
# assembled by JavaScript in the visitor's browser after load, so a crawler
# fetching this domain sees an empty document and concludes the site has no
# content. These pages are the same numbers as the API, rendered as real HTML
# at stable URLs, so the projections can actually be found. They read from
# `service`, never from a second model, so a page and the app cannot quote
# different figures for the same player.

SEASON_LABEL = os.getenv("SEASON_LABEL", "2026/27")
PAGE_HORIZON = 5


def _canonical(path: str) -> str:
    base = os.getenv("PUBLIC_BASE_URL", request.host_url.rstrip("/"))
    return f"{base.rstrip('/')}{path}"


def _page_context():
    """Projections, elements and slugs — everything a public page starts from."""
    projection_set = service.projections_for(PAGE_HORIZON, ml_scorer.DEFAULT_ENGINE)
    data = projection_set.data
    elements = [
        e
        for e in data.get("elements", [])
        if rules.position_of(e) in ("GKP", "DEF", "MID", "FWD")
    ]
    return {
        "projections": projection_set.projections,
        "elements": elements,
        "teams": {
            int(t["id"]): t.get("short_name", "UNK") for t in data.get("teams", [])
        },
        "slugs": content.build_slug_index(elements),
        "gameweeks": projection_set.gameweeks,
    }


def _ranked_rows(ctx, position=None, limit=None):
    """Players sorted by horizon xPts, shaped for a table row."""
    rows = []
    for element in ctx["elements"]:
        if position and rules.position_of(element) != position:
            continue
        projection = ctx["projections"].get(int(element["id"]))
        if projection is None:
            continue
        rows.append(
            {
                "id": int(element["id"]),
                "slug": content.player_slug(element),
                "web_name": element.get("web_name", ""),
                "team": ctx["teams"].get(int(element.get("team", 0)), "UNK"),
                "position": rules.position_of(element),
                "price": int(element.get("now_cost", 0)) / 10,
                "xpts_next": float(projection.get("xpts_next") or 0.0),
                "horizon_xpts": float(projection.get("horizon_xpts") or 0.0),
                "per_gameweek": [
                    g["xpts"] for g in projection.get("per_gameweek", [])
                ],
            }
        )
    rows.sort(key=lambda r: (-r["horizon_xpts"], r["id"]))
    return rows[:limit] if limit else rows


@app.route("/")
def home():
    ctx = _page_context()
    state = fpl_client.season_state() or {}
    return render_template(
        "landing.html",
        canonical=_canonical("/"),
        season=SEASON_LABEL,
        horizon=PAGE_HORIZON,
        gameweek=state.get("gameweek"),
        top=_ranked_rows(ctx, limit=20),
    )


@app.route("/projections")
def projections_page():
    ctx = _page_context()
    state = fpl_client.season_state() or {}
    groups = [{"title": "Top 40 overall", "players": _ranked_rows(ctx, limit=40)}]
    for position, title in (
        ("GKP", "Goalkeepers"),
        ("DEF", "Defenders"),
        ("MID", "Midfielders"),
        ("FWD", "Forwards"),
    ):
        groups.append(
            {"title": title, "players": _ranked_rows(ctx, position, limit=25)}
        )
    return render_template(
        "projections.html",
        canonical=_canonical("/projections"),
        season=SEASON_LABEL,
        horizon=PAGE_HORIZON,
        gameweek=state.get("gameweek"),
        gameweeks=ctx["gameweeks"],
        groups=groups,
    )


@app.route("/players")
def players_index_page():
    ctx = _page_context()
    groups = []
    for position, title in (
        ("GKP", "Goalkeepers"),
        ("DEF", "Defenders"),
        ("MID", "Midfielders"),
        ("FWD", "Forwards"),
    ):
        groups.append(
            {"title": title, "players": _ranked_rows(ctx, position, limit=30)}
        )
    return render_template(
        "players_index.html",
        canonical=_canonical("/players"),
        season=SEASON_LABEL,
        groups=groups,
    )


@app.route("/players/<slug>")
def player_page(slug):
    ctx = _page_context()
    element_id = ctx["slugs"].get(slug)
    if element_id is None:
        return render_template("base.html", canonical=_canonical("/players")), 404

    element = next(e for e in ctx["elements"] if int(e["id"]) == element_id)
    position = rules.position_of(element)
    same_position = _ranked_rows(ctx, position)
    rank = next(
        (i for i, r in enumerate(same_position, 1) if r["id"] == element_id),
        len(same_position),
    )
    alternatives = [r for r in same_position[:6] if r["id"] != element_id][:5]

    page = content.player_page(
        element,
        ctx["projections"][element_id],
        ctx["teams"].get(int(element.get("team", 0)), "UNK"),
        rank_in_position=rank,
        position_total=len(same_position),
        alternatives=alternatives,
        horizon=PAGE_HORIZON,
    )
    return render_template(
        "player.html",
        canonical=_canonical(f"/players/{slug}"),
        season=SEASON_LABEL,
        p=page,
        verdict=content.verdict(page),
    )


@app.route("/captain")
def captain_page():
    ctx = _page_context()
    state = fpl_client.season_state() or {}
    picks = []
    for row in _global_captain_ranking(15):
        element = next(
            (e for e in ctx["elements"] if int(e["id"]) == row["id"]), None
        )
        picks.append(
            {
                **row,
                "slug": content.player_slug(element) if element else "",
                "fixture": " + ".join(
                    f"{f['opponent']} ({'H' if f['home'] else 'A'})"
                    for f in row["fixtures"]
                ),
            }
        )
    return render_template(
        "captain.html",
        canonical=_canonical("/captain"),
        season=SEASON_LABEL,
        gameweek=state.get("gameweek"),
        picks=picks,
    )


@app.route("/fixtures")
def fixtures_page():
    state = fpl_client.season_state()
    if not state:
        return render_template("base.html", canonical=_canonical("/fixtures")), 503
    start, count = state["gameweek"], 8
    data = fpl_client.bootstrap()
    result = ticker.build_ticker(
        fpl_client.fixtures() or [], data.get("teams", []), start, count
    )
    teams = sorted(result.get("teams", []), key=lambda t: t.get("avg_fdr", 5))
    for team in teams:
        for cell in team.get("cells", []):
            fixtures = cell.get("fixtures") or []
            cell["label"] = (
                " + ".join(
                    f"{f['opponent']}{'' if f['home'] else ' (a)'}" for f in fixtures
                )
                or "BLANK"
            )
            cell["fdr"] = (
                round(sum(f["fdr"] for f in fixtures) / len(fixtures), 1)
                if fixtures
                else None
            )
    return render_template(
        "fixtures.html",
        canonical=_canonical("/fixtures"),
        season=SEASON_LABEL,
        start=start,
        count=count,
        gameweeks=list(range(start, start + count)),
        teams=teams,
        swings=result.get("swings", []),
    )


@app.route("/robots.txt")
def robots():
    return Response(
        "User-agent: *\nAllow: /\nDisallow: /api/\n\n"
        f"Sitemap: {_canonical('/sitemap.xml')}\n",
        mimetype="text/plain",
    )


@app.route("/sitemap.xml")
def sitemap():
    """Every indexable URL. Player pages are capped: 700 near-identical pages
    would compete with each other for the same queries rather than rank."""
    urls = [("/", "daily", "1.0"), ("/projections", "hourly", "0.9"),
            ("/captain", "hourly", "0.9"), ("/fixtures", "daily", "0.8"),
            ("/players", "daily", "0.8")]
    ctx = _page_context()
    for row in _ranked_rows(ctx, limit=content.INDEXABLE_PLAYER_COUNT):
        urls.append((f"/players/{row['slug']}", "daily", "0.7"))

    body = ["<?xml version='1.0' encoding='UTF-8'?>",
            "<urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>"]
    for path, freq, priority in urls:
        body.append(
            f"<url><loc>{_canonical(path)}</loc>"
            f"<changefreq>{freq}</changefreq>"
            f"<priority>{priority}</priority></url>"
        )
    body.append("</urlset>")
    return Response("\n".join(body), mimetype="application/xml")


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


@app.route("/api/player/<int:player_id>/projection")
def player_projection(player_id):
    """The gameweek-by-gameweek breakdown behind one player's horizon xPts.

    A separate route rather than a field on `/api/players`: the components are
    nine floats per gameweek per player, which would grow the list payload by
    an order of magnitude to serve a panel that is only ever open for one
    player at a time.
    """
    horizon = _int_arg("horizon", service.DEFAULT_HORIZON, 1, service.MAX_HORIZON)
    projection_set = service.projections_for(horizon, _engine_arg())
    projection = projection_set.projections.get(player_id)

    element = next(
        (
            p
            for p in projection_set.data.get("elements", [])
            if int(p["id"]) == player_id
        ),
        None,
    )
    if projection is None or element is None:
        return jsonify({"code": "player_not_found", "error": "Player not found"}), 404

    return jsonify(
        {
            "id": player_id,
            "web_name": element.get("web_name", ""),
            "horizon": horizon,
            "horizon_xpts": projection.get("horizon_xpts", 0.0),
            "per_gameweek": projection.get("per_gameweek", []),
            "engine": projection_set.engine,
            "meta": fpl_client.meta({"engine": projection_set.engine}),
        }
    )


@app.route("/api/session")
def session_state():
    """Whether the stored FPL session cookie still works.

    Exists to be polled by a scheduled check: a cookie fails silently, and the
    default way to discover that is a team screen that quietly stopped
    updating. Reports state only — never any part of the cookie itself.
    """
    user_id = _int_arg("user_id", 3022850)
    status = fpl_client.session_status(user_id)
    # `ok: false` here is a truthful report, not a failed request, so the route
    # is a 200 and the caller reads `state`.
    return jsonify({"user_id": user_id, **status, "meta": fpl_client.meta()})


def _team_response(user_id: int, horizon: int, cookie: str | None = None):
    """Build the squad payload for one entry.

    Shared by the single-user route (deployment-wide cookie) and the
    per-account route (that user's own vaulted cookie) so the two can never
    drift in shape. Returns ``(body, status)``.
    """
    state = fpl_client.season_state()
    if not state:
        return {
            "code": "upstream_unavailable",
            "error": "Failed to fetch gameweek state",
        }, 503

    picks_payload = None
    if not state["started"]:
        # No picks exist for anyone yet, so a stale ID would otherwise stay
        # hidden until the deadline passes. Surface it now.
        if fpl_client.entry(user_id) is None:
            return {
                "code": "entry_not_found",
                "error": f"No FPL team found with ID {user_id} this season.",
            }, 404
        # A drafted squad does exist before the deadline — it just isn't public.
        # With a session cookie we can read it; without one this is still a 503,
        # exactly as before.
        try:
            picks_payload = fpl_client.my_team(user_id, cookie=cookie)
        except fpl_client.NotAuthenticated as exc:
            return {
                "code": "season_not_started",
                "error": "The season hasn't kicked off yet.",
                "detail": str(exc),
                "gameweek": state["gameweek"],
                "gameweek_name": state["gameweek_name"],
                "deadline": state["deadline"],
            }, 503

    if picks_payload is None and state["started"] and cookie:
        # In-season the cookie is still the best source — it sees the live
        # draft for next gameweek, not the one locked at the last deadline.
        # But it's an upgrade here, not a requirement: a dead cookie degrades
        # to the public path rather than failing the request.
        try:
            picks_payload = fpl_client.my_team(user_id, cookie=cookie)
        except fpl_client.NotAuthenticated:
            picks_payload = None

    if picks_payload is None and state["started"]:
        # No cookie: fold post-deadline transfers into the last locked squad.
        # Ownership changes are public and irreversible the moment they're
        # made; only lineup choices wait for the deadline.
        picks_payload = service.current_picks(user_id, state["gameweek"])

    if picks_payload is None:
        picks_payload = fpl_client.picks(user_id, state["gameweek"])
    if not picks_payload or not picks_payload.get("picks"):
        return {
            "code": "entry_not_found",
            "error": "Failed to fetch gameweek picks. Check the user ID.",
        }, 404

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
    return {
        "gameweek": state["gameweek"],
        "deadline": state["deadline"],
        "user_id": user_id,
        "lineup": lineup,
        "bench": bench,
        "bank": history.get("bank"),
        "squad_value": history.get("value"),
        "active_chip": picks_payload.get("active_chip"),
        # Pre-deadline this squad is a draft the manager can still change,
        # and the client says so rather than presenting it as settled. A
        # reconstructed squad is provisional too — ownership is fact, but the
        # lineup shown is last week's carried forward.
        "provisional": picks_payload.get("source") in ("my_team", "reconstructed"),
        "reconstructed": picks_payload.get("source") == "reconstructed",
        "engine": projection_set.engine,
        "meta": fpl_client.meta(
            {"gameweek": state["gameweek"], "engine": projection_set.engine}
        ),
    }, 200


@app.route("/api/team")
def team_data():
    """The user's current squad, with projections."""
    user_id = _int_arg("user_id", 3022850)
    horizon = _int_arg("horizon", 5, 1, service.MAX_HORIZON)
    body, status = _team_response(user_id, horizon)
    return jsonify(body), status


# ------------------------------------------------------------------ accounts
#
# App accounts, not Premier League accounts. Users register with an email and
# password *for this app* and link their FPL entry by Team ID — the model every
# large FPL tool uses, because the public API serves squads, transfers and
# leagues from the ID alone once the season starts. We never ask for, see, or
# store a Premier League password. The optional extra is a per-user FPL session
# cookie for pre-deadline squad reads, held encrypted and never returned.


def _bearer_token() -> str | None:
    header = request.headers.get("Authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip() or None
    return None


def _current_user() -> dict | None:
    return accounts.user_for_token(_bearer_token())


def _auth_required():
    return jsonify({"code": "auth_required", "error": "Sign in first."}), 401


# --- Google sign-in (server-side OAuth code flow) ---
#
# The redirect flow, not the JS popup: the browser goes to Google, Google
# comes back to /callback with a one-time code, and this server exchanges it
# for the identity — so the client secret never ships in the bundle and the
# Flutter app needs no Google SDK. The id_token arrives directly from
# Google's token endpoint over TLS, which is why decoding it without a
# signature check is sound here; a token *presented by a client* could never
# be trusted that way.

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GOOGLE_STATE_TTL = 600
_google_states: dict[str, float] = {}


def _google_client():
    return (
        os.getenv("GOOGLE_OAUTH_CLIENT_ID", "").strip(),
        os.getenv("GOOGLE_OAUTH_CLIENT_SECRET", "").strip(),
    )


def _google_redirect_uri() -> str:
    base = request.host_url.rstrip("/")
    # Behind Heroku's router the request looks like http; Google requires the
    # registered https URI to match exactly. Localhost is the one place plain
    # http is legitimate (and what Google's console allows).
    if "localhost" not in base and "127.0.0.1" not in base:
        base = "https://" + base.split("://", 1)[1]
    return f"{base}/api/auth/google/callback"


def _decode_jwt_payload(token: str) -> dict:
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))


@app.route("/api/auth/config")
def auth_config():
    """What sign-in options this deployment offers, so the client can hide
    buttons that would dead-end."""
    return jsonify({"google": bool(_google_client()[0])})


@app.route("/api/auth/google/start")
def google_start():
    client_id, client_secret = _google_client()
    if not client_id or not client_secret:
        return jsonify(
            {"code": "google_not_configured", "error": "Google sign-in isn't set up."}
        ), 503

    now = time.time()
    for stale in [s for s, exp in _google_states.items() if exp < now]:
        _google_states.pop(stale, None)
    state = secrets.token_urlsafe(24)
    _google_states[state] = now + _GOOGLE_STATE_TTL

    return redirect(
        GOOGLE_AUTH_URL
        + "?"
        + urlencode(
            {
                "client_id": client_id,
                "redirect_uri": _google_redirect_uri(),
                "response_type": "code",
                "scope": "openid email",
                "state": state,
                "prompt": "select_account",
            }
        )
    )


@app.route("/api/auth/google/callback")
def google_callback():
    # Errors land back in the app as a query code, not a JSON wall: the user
    # is mid-redirect in a browser, not a JSON consumer.
    def fail(code):
        return redirect(f"/app/#/account?auth_error={code}")

    if _google_states.pop(request.args.get("state", ""), None) is None:
        return fail("state_mismatch")
    code = request.args.get("code")
    if not code:
        return fail(request.args.get("error", "cancelled"))

    client_id, client_secret = _google_client()
    try:
        exchange = requests.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": _google_redirect_uri(),
                "grant_type": "authorization_code",
            },
            timeout=10,
        )
    except requests.RequestException:
        return fail("google_unreachable")
    if exchange.status_code != 200:
        return fail("exchange_failed")

    try:
        claims = _decode_jwt_payload(exchange.json().get("id_token", ""))
    except Exception:
        return fail("bad_token")
    if claims.get("aud") != client_id:
        return fail("bad_audience")
    if not claims.get("email") or not claims.get("email_verified"):
        # An unverified email must never consolidate into an existing account.
        return fail("unverified_email")

    try:
        token, created = accounts.login_google(
            str(claims.get("sub", "")), claims["email"]
        )
    except accounts.AccountError:
        return fail("bad_google_identity")
    if created:
        _send_welcome(claims["email"])
    # Land on the team, not the settings page. Signing in is a means to seeing
    # your squad; dropping the user on a form is answering a question they
    # didn't ask.
    return redirect(f"/app/#/team?token={token}")


@app.route("/api/me/password", methods=["POST"])
def me_set_password():
    """Add or change a password from inside a signed-in session.

    For a Google-first account this is what makes email+password work later —
    the safe half of consolidation, because being here proves ownership.
    """
    user = _current_user()
    if user is None:
        return _auth_required()
    body = request.get_json(silent=True) or {}
    try:
        accounts.set_password(user["id"], body.get("password", ""))
    except accounts.AccountError as exc:
        return jsonify({"code": exc.code, "error": str(exc)}), 400
    return jsonify({"ok": True, "methods": accounts.login_methods(user["id"])})


@app.route("/api/auth/register", methods=["POST"])
def auth_register():
    if _auth_throttled(_client_id()):
        return jsonify(
            {"code": "throttled", "error": "Too many attempts. Try again later."}
        ), 429
    body = request.get_json(silent=True) or {}
    try:
        user = accounts.register(body.get("email", ""), body.get("password", ""))
        token = accounts.authenticate(body.get("email", ""), body.get("password", ""))
    except accounts.AccountError as exc:
        return jsonify({"code": exc.code, "error": str(exc)}), 400
    _send_welcome(user["email"])
    return jsonify({"token": token, "user": user}), 201


@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    if _auth_throttled(_client_id()):
        return jsonify(
            {"code": "throttled", "error": "Too many attempts. Try again later."}
        ), 429
    body = request.get_json(silent=True) or {}
    try:
        token = accounts.authenticate(body.get("email", ""), body.get("password", ""))
    except accounts.AccountError as exc:
        # 401, and the same body for unknown email and wrong password — see
        # accounts.authenticate for why they are indistinguishable on purpose.
        return jsonify({"code": exc.code, "error": str(exc)}), 401
    user = accounts.user_for_token(token)
    return jsonify({"token": token, "user": user})


def _send_welcome(email: str) -> None:
    """Greet a brand-new account, once.

    Also the honest place to disclose that deadline briefings are on: an email
    the user did not ask for is only acceptable if the first one says so and
    shows the way out. Best-effort — a failure here must never break signing up.
    """
    account_url = _canonical("/app/") + "#/account"
    text = (
        "Welcome to FPL Companion.\n\n"
        "Three things worth knowing:\n\n"
        "1. Link your FPL Team ID and every screen fills in with your own "
        "squad — projections, captain picks, transfers, mini-league.\n"
        "2. Before each deadline we'll email you one short briefing: your "
        "captain, the transfer worth making (or a clear call to roll), and "
        "anything in your squad that has become a problem.\n"
        "3. If you'd rather not have that, one switch turns it off:\n"
        f"   {account_url}\n\n"
        "Projections are estimates, not guarantees. Check team news before "
        "the deadline.\n"
    )
    html = (
        '<div style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,'
        'Roboto,Helvetica,Arial,sans-serif;max-width:520px;margin:0 auto;'
        'color:#1a1a2e">'
        "<h1 style=\"font-size:20px\">Welcome to FPL Companion</h1>"
        "<p>Link your FPL Team ID and every screen fills in with your own "
        "squad — projections, captain picks, transfers, mini-league.</p>"
        "<p>Before each deadline we'll send one short briefing: your captain, "
        "the transfer worth making (or a clear call to roll), and anything in "
        "your squad that has become a problem.</p>"
        f'<p style="font-size:13px;color:#6b7280">Prefer not to get those? '
        f'One switch turns them off: <a href="{account_url}">your account '
        "settings</a>.</p>"
        '<p style="font-size:12px;color:#6b7280;border-top:1px solid #e6e8ec;'
        'padding-top:12px">Projections are estimates, not guarantees. Check '
        "team news before the deadline.</p></div>"
    )
    delivery = mailer.send(
        to=email, subject="Welcome to FPL Companion", text=text, html=html
    )
    if not delivery.sent:
        print(f"[welcome] not sent to {email}: {delivery.reason}", flush=True)


def _reset_email(token: str, email: str) -> tuple[str, str]:
    link = f"{_canonical('/app/')}#/reset?token={token}"
    minutes = accounts.RESET_TTL_SECONDS // 60
    text = (
        "Someone asked to reset the password for your FPL Companion account.\n\n"
        f"Open this link to choose a new one:\n{link}\n\n"
        f"The link works once and expires in {minutes} minutes.\n\n"
        "If this wasn't you, ignore this email — nothing has changed, and your "
        "password still works.\n"
    )
    html = (
        '<div style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,'
        'Roboto,Helvetica,Arial,sans-serif;max-width:520px;margin:0 auto">'
        "<p>Someone asked to reset the password for your FPL Companion "
        "account.</p>"
        f'<p><a href="{link}" style="display:inline-block;background:#37003c;'
        'color:#fff;padding:12px 20px;border-radius:8px;text-decoration:none;'
        'font-weight:600">Choose a new password</a></p>'
        f"<p style=\"font-size:13px;color:#6b7280\">The link works once and "
        f"expires in {minutes} minutes.</p>"
        '<p style="font-size:13px;color:#6b7280">If this wasn\'t you, ignore '
        "this email — nothing has changed, and your password still works.</p>"
        "</div>"
    )
    return text, html


@app.route("/api/auth/password-reset/request", methods=["POST"])
def auth_password_reset_request():
    """Start a password reset.

    The response is identical whether or not the address is registered. An
    unauthenticated endpoint that says "no such account" is an enumeration
    oracle, and this one is necessarily reachable by someone not signed in —
    so the *email* is sent only to a real account, while the *answer* tells
    the caller nothing.
    """
    if _auth_throttled(_client_id()):
        return jsonify(
            {"code": "throttled", "error": "Too many attempts. Try again later."}
        ), 429

    body = request.get_json(silent=True) or {}
    minted = accounts.create_reset_token(body.get("email", ""))

    if minted is not None:
        token, user = minted
        text, html = _reset_email(token, user["email"])
        delivery = mailer.send(
            to=user["email"],
            subject="Reset your FPL Companion password",
            text=text,
            html=html,
        )
        if not delivery.sent:
            # Logged, never returned: the caller must not learn from the
            # response that an account exists here either.
            print(f"[reset] could not send to user {user['id']}: {delivery.reason}", flush=True)
            if app.debug or os.getenv("RESET_LINK_TO_LOG", "").lower() == "true":
                print(f"[reset] link for {user['email']}: /app/#/reset?token={token}", flush=True)

    return jsonify(
        {
            "ok": True,
            "message": "If that email has an account, a reset link is on its way.",
        }
    )


@app.route("/api/auth/password-reset/confirm", methods=["POST"])
def auth_password_reset_confirm():
    """Consume a reset link and set the new password, returning a session.

    Signing the user straight in is deliberate: they have just proved control
    of the mailbox and chosen a password, so making them type it again buys
    nothing but a chance to mistype it.
    """
    if _auth_throttled(_client_id()):
        return jsonify(
            {"code": "throttled", "error": "Too many attempts. Try again later."}
        ), 429

    body = request.get_json(silent=True) or {}
    try:
        user = accounts.reset_password(
            body.get("token", ""), body.get("password", "")
        )
    except accounts.AccountError as exc:
        return jsonify({"code": exc.code, "error": str(exc)}), 400

    token = accounts.authenticate(user["email"], body.get("password", ""))
    return jsonify({"token": token, "user": user})


@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    accounts.logout(_bearer_token())
    # Idempotent: logging out an already-dead token is success, not an error.
    return jsonify({"ok": True})


@app.route("/api/auth/me")
def auth_me():
    user = _current_user()
    if user is None:
        return _auth_required()
    link = accounts.fpl_link(user["id"])
    return jsonify(
        {
            "user": user,
            "fpl": link,
            "fpl_connected": accounts.has_cookie(user["id"]),
            "methods": accounts.login_methods(user["id"]),
            # Carried here rather than only on the digest preview, which cannot
            # be built before the season starts — the toggle needs a source of
            # truth that exists all year.
            "deadline_email": accounts.wants_deadline_email(user["id"]),
        }
    )


@app.route("/api/me/fpl", methods=["POST"])
def me_link_fpl():
    """Link the account to an FPL entry by Team ID, verifying it exists."""
    user = _current_user()
    if user is None:
        return _auth_required()

    body = request.get_json(silent=True) or {}
    try:
        entry_id = int(body.get("entry_id"))
    except (TypeError, ValueError):
        return jsonify({"code": "bad_entry_id", "error": "entry_id must be a number"}), 400

    entry = fpl_client.entry(entry_id)
    if entry is None:
        return jsonify(
            {
                "code": "entry_not_found",
                "error": f"No FPL team found with ID {entry_id} this season.",
            }
        ), 404

    link = accounts.link_fpl(
        user["id"],
        entry_id,
        team_name=entry.get("name", ""),
        manager_name=f"{entry.get('player_first_name', '')} "
        f"{entry.get('player_last_name', '')}".strip(),
    )
    return jsonify({"fpl": link})


@app.route("/api/me/fpl-cookie", methods=["POST", "DELETE"])
def me_fpl_cookie():
    """Store (or remove) the user's own FPL session cookie.

    Validated against their linked entry before storing: a cookie that doesn't
    authenticate today will not magically start working tomorrow, and rejecting
    it now is the only moment the user is present to fix it. Write-only — no
    GET exists, and no response ever contains the value.
    """
    user = _current_user()
    if user is None:
        return _auth_required()

    if request.method == "DELETE":
        accounts.delete_cookie(user["id"])
        return jsonify({"ok": True, "fpl_connected": False})

    link = accounts.fpl_link(user["id"])
    if link is None:
        return jsonify(
            {"code": "no_fpl_link", "error": "Link your FPL Team ID first."}
        ), 400

    body = request.get_json(silent=True) or {}
    cookie = (body.get("cookie") or "").strip()
    if cookie.lower().startswith("cookie:"):
        cookie = cookie.split(":", 1)[1].strip()
    if not cookie:
        return jsonify({"code": "empty_cookie", "error": "No cookie provided."}), 400

    try:
        payload = fpl_client.my_team(link["entry_id"], cookie=cookie)
    except fpl_client.NotAuthenticated:
        return jsonify(
            {
                "code": "cookie_rejected",
                "error": "FPL did not accept that cookie. Copy the full Cookie "
                "header from a logged-in fantasy.premierleague.com tab.",
            }
        ), 400
    if payload is None:
        return jsonify(
            {
                "code": "cookie_mismatch",
                "error": "That cookie authenticates, but not for the linked team.",
            }
        ), 400

    accounts.store_cookie(user["id"], cookie)
    return jsonify({"ok": True, "fpl_connected": True})


@app.route("/api/me/team")
def me_team():
    """The signed-in user's squad — their linked entry, their vaulted cookie."""
    user = _current_user()
    if user is None:
        return _auth_required()
    link = accounts.fpl_link(user["id"])
    if link is None:
        return jsonify(
            {"code": "no_fpl_link", "error": "Link your FPL Team ID first."}
        ), 400

    horizon = _int_arg("horizon", 5, 1, service.MAX_HORIZON)
    body, status = _team_response(
        link["entry_id"], horizon, cookie=accounts.cookie_for(user["id"])
    )
    return jsonify(body), status


@app.route("/api/me/transfers")
def me_transfers():
    """The signed-in user's transfer history, with player names resolved."""
    user = _current_user()
    if user is None:
        return _auth_required()
    link = accounts.fpl_link(user["id"])
    if link is None:
        return jsonify(
            {"code": "no_fpl_link", "error": "Link your FPL Team ID first."}
        ), 400

    raw = fpl_client.transfers(link["entry_id"])
    if raw is None:
        return jsonify(
            {"code": "entry_not_found", "error": "Could not read transfer history."}
        ), 404

    data = fpl_client.bootstrap()
    elements = {int(p["id"]): p for p in data.get("elements", [])}

    def _name(element_id):
        element = elements.get(int(element_id or 0), {})
        return element.get("web_name", "Unknown")

    return jsonify(
        {
            "transfers": [
                {
                    "gameweek": t.get("event"),
                    "in": _name(t.get("element_in")),
                    "in_cost": (t.get("element_in_cost") or 0) / 10,
                    "out": _name(t.get("element_out")),
                    "out_cost": (t.get("element_out_cost") or 0) / 10,
                    "time": t.get("time"),
                }
                for t in raw
            ],
            "meta": fpl_client.meta(),
        }
    )


@app.errorhandler(vision.VisionUnavailable)
def _handle_vision_unavailable(error):
    return jsonify(error.to_dict()), error.status


@app.errorhandler(squad_import.ImportError_)
def _handle_import_error(error):
    return jsonify({"code": error.code, "error": str(error)}), 400


# Anonymous, so throttled harder than the authenticated routes: each call costs
# a model invocation against the operator's own subscription, and the whole
# point of the feature is that a stranger can use it without signing up.
_IMPORT_LIMIT = int(os.getenv("IMPORT_ATTEMPTS_PER_WINDOW", 5))
_IMPORT_WINDOW_SECONDS = int(os.getenv("IMPORT_WINDOW_SECONDS", 900))
_import_attempts: dict[str, list[float]] = {}


def _import_throttled(client: str) -> bool:
    now = time.time()
    window = [
        t for t in _import_attempts.get(client, []) if now - t < _IMPORT_WINDOW_SECONDS
    ]
    if len(window) >= _IMPORT_LIMIT:
        _import_attempts[client] = window
        return True
    window.append(now)
    _import_attempts[client] = window
    return False


@app.route("/api/import/screenshot", methods=["POST"])
def import_screenshot():
    """Read a squad out of an uploaded screenshot and rate it.

    Deliberately unauthenticated. Team ID and the session connect both require
    a visitor who is already committed; this is the on-ramp for someone who
    arrived thirty seconds ago and wants to see whether the tool is any good.

    Accepts a multipart file or a base64 body, because a web file picker and a
    mobile client naturally produce different things.
    """
    if not vision.is_configured():
        return jsonify(
            {
                "code": "vision_not_configured",
                "error": "Screenshot import isn't enabled on this server.",
            }
        ), 503
    if _import_throttled(_client_id()):
        return jsonify(
            {
                "code": "throttled",
                "error": "Too many screenshots. Try again in a few minutes.",
            }
        ), 429

    upload = request.files.get("image")
    if upload is not None:
        payload, mime = upload.read(), (upload.mimetype or "")
    else:
        body = request.get_json(silent=True) or {}
        payload, mime = body.get("image", ""), body.get("mime", "")
    if not payload:
        return jsonify({"code": "bad_image", "error": "No image was uploaded."}), 400

    result = service.import_screenshot(payload, mime)

    # Return full player payloads so the client can draw the squad on a pitch
    # and open the same stats sheet it uses everywhere else. A thinner shape
    # would force a second, parallel player model on the client.
    projection_set = service.projections_for(
        service.DEFAULT_HORIZON, ml_scorer.DEFAULT_ENGINE
    )
    elements = {int(e["id"]): e for e in projection_set.data.get("elements", [])}
    teams = {
        int(t["id"]): t.get("short_name", "UNK")
        for t in projection_set.data.get("teams", [])
    }
    fcps_scores, _ = service.fcps_for()

    def _full(row):
        element = elements.get(int(row["id"]))
        if element is None:
            return None
        payload = _player_payload(
            element,
            teams.get(int(element.get("team", 0)), "UNK"),
            projection_set.projections.get(int(row["id"]), {}),
            fcps_scores.get(int(row["id"])),
        )
        payload["starting_eleven"] = bool(row.get("starting_eleven"))
        return payload

    starters, bench = squad_import.best_eleven(result["players"])
    for row in starters:
        row["starting_eleven"] = True
    lineup = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
    for row in starters:
        full = _full(row)
        if full and full["position"] in lineup:
            lineup[full["position"]].append(full)
    result["lineup"] = lineup
    result["bench"] = [p for p in (_full(r) for r in bench) if p]
    result["meta"] = fpl_client.meta()
    return jsonify(result)


@app.route("/api/import/config")
def import_config():
    """Whether the client should offer screenshot upload at all."""
    return jsonify({"screenshot": vision.is_configured()})


@app.route("/api/me/digest")
def me_digest():
    """Preview the pre-deadline briefing for the signed-in user.

    Exists so a user can see exactly what they are opting into before they
    agree to be emailed, and so the wording can be checked without waiting for
    a Friday.
    """
    user = _current_user()
    if user is None:
        return _auth_required()
    link = accounts.fpl_link(user["id"])
    if link is None:
        return jsonify(
            {"code": "no_fpl_link", "error": "Link your FPL Team ID first."}
        ), 400

    briefing = service.deadline_digest(
        link["entry_id"], manager_name=link.get("manager_name", "")
    )
    return jsonify(
        {
            **briefing,
            "text": digest.render_text(briefing),
            "subscribed": accounts.wants_deadline_email(user["id"]),
            "email_configured": mailer.is_configured(),
        }
    )


@app.route("/api/me/digest/subscribe", methods=["POST", "DELETE"])
def me_digest_subscribe():
    """Opt in or out. Opt-in by default absent: nobody signed up to be emailed."""
    user = _current_user()
    if user is None:
        return _auth_required()
    accounts.set_deadline_email(user["id"], request.method == "POST")
    return jsonify({"subscribed": request.method == "POST"})


@app.route("/api/me/leagues")
def me_leagues():
    """The signed-in user's mini-leagues, discovered from their entry.

    Never asked for. Making a user hunt for a League ID is making them leave.
    """
    user = _current_user()
    if user is None:
        return _auth_required()
    link = accounts.fpl_link(user["id"])
    if link is None:
        return jsonify(
            {"code": "no_fpl_link", "error": "Link your FPL Team ID first."}
        ), 400

    entry = fpl_client.entry(link["entry_id"])
    if entry is None:
        return jsonify(
            {"code": "entry_not_found", "error": "Could not read your entry."}
        ), 404

    all_leagues = leagues.classify_leagues(entry)
    return jsonify(
        {
            "leagues": [le for le in all_leagues if le["meaningful"]],
            # Kept, but separated: FPL auto-enrols everyone into country and
            # club leagues with millions of members, and ranking advice about
            # those is noise dressed as insight.
            "auto_enrolled": [le for le in all_leagues if not le["meaningful"]],
            "meta": fpl_client.meta(),
        }
    )


@app.route("/api/me/league/<int:league_id>")
def me_league(league_id):
    """Standings, rival exposure and a chase-or-protect read for one league."""
    user = _current_user()
    if user is None:
        return _auth_required()
    link = accounts.fpl_link(user["id"])
    if link is None:
        return jsonify(
            {"code": "no_fpl_link", "error": "Link your FPL Team ID first."}
        ), 400

    result = service.league_analysis(
        link["entry_id"],
        league_id,
        horizon=_int_arg("horizon", service.DEFAULT_HORIZON, 1, service.MAX_HORIZON),
    )
    result["meta"] = fpl_client.meta()
    return jsonify(result)


@app.route("/api/me/recommendations")
def me_recommendations():
    """AI transfer advice for the signed-in user's linked team."""
    user = _current_user()
    if user is None:
        return _auth_required()
    link = accounts.fpl_link(user["id"])
    if link is None:
        return jsonify(
            {"code": "no_fpl_link", "error": "Link your FPL Team ID first."}
        ), 400

    result = service.recommendations(
        entry_id=link["entry_id"],
        horizon=_int_arg("horizon", service.DEFAULT_HORIZON, 1, service.MAX_HORIZON),
        max_transfers=_int_arg("max_transfers", service.MAX_SEARCH_TRANSFERS, 0, 3),
        free_transfers_override=_int_arg("free_transfers", None, 0, 5),
        bank_override=_int_arg("bank", None, 0),
        include_hits=_bool_arg("include_hits", True),
        engine=_engine_arg(),
    )
    if narrative.would_call(result["plans"]):
        try:
            llm_budget.check_client(_client_id())
        except llm_budget.ClientThrottled:
            return jsonify(result)
    narrative.annotate(result["plans"])
    return jsonify(result)


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

    # Strictly additive. A throttled caller still gets their fully computed,
    # non-LLM recommendations — narration is the garnish, never the dish — so
    # being over the limit drops the prose rather than failing the request.
    # Metered only when a call would actually be spent.
    if narrative.would_call(result["plans"]):
        try:
            llm_budget.check_client(_client_id())
        except llm_budget.ClientThrottled:
            return jsonify(result)
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

    refresh = _bool_arg("refresh", False)

    # Only charged when a call could actually happen. A cache hit is free, and
    # `refresh` is the one way a client can force a spend, so it must be metered.
    if refresh or not service.fcps_is_cached(user_id):
        try:
            llm_budget.check_client(_client_id())
        except llm_budget.ClientThrottled as error:
            return jsonify({"code": "fcps_throttled", "error": str(error)}), 429

    return jsonify(service.fcps_advice(entry_id=user_id, refresh=refresh))


@app.route("/api/draft-squad")
def draft_squad():
    """A recommended opening fifteen, for the window before the GW1 deadline.

    Every other advice route needs a squad to read, and FPL publishes none until
    the deadline passes — so for the three weeks when the only task is picking a
    team, the app had nothing to say. This is that gap.

    The result does not depend on `user_id`: there are no picks to personalise
    against. One squad, computed once, served to everyone.
    """
    pinned = [p.strip() for p in (request.args.get("pin") or "").split(",") if p.strip()]

    # Only metered when prose would actually be generated. The squad itself is
    # arithmetic and free; a cached summary costs nothing either.
    if not service.draft_summary_is_cached(
        horizon=_int_arg("horizon", service.DEFAULT_HORIZON, 1, service.MAX_HORIZON),
        engine=_engine_arg(),
        pinned=pinned,
    ):
        try:
            llm_budget.check_client(_client_id())
        except llm_budget.ClientThrottled as error:
            return jsonify({"code": "draft_throttled", "error": str(error)}), 429

    return jsonify(
        service.draft_squad(
            horizon=_int_arg("horizon", service.DEFAULT_HORIZON, 1, service.MAX_HORIZON),
            engine=_engine_arg(),
            pinned=pinned,
        )
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
            "scoring": ml_scorer.describe(team_games=_team_games_played()),
            "fcps": {
                "available": fcps_llm.is_configured(),
                "model": fcps_llm.model_name(),
                "effort": fcps_llm.effort_level(),
                "cache_ttl_seconds": fcps_llm.CACHE_TTL_SECONDS,
                "reason": None
                if fcps_llm.is_configured()
                else "The Claude CLI is not installed or not on this server's PATH.",
            },
            "narrative": {
                "available": narrative.is_enabled(),
                "model": narrative.model_name(),
            },
            "research": research.status(),
            # Surfaced so the client can explain a 429 rather than showing a
            # retry button for a condition retrying won't fix. Counts only —
            # nothing here identifies a caller.
            "budget": {
                "daily_ceiling": llm_budget.DAILY_CALL_CEILING,
                "remaining_today": llm_budget.remaining_today(),
                "client_hourly_limit": llm_budget.CLIENT_CALLS_PER_HOUR,
            },
            "meta": fpl_client.meta(),
        }
    )


def _global_captain_ranking(limit: int):
    """Best armband candidates in the league, for anyone — no squad required.

    The squad-scoped ranking below can only answer "who, of my fifteen?", which
    needs a signed-in user and a season already under way. This answers "who,
    of everyone?", which is the question a visitor with no account is asking,
    and it is the one that can be published on a public page.
    """
    projection_set = service.projections_for(1, _engine_arg())
    projections, data = projection_set.projections, projection_set.data
    teams = {int(t["id"]): t.get("short_name", "UNK") for t in data.get("teams", [])}
    elements = {int(e["id"]): e for e in data.get("elements", [])}

    ranked = []
    for pid, projection in projections.items():
        element = elements.get(pid)
        if element is None:
            continue
        xpts_next = float(projection.get("xpts_next") or 0.0)
        if xpts_next <= 0:
            continue
        # A blank gameweek cannot be captained; an unavailable player should not
        # be suggested for the one pick that doubles.
        fixtures = (projection.get("per_gameweek") or [{}])[0].get("fixtures") or []
        if not fixtures:
            continue
        ranked.append(
            {
                "id": pid,
                "web_name": element.get("web_name", ""),
                "team": teams.get(int(element.get("team", 0)), "UNK"),
                "position": rules.position_of(element),
                "price": int(element.get("now_cost", 0)) / 10,
                "fixtures": fixtures,
                "xpts": round(xpts_next, 2),
                "xpts_captained": round(xpts_next * 2, 2),
                "minutes_risk": projection.get("minutes_risk", "medium"),
                "selected_by_percent": _safe_float(element.get("selected_by_percent")),
            }
        )

    ranked.sort(key=lambda r: (-r["xpts"], r["id"]))
    return ranked[:limit]


@app.route("/api/captain")
def captain_picks():
    """Rank candidates for the armband.

    With ``user_id``, ranks that manager's own squad. Without it, ranks the
    whole league — which works before a ball is kicked and needs no account.
    """
    user_id = request.args.get("user_id", type=int)

    if user_id is None:
        state = fpl_client.season_state() or {}
        return jsonify(
            {
                "scope": "global",
                "gameweek": state.get("gameweek"),
                "deadline": state.get("deadline"),
                "picks": _global_captain_ranking(_int_arg("limit", 20, 1, 100)),
                "meta": fpl_client.meta(),
            }
        )

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
