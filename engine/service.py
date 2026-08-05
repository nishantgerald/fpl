"""Assembles engine outputs into API responses.

This is the seam between the pure engine (``rules``, ``money``, ``xpts``,
``optimizer``) and the network (``fpl_client``). Keeping it thin means the
decision path stays testable without a socket.

Three scoring systems live behind this module and they are deliberately not
merged:

``xpts``   :mod:`engine.xpts` — a hand-built component model, in points.
``ml``     :mod:`ml.predict` via :mod:`engine.ml_scorer` — a trained model, in
           points. Falls back to ``xpts`` when no artifact is deployed.
``fcps``   :mod:`engine.fcps` — the incumbent 0-1000 composite, plus
           :mod:`engine.fcps_llm`, which writes it up in prose.

xPts and ML are interchangeable inputs to the same optimiser. FCPS is not: it
isn't denominated in points, so it can rank players but cannot price a -4 hit,
and it drives its own written-advice endpoint rather than the optimiser.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from . import fcps as fcps_mod
from . import fcps_llm, fpl_client, reconstruct, research
from . import free_transfers as ft_mod
from . import ml_scorer, money, optimizer, rules

DEFAULT_HORIZON = 5
MAX_HORIZON = 8
MAX_SEARCH_TRANSFERS = 3


class ServiceError(Exception):
    """A user-facing failure with a machine-readable code."""

    def __init__(self, code: str, message: str, status: int = 400, **extra):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.extra = extra

    def to_dict(self) -> dict:
        return {"code": self.code, "error": self.message, **self.extra}


# ---------------------------------------------------------------- squad state


def current_picks(entry_id: int, gameweek: int) -> dict | None:
    """The squad the manager owns *now*, not as of the last deadline.

    Fetches the last locked picks and folds in any transfers made since —
    which are public and irreversible the moment they're confirmed. Returns a
    picks-shaped payload with ``source: "reconstructed"`` and an exact bank,
    or ``None`` when nothing has changed since the deadline (the common case),
    so the caller keeps the authoritative payload.

    Free Hit is the one chip that rewrites history: its squad evaporates when
    its gameweek ends, so the base rolls back one week and the Free Hit's own
    transfers are excluded. Wildcards change nothing here — their transfers
    are ordinary and permanent.
    """
    transfers = fpl_client.transfers(entry_id)
    if not transfers:
        return None
    if not any(int(t.get("event") or 0) > gameweek for t in transfers):
        return None

    history = fpl_client.history(entry_id) or {}
    base_gw = reconstruct.base_gameweek(gameweek, history.get("chips") or [])
    base = fpl_client.picks(entry_id, base_gw)
    if not base or not base.get("picks"):
        return None
    return reconstruct.reconstruct(base, transfers, gameweek)


def load_squad_state(entry_id: int, gameweek: int) -> dict:
    """The manager's squad, bank and free transfers, from public data only.

    Raises :class:`ServiceError` with ``entry_not_found`` if the id doesn't
    resolve — FPL reissues entry ids each season, so a saved id going stale is
    an expected condition, not an error.
    """
    data = fpl_client.bootstrap()
    elements = {int(p["id"]): p for p in data.get("elements", [])}

    picks_payload = fpl_client.picks(entry_id, gameweek)
    if not picks_payload or not picks_payload.get("picks"):
        raise ServiceError(
            "entry_not_found",
            f"No FPL squad found for entry {entry_id} in gameweek {gameweek}.",
            status=404,
        )

    # Advice about transfers must start from the squad the manager owns *now*.
    # Recommending the sale of a player they already sold on Tuesday isn't a
    # recommendation, it's a bug report about our own data.
    rebuilt = current_picks(entry_id, gameweek)
    transfers_pending = 0
    if rebuilt is not None:
        picks_payload = rebuilt
        transfers_pending = int(rebuilt.get("transfers_applied") or 0)

    squad: list[Mapping] = []
    missing: list[int] = []
    for pick in picks_payload["picks"]:
        element = elements.get(int(pick["element"]))
        if element is None:
            # A pick we can't resolve is skipped rather than crashing the
            # request — the old organize_team() raised KeyError here.
            missing.append(int(pick["element"]))
            continue
        squad.append(element)

    entry_history = picks_payload.get("entry_history") or {}
    bank = entry_history.get("bank")
    squad_value = entry_history.get("value")

    if (bank is None or squad_value is None) and not transfers_pending:
        entry_payload = fpl_client.entry(entry_id) or {}
        bank = bank if bank is not None else entry_payload.get("last_deadline_bank")
        squad_value = entry_payload.get("last_deadline_value")

    if transfers_pending:
        # The reconstructed bank is exact; the deadline-time squad value is
        # not, because the squad it priced no longer exists. Dropping it sends
        # selling prices down the estimation path instead of apportioning a
        # stale total across a changed squad.
        squad_value = None

    bank = int(bank) if bank is not None else 0
    # `value` is squad selling value *plus* bank, so the squad's own selling
    # total is the difference.
    selling_total = (int(squad_value) - bank) if squad_value is not None else None

    selling_prices = money.estimate_selling_prices(squad, selling_total)
    confidence = money.selling_price_confidence(squad, selling_total)

    history_payload = fpl_client.history(entry_id) or {}
    derived_ft = ft_mod.derive_free_transfers(
        history_payload.get("current"), history_payload.get("chips")
    )
    # Transfers already made this week consumed free transfers before we got
    # here; history won't show that until the next deadline. Advice priced
    # against transfers the manager no longer has would systematically
    # under-count hits.
    derived_ft = max(0, derived_ft - transfers_pending)

    return {
        "squad": squad,
        "bank": bank,
        "selling_prices": selling_prices,
        "selling_price_confidence": confidence,
        "squad_selling_value": selling_total
        if selling_total is not None
        else sum(selling_prices.values()),
        "selling_price_estimated": selling_total is not None,
        "free_transfers": derived_ft,
        "active_chip": picks_payload.get("active_chip"),
        "picks": picks_payload["picks"],
        "unresolved_picks": missing,
    }


# ---------------------------------------------------------------- projections


# Projecting ~700 players over 8 gameweeks is cheap but not free, and every
# endpoint needs it. Memoised against the bootstrap payload's identity, so it
# recomputes exactly when the underlying data changes and not otherwise.
_projection_cache: dict[tuple, tuple] = {}
_PROJECTION_CACHE_SIZE = 12


class Projection:
    """The result of projecting every player, plus which engine produced it.

    A tuple would have been fine until the engine became selectable. It isn't
    now: the caller needs to know that it asked for ``ml`` and got ``xpts``
    because no artifact is deployed, and a fourth positional element is exactly
    the kind of thing that gets unpacked in the wrong order.
    """

    __slots__ = ("projections", "gameweeks", "data", "engine", "engine_requested")

    def __init__(self, projections, gameweeks, data, engine, engine_requested):
        self.projections = projections
        self.gameweeks = gameweeks
        self.data = data
        self.engine = engine
        self.engine_requested = engine_requested

    @property
    def fell_back(self) -> bool:
        return self.engine != self.engine_requested

    def __iter__(self):
        """Backwards compatible with ``projections, gameweeks, data = ...``."""
        return iter((self.projections, self.gameweeks, self.data))


def projections_for(horizon: int, engine: str = ml_scorer.DEFAULT_ENGINE) -> Projection:
    """Project every player over ``horizon`` gameweeks with the chosen engine."""
    engine = engine if engine in ml_scorer.ENGINES else ml_scorer.DEFAULT_ENGINE

    data = fpl_client.bootstrap()
    fixtures = fpl_client.fixtures() or []
    state = fpl_client.season_state()
    if not state:
        raise ServiceError(
            "upstream_unavailable", "Could not read the FPL gameweek state.", status=503
        )

    gameweek = state["gameweek"]
    events = data.get("events", [])

    # Keyed on the bootstrap object's identity, and we keep a reference to it so
    # the id can't be recycled onto a different payload.
    key = (horizon, gameweek, engine)
    cached = _projection_cache.get(key)
    if cached is not None and cached[0] is data:
        return cached[1]

    projections, engine_used = ml_scorer.project_all(
        data.get("elements", []),
        fixtures,
        data.get("teams", []),
        events,
        gameweek,
        horizon,
        engine=engine,
    )
    gameweeks = [
        g["gameweek"]
        for g in next(iter(projections.values()), {}).get("per_gameweek", [])
    ] or [gameweek]

    result = Projection(projections, gameweeks, data, engine_used, engine)
    if len(_projection_cache) >= _PROJECTION_CACHE_SIZE:
        _projection_cache.clear()
    _projection_cache[key] = (data, result)
    return result


# ---------------------------------------------------------------- FCPS

_fcps_cache: dict[int, tuple] = {}


def fcps_for() -> tuple[dict[int, dict], dict]:
    """FCPS for every player. Returns ``(scores, bootstrap_data)``.

    Kept separate from :func:`projections_for` because FCPS has no horizon — it
    is defined against a fixed 3-fixture lookahead — and because the two scores
    must stay independently computed. Blending them would produce a third number
    that is neither, which is how a product ends up with three scores and no
    answer.
    """
    data = fpl_client.bootstrap()
    fixtures = fpl_client.fixtures() or []
    state = fpl_client.season_state()
    gameweek = state["gameweek"] if state else 1

    cached = _fcps_cache.get(gameweek)
    if cached is not None and cached[0] is data:
        return cached[1], data

    scores = fcps_mod.score_all(data.get("elements", []), fixtures, gameweek)
    _fcps_cache.clear()
    _fcps_cache[gameweek] = (data, scores)
    return scores, data


def _research_digest() -> str | None:
    """Today's news digest, when the feature is on and a fresh one exists.

    Deliberately best-effort: the column is worth writing without it, so a
    missing or stale digest degrades the advice rather than failing the request.
    """
    try:
        if not research.is_enabled():
            return None
        record = research.current()
        return record["digest"] if record else None
    except Exception:
        return None


def fcps_is_cached(entry_id: int) -> bool:
    """Whether advice for this entry is already cached, so no call is needed.

    Lets the route meter only requests that could actually spend a model call —
    charging a cache hit against a caller's hourly share would throttle people
    for re-reading a page.

    Cheap and side-effect-free: it reads the season state, which is itself
    cached, and then does a dict/disk lookup. Never raises; an unknown state is
    reported as "not cached", which errs toward metering.
    """
    try:
        state = fpl_client.season_state()
        if not state or not state["started"]:
            return False
        key = (entry_id, state["gameweek"], fcps_llm.model_name())
        return fcps_llm.peek_cache(key) is not None
    except Exception:
        return False


def fcps_advice(entry_id: int, refresh: bool = False) -> dict:
    """The FCPS transfer column: rank by FCPS, then ask the model to write it up.

    Raises :class:`engine.fcps_llm.FcpsUnavailable` when the server has no API
    key or the model call fails — never a 200 with an error message where the
    advice should be.
    """
    state = fpl_client.season_state()
    if not state:
        raise ServiceError(
            "upstream_unavailable", "Could not read the FPL gameweek state.", status=503
        )
    if not state["started"]:
        raise ServiceError(
            "season_not_started",
            "FCPS advice starts once the season is under way.",
            status=503,
            gameweek=state["gameweek"],
            gameweek_name=state["gameweek_name"],
            deadline=state["deadline"],
        )

    squad_state = load_squad_state(entry_id, state["gameweek"])
    scores, data = fcps_for()
    teams = {int(t["id"]): t.get("short_name", "UNK") for t in data.get("teams", [])}
    starting = {
        int(p["element"]) for p in squad_state["picks"] if int(p.get("multiplier", 0)) > 0
    }

    squad_rows = [
        fcps_llm.player_row(
            element,
            scores.get(int(element["id"]), {}),
            teams.get(int(element.get("team", 0)), "UNK"),
            in_squad=True,
            starting=int(element["id"]) in starting,
        )
        for element in squad_state["squad"]
        if int(element["id"]) in scores
    ]
    squad_rows.sort(key=lambda r: (r["position"], -r["fcps"]))

    squad_ids = {int(e["id"]) for e in squad_state["squad"]}
    shortlist = [
        element
        for element in fcps_mod.top_by_position(scores, data.get("elements", []))
        if int(element["id"]) not in squad_ids
    ]
    shortlist_rows = [
        fcps_llm.player_row(
            element,
            scores[int(element["id"])],
            teams.get(int(element.get("team", 0)), "UNK"),
        )
        for element in shortlist
    ]

    result = fcps_llm.advise(
        squad_rows=squad_rows,
        shortlist_rows=shortlist_rows,
        gameweek=state["gameweek"],
        bank=squad_state["bank"],
        free_transfers=squad_state["free_transfers"],
        cache_key=(entry_id, state["gameweek"], fcps_llm.model_name()),
        refresh=refresh,
        digest=_research_digest(),
    )
    result.update(
        {
            "squad": squad_rows,
            "shortlist": shortlist_rows[:40],
            "bank": squad_state["bank"],
            "free_transfers": squad_state["free_transfers"],
            "deadline": state["deadline"],
            "meta": fpl_client.meta({"gameweek": state["gameweek"]}),
        }
    )
    return result


# ---------------------------------------------------------------- recommendations


def recommendations(
    entry_id: int,
    horizon: int = DEFAULT_HORIZON,
    max_transfers: int = MAX_SEARCH_TRANSFERS,
    free_transfers_override: int | None = None,
    bank_override: int | None = None,
    include_hits: bool = True,
    engine: str = ml_scorer.DEFAULT_ENGINE,
) -> dict:
    """The headline endpoint: rules-legal transfer advice, scored by ``engine``.

    ``engine`` chooses *what scores a player* — the hand-built component model,
    the trained model, or their mean. It never changes what is legal: the search
    and the final re-verification are the same code either way, so an ML-scored
    plan is exactly as executable as an xPts-scored one.
    """
    horizon = max(1, min(MAX_HORIZON, int(horizon)))
    max_transfers = max(0, min(MAX_SEARCH_TRANSFERS, int(max_transfers)))

    state = fpl_client.season_state()
    if not state:
        raise ServiceError(
            "upstream_unavailable", "Could not read the FPL gameweek state.", status=503
        )
    if not state["started"]:
        raise ServiceError(
            "season_not_started",
            "The season hasn't kicked off yet — recommendations start at GW1.",
            status=503,
            gameweek=state["gameweek"],
            gameweek_name=state["gameweek_name"],
            deadline=state["deadline"],
        )

    squad_state = load_squad_state(entry_id, state["gameweek"])
    projection = projections_for(horizon, engine)
    projections, gameweeks, data = projection.projections, projection.gameweeks, projection.data

    bank = bank_override if bank_override is not None else squad_state["bank"]
    free_transfers = (
        free_transfers_override
        if free_transfers_override is not None
        else squad_state["free_transfers"]
    )

    quotas = rules.squad_quotas(data.get("element_types"))
    club_limit = rules.max_per_club(data.get("game_settings"))

    result = optimizer.optimise(
        squad=squad_state["squad"],
        all_elements=data.get("elements", []),
        projections=projections,
        gameweeks=gameweeks,
        bank=bank,
        free_transfers=free_transfers,
        selling_prices=squad_state["selling_prices"],
        quotas=quotas,
        club_limit=club_limit,
        max_transfers=max_transfers,
        include_hits=include_hits,
    )

    teams = {int(t["id"]): t for t in data.get("teams", [])}
    _decorate_team_names(result["plans"], teams)
    if result["hold"].get("best_rejected"):
        _decorate_team_names([result["hold"]["best_rejected"]], teams)

    warnings: list[str] = []
    if squad_state["selling_price_estimated"]:
        warnings.append(
            "Selling prices are estimated from public data. Pass ?bank= to override "
            "if you know your exact figures."
        )
    if squad_state["unresolved_picks"]:
        warnings.append(
            f"{len(squad_state['unresolved_picks'])} squad member(s) could not be "
            "resolved against the current player list and were excluded."
        )
    if squad_state["active_chip"]:
        warnings.append(
            f"You have the {squad_state['active_chip']} chip active this gameweek; "
            "transfer advice assumes normal rules."
        )
    if projection.fell_back:
        warnings.append(
            f"The {projection.engine_requested} engine isn't available on this "
            f"server, so these numbers come from the {projection.engine} model."
        )

    return {
        "gameweek": state["gameweek"],
        "deadline": state["deadline"],
        "horizon": horizon,
        "engine": projection.engine,
        "engine_requested": projection.engine_requested,
        "budget": {
            "bank": bank,
            "squad_selling_value": squad_state["squad_selling_value"],
            "selling_price_estimated": squad_state["selling_price_estimated"],
            "confidence": squad_state["selling_price_confidence"],
        },
        "free_transfers": free_transfers,
        "free_transfers_source": (
            "override" if free_transfers_override is not None else "derived"
        ),
        "recommendation": "transfer" if result["plans"] else "hold",
        "plans": result["plans"],
        "hold": result["hold"],
        "warnings": warnings,
        "meta": fpl_client.meta({"gameweek": state["gameweek"]}),
    }


def _decorate_team_names(plans: Sequence[Mapping], teams: Mapping[int, Mapping]) -> None:
    """Attach human-readable club short names to every player reference."""
    for plan in plans:
        for transfer in plan.get("transfers", ()):
            for side in ("out", "in"):
                ref = transfer.get(side)
                if ref is not None:
                    team = teams.get(ref.get("team_id"))
                    ref["team"] = str(team.get("short_name", "UNK")) if team else "UNK"


# ---------------------------------------------------------------- draft squad

# The recommended opening squad does not depend on who is asking — there are no
# picks to read before the deadline — so it is computed once for the whole site
# rather than per visitor. That also makes the written summary affordable: one
# model call a day serves everyone, instead of one per manager.
_draft_cache: dict[str, tuple] = {}
_DRAFT_TTL_SECONDS = 3600


def draft_squad(horizon: int = 5, engine: str = "xpts", pinned=()) -> dict:
    """A recommended opening fifteen, for the window before the GW1 deadline.

    Raises :class:`ServiceError` once the season is under way — at that point
    the real advice routes apply and a generic draft would be actively wrong.
    """
    import time as _time

    from . import draft

    state = fpl_client.season_state()
    if not state:
        raise ServiceError(
            "upstream_unavailable", "Could not read the FPL gameweek state.", status=503
        )
    if state["started"]:
        raise ServiceError(
            "season_in_progress",
            "The season has started — use /api/recommendations for your squad.",
            status=409,
            gameweek=state["gameweek"],
        )

    key = f"{horizon}:{engine}:{','.join(sorted(pinned))}"
    cached = _draft_cache.get(key)
    if cached and _time.time() - cached[0] < _DRAFT_TTL_SECONDS:
        return cached[1]

    projection = projections_for(horizon, engine)
    data = projection.data
    teams = {int(t["id"]): t.get("short_name", "UNK") for t in data.get("teams", [])}

    rows = draft.candidates(data.get("elements", []), projection.projections, teams)
    built = draft.build(rows, pinned=pinned)
    if built is None:
        raise ServiceError(
            "draft_unavailable",
            "Could not assemble a legal squad from the current player data.",
            status=503,
        )

    built.update(
        {
            "gameweek": state["gameweek"],
            "gameweek_name": state["gameweek_name"],
            "deadline": state["deadline"],
            "horizon": horizon,
            "engine": projection.engine,
            "engine_requested": projection.engine_requested,
            "pool_size": len(rows),
            "summary": _draft_summary(built),
        }
    )
    _draft_cache[key] = (_time.time(), built)
    return built


def _draft_summary(built: Mapping) -> dict:
    """The written rationale, or a machine-readable reason there isn't one.

    Best-effort by design: the squad is the product and it stands on its own
    numbers. Prose is an enhancement, so a spent budget or an absent CLI must
    degrade it rather than fail the request.
    """
    from . import draft_llm

    try:
        return draft_llm.summarise(built, digest=_research_digest())
    except Exception as error:
        return {"available": False, "reason": type(error).__name__}


def draft_summary_is_cached(horizon: int, engine: str, pinned=()) -> bool:
    """Whether a draft summary already exists, so the route can skip metering.

    Charging a caller's hourly share for a cached response would throttle people
    for reloading a page. Cheap and side-effect-free; never raises.
    """
    import time as _time

    try:
        key = f"{horizon}:{engine}:{','.join(sorted(pinned))}"
        cached = _draft_cache.get(key)
        return bool(cached and _time.time() - cached[0] < _DRAFT_TTL_SECONDS)
    except Exception:
        return False
