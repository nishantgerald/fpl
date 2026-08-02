"""Adapts the trained model into the projection shape the optimiser consumes.

The division of labour, which is the whole architectural bet of this pass:

* The **ML model** says how many points a player will score. It is good at that
  and knows nothing about FPL's rules.
* The **optimiser** (:mod:`engine.optimizer`) enumerates only squads that are
  legal — 15 players, 2/5/5/3, three per club, affordable at real selling prices,
  a fieldable formation — and knows nothing about football.

Neither is useful alone. A model picking a squad directly produces the best
fifteen players in the league, which is not a squad anyone can own. A rules
engine with no model picks legally and badly. So the model scores and the
optimiser searches, and the interface between them is this module: a
``{player_id: projection}`` mapping identical in shape to
:func:`engine.xpts.project_all`, so ``optimizer.optimise`` is unchanged and can't
tell which engine produced its inputs.

Three engines are selectable at the API, and all three go through the same
optimiser:

``xpts``   the hand-built component model. Always available, no dependencies.
``ml``     the trained model. Falls back to ``xpts`` when no artifact is present.
``blend``  the mean of the two. Usually the safest default in production: two
           uncorrelated errors averaged is a smaller error, and a regression in
           one engine is halved rather than shipped whole.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from . import rules, xpts

ENGINES = ("xpts", "ml", "blend")
DEFAULT_ENGINE = "xpts"

BLEND_WEIGHT = 0.5


def is_available() -> bool:
    """Whether a trained artifact is loadable in this process."""
    predictor = _predictor()
    return predictor is not None


def describe() -> dict:
    """What the client shows in the engine picker, without guessing."""
    predictor = _predictor()
    if predictor is None:
        return {
            "available": False,
            "engines": ["xpts"],
            "reason": (
                "No trained model artifact on this server. Run `python -m ml.train` "
                "and redeploy to enable the ML engine."
            ),
        }
    metadata = predictor.metadata
    return {
        "available": True,
        "engines": list(ENGINES),
        "model": predictor.name,
        "trained_on": predictor.trained_on,
        "features": len(predictor.feature_names),
        "test_scores": metadata.get("test_scores", {}),
        "selection_metric": metadata.get("selection_metric"),
    }


def _predictor():
    """Import lazily — the web process must start without numpy or sklearn."""
    try:
        from ml.predict import get_predictor
    except ImportError:
        return None
    return get_predictor()


# ---------------------------------------------------------------- projections


def project_all(
    elements: Sequence[Mapping],
    fixtures: Sequence[Mapping],
    teams: Sequence[Mapping],
    events: Sequence[Mapping],
    from_gameweek: int,
    horizon: int = 5,
    engine: str = "ml",
) -> tuple[dict[int, dict], str]:
    """Projections for every player. Returns ``(projections, engine_actually_used)``.

    The second element is not decoration: when the ML artifact is missing the
    caller asked for one engine and got another, and the response says so rather
    than silently labelling hand-built numbers as model output.
    """
    engine = engine if engine in ENGINES else DEFAULT_ENGINE

    baseline = xpts.project_all(
        elements, fixtures, teams, events, from_gameweek, horizon
    )
    if engine == "xpts":
        return baseline, "xpts"

    predictor = _predictor()
    if predictor is None:
        return baseline, "xpts"

    gameweeks = [entry["gameweek"] for entry in _any_per_gameweek(baseline)] or [
        from_gameweek
    ]
    fixture_index = xpts.build_fixture_index(fixtures, from_gameweek)
    team_games = xpts.team_games_played(teams, events)

    try:
        raw = predictor.score(elements, fixture_index, gameweeks, team_games)
    except Exception as error:  # never let inference take down the endpoint
        print(f"[ml] scoring failed, falling back to xpts: {type(error).__name__}: {error}")
        return baseline, "xpts"

    elements_by_id = {int(e["id"]): e for e in elements}
    projections: dict[int, dict] = {}

    for pid, base in baseline.items():
        model_row = raw.get(pid)
        if not model_row:
            projections[pid] = base
            continue
        element = elements_by_id.get(pid)
        if element is None:
            projections[pid] = base
            continue
        projections[pid] = _merge(base, model_row, element, engine)

    return projections, engine


def _merge(base: Mapping, model_row: Mapping[int, float], element: Mapping, engine: str) -> dict:
    """Rebuild one projection with model numbers, preserving every other field.

    The component decomposition (``goals``, ``clean_sheet``, ...) belongs to the
    hand-built model and cannot be recovered from a gradient-boosted prediction.
    Rather than fabricate a plausible-looking breakdown, the components are
    rescaled to sum to the model's total and flagged ``components_estimated`` —
    the UI shows the waterfall as indicative and says which engine produced it.
    """
    per_gameweek = []
    for offset, entry in enumerate(base.get("per_gameweek", ())):
        gameweek = entry["gameweek"]
        model_points = model_row.get(gameweek)
        if model_points is None:
            per_gameweek.append(dict(entry))
            continue

        # Availability is applied here, not learned: the archive has no injury
        # flags, so the model predicts points conditional on the player featuring
        # and this is the same factor engine.xpts uses.
        model_points *= xpts.availability_factor(element, offset)

        if engine == "blend":
            model_points = (
                BLEND_WEIGHT * model_points + (1.0 - BLEND_WEIGHT) * entry["xpts"]
            )

        components = _rescale(entry.get("components", {}), entry["xpts"], model_points)
        per_gameweek.append(
            {
                "gameweek": gameweek,
                "xpts": round(model_points, 2),
                "fixtures": entry.get("fixtures", []),
                "components": components,
                "components_estimated": True,
                "xpts_baseline": entry["xpts"],
            }
        )

    horizon = round(sum(g["xpts"] for g in per_gameweek), 2)
    price = int(element.get("now_cost", 0)) or 1

    merged = dict(base)
    merged.update(
        {
            "per_gameweek": per_gameweek,
            "horizon_xpts": horizon,
            "xpts_next": per_gameweek[0]["xpts"] if per_gameweek else 0.0,
            "xpts_per_million": round(horizon / (price / 10.0), 3),
            "engine": engine,
        }
    )
    return merged


def _rescale(components: Mapping[str, float], old_total: float, new_total: float) -> dict:
    if not components:
        return {}
    if old_total <= 0:
        share = new_total / max(1, len(components))
        return {key: round(share, 3) for key in components}
    scale = new_total / old_total
    return {key: round(value * scale, 3) for key, value in components.items()}


def _any_per_gameweek(projections: Mapping[int, Mapping]) -> Sequence[Mapping]:
    for projection in projections.values():
        entries = projection.get("per_gameweek")
        if entries:
            return entries
    return ()


def position_of(element: Mapping) -> str:
    """Re-exported so callers don't need to import two modules for one lookup."""
    return rules.position_of(element)
