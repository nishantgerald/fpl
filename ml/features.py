"""Feature definitions. The only place a feature is allowed to be invented.

Two invariants govern everything here, and both are enforced by tests.

**1. No lookahead.** A feature for gameweek *g* is built only from rows with
gameweek < *g*, plus facts about *g* that are published before its deadline (the
fixture list, home/away, FPL's difficulty rating, and the player's price, which
is locked at the deadline). Mechanically this is a ``groupby(...).shift(1)``
before every cumulative sum: the shift is the whole safety property, and
:func:`assert_no_lookahead` re-derives one row by hand to check it.

**2. Feature parity.** Every feature has two derivations that must agree:

* :func:`build_frame` — from historical per-gameweek rows, by accumulating.
* :func:`from_bootstrap` — from a live ``bootstrap-static`` element, directly.

This constraint is why the feature set looks conservative. Several obviously
useful signals were dropped for failing it:

``minutes in the last 4 gameweeks``
    Strongly predictive; ``bootstrap-static`` publishes no rolling minutes, so a
    model trained on it would be blind at serve time. FPL's ``form`` field is the
    one recency signal available on both sides, so it is the only one used.
``team / opponent strength ratings``
    Present in ``teams.csv`` from 2020-21 and in every live bootstrap, but FPL
    rescaled them in 2022 and the archived and live scales do not match. FPL's
    per-fixture difficulty rating (``mean_fdr``) carries most of the same
    information on a stable 1-5 scale, so it is used instead.
``ownership``
    ``merged_gw`` records a raw ``selected`` count, and converting it to the
    percentage the live API publishes needs a total-players figure the archive
    doesn't carry.
``injury / availability flags``
    Not archived at all. Availability is applied *after* the model as a
    multiplier by :mod:`engine.ml_scorer`, exactly as :mod:`engine.xpts` does —
    the model predicts points conditional on the player being fit, and being
    unfit is a separate, better-observed fact.

Dropping a real signal to preserve parity costs accuracy in the backtest. Keeping
it would cost accuracy in production while *improving* the backtest, which is the
more expensive mistake and the harder one to notice.
"""

from __future__ import annotations

from typing import Mapping, Sequence

# The canonical feature order. Persisted with the model artifact; a mismatch
# between this list and the artifact's list is a hard error at load time, not a
# silently reordered vector.
FEATURES: tuple[str, ...] = (
    # Recent form — the only rolling window with a live analogue.
    "form",
    # Season-to-date productivity.
    "points_per_game",
    "pts_per_90",
    "minutes_per_team_game",
    "start_rate",
    "goals_per_90",
    "assists_per_90",
    "cs_per_game",
    "gc_per_90",
    "saves_per_90",
    # Underlying (Opta) rates. NaN before 2022-23; see `has_xstats`.
    "xg_per_90",
    "xa_per_90",
    "xgc_per_90",
    # Bonus-point system.
    "bps_per_90",
    "bonus_per_game",
    # ICT.
    "ict_per_game",
    "influence_per_game",
    "creativity_per_game",
    "threat_per_game",
    # Static / contextual.
    "price",
    "team_games_played",
    "is_gkp",
    "is_def",
    "is_mid",
    "is_fwd",
    # The target gameweek's fixture context — published before its deadline.
    "n_fixtures",
    "mean_fdr",
    "home_share",
    # FPL's own published expectation, as a feature rather than a competitor.
    "fpl_xp",
    # Regime flag: was Opta data available for this row at all.
    "has_xstats",
)

TARGET = "y_points"

# Carried through the frame but never fed to a model: the raw season-to-date
# quantities the FCPS baseline needs to be reconstructed exactly as the app
# computes it. Prefixed `bl_` so a model can never pick them up by accident.
BASELINE_INPUTS: tuple[str, ...] = ("bl_total_points", "bl_ict_index", "bl_next3_fdr")

# Label-side columns. These describe the *outcome* of the target gameweek and
# must never be used as inputs. `gw_minutes` exists solely so the two-stage model
# can learn "did this player feature" as a separate head; it is passed to `fit`
# and never to `predict`.
LABEL_EXTRAS: tuple[str, ...] = ("gw_minutes",)

# Identifier columns carried alongside the features for splitting and reporting.
KEYS: tuple[str, ...] = ("season", "element", "GW", "position", "team_id")

POSITIONS = ("GKP", "DEF", "MID", "FWD")

# Accumulators: (panel column, name of its running total).
_CUMULATIVE = (
    ("total_points", "cum_points"),
    ("minutes", "cum_minutes"),
    ("is_start", "cum_starts"),
    ("goals_scored", "cum_goals"),
    ("assists", "cum_assists"),
    ("clean_sheets", "cum_cs"),
    ("goals_conceded", "cum_gc"),
    ("saves", "cum_saves"),
    ("bps", "cum_bps"),
    ("bonus", "cum_bonus"),
    ("ict_index", "cum_ict"),
    ("influence", "cum_influence"),
    ("creativity", "cum_creativity"),
    ("threat", "cum_threat"),
    ("expected_goals", "cum_xg"),
    ("expected_assists", "cum_xa"),
    ("expected_goals_conceded", "cum_xgc"),
)

FORM_WINDOW = 4  # FPL's own `form` is a 30-day mean, which is ~4 gameweeks.


def _safe_div(numerator, denominator):
    """Element-wise divide where a zero denominator yields 0, not inf or NaN."""
    import numpy as np

    denominator = np.asarray(denominator, dtype="float64")
    numerator = np.asarray(numerator, dtype="float64")
    out = np.zeros_like(numerator, dtype="float64")
    mask = denominator > 0
    np.divide(numerator, denominator, out=out, where=mask)
    return out


def build_frame(panel):
    """Turn the raw panel into ``(features + keys + target)``, leak-free.

    ``panel`` is the output of :func:`ml.panel.build`. The returned frame has one
    row per player-gameweek from :data:`ml.config.MIN_GAMEWEEK` onwards.
    """
    import numpy as np
    import pandas as pd

    from . import config

    df = panel.sort_values(["season", "element", "GW"]).reset_index(drop=True).copy()
    df["_played"] = (df["minutes"].fillna(0) > 0).astype(float)
    df["_row"] = 1.0

    group = df.groupby(["season", "element"], sort=False)

    # ── The shift that makes this honest ────────────────────────────────────
    # Every accumulator is over rows *strictly before* the current one. Remove
    # the shift(1) and every metric downstream becomes meaningless, because the
    # gameweek's own points would be inside its own "points to date".
    def prior_cumsum(source: str):
        return group[source].transform(lambda s: s.shift(1).cumsum())

    for source, name in _CUMULATIVE:
        if source in df.columns:
            df[name] = prior_cumsum(source)
        else:
            df[name] = np.nan

    df["team_games_played"] = prior_cumsum("_row")
    df["appearances"] = prior_cumsum("_played")

    df["form"] = group["total_points"].transform(
        lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean()
    )

    games = df["team_games_played"].fillna(0).to_numpy()
    apps = df["appearances"].fillna(0).to_numpy()
    minutes = df["cum_minutes"].fillna(0).to_numpy()

    out = pd.DataFrame(index=df.index)
    out["form"] = df["form"].fillna(0.0)
    out["points_per_game"] = _safe_div(df["cum_points"].fillna(0), apps)
    out["pts_per_90"] = _safe_div(df["cum_points"].fillna(0) * 90.0, minutes)
    out["minutes_per_team_game"] = _safe_div(minutes, games)
    out["start_rate"] = _safe_div(df["cum_starts"].fillna(0), games)
    out["goals_per_90"] = _safe_div(df["cum_goals"].fillna(0) * 90.0, minutes)
    out["assists_per_90"] = _safe_div(df["cum_assists"].fillna(0) * 90.0, minutes)
    out["cs_per_game"] = _safe_div(df["cum_cs"].fillna(0), games)
    out["gc_per_90"] = _safe_div(df["cum_gc"].fillna(0) * 90.0, minutes)
    out["saves_per_90"] = _safe_div(df["cum_saves"].fillna(0) * 90.0, minutes)
    out["bps_per_90"] = _safe_div(df["cum_bps"].fillna(0) * 90.0, minutes)
    out["bonus_per_game"] = _safe_div(df["cum_bonus"].fillna(0), games)
    out["ict_per_game"] = _safe_div(df["cum_ict"].fillna(0), games)
    out["influence_per_game"] = _safe_div(df["cum_influence"].fillna(0), games)
    out["creativity_per_game"] = _safe_div(df["cum_creativity"].fillna(0), games)
    out["threat_per_game"] = _safe_div(df["cum_threat"].fillna(0), games)

    # Opta rates stay NaN rather than 0 where the season predates them, so the
    # tree learns "unknown" instead of "none".
    for name, accumulator in (
        ("xg_per_90", "cum_xg"),
        ("xa_per_90", "cum_xa"),
        ("xgc_per_90", "cum_xgc"),
    ):
        rate = _safe_div(df[accumulator].fillna(0) * 90.0, minutes)
        out[name] = np.where(df[accumulator].notna(), rate, np.nan)

    # A season whose source file lacked a column leaves it missing rather than
    # present-and-NaN, and `pd.to_numeric(None)` raises. Absent must degrade to
    # NaN, not to a stack trace four seasons into an assembly run.
    def column(name: str):
        if name not in df.columns:
            return pd.Series(np.nan, index=df.index, dtype="float64")
        return pd.to_numeric(df[name], errors="coerce")

    out["price"] = column("value").fillna(0) / 10.0
    out["team_games_played"] = games

    position = df["position"].astype(str) if "position" in df.columns else pd.Series(
        "", index=df.index
    )
    for code in POSITIONS:
        out[f"is_{code.lower()}"] = (position == code).astype(float)

    out["n_fixtures"] = column("n_fixtures").fillna(0)
    out["mean_fdr"] = column("fdr")
    out["home_share"] = column("home_share")
    out["fpl_xp"] = column("xP")
    out["has_xstats"] = column("has_xstats").fillna(0.0)

    # Not features — inputs the FCPS baseline needs to be reconstructed exactly
    # as the app computes it. Prefixed so they can never be swept into a model
    # by a careless `frame.drop(columns=KEYS)`.
    out["bl_total_points"] = df["cum_points"].fillna(0)
    out["bl_ict_index"] = df["cum_ict"].fillna(0)
    out["bl_next3_fdr"] = column("next3_fdr")

    for key in KEYS:
        out[key] = df[key] if key in df.columns else np.nan
    out[TARGET] = pd.to_numeric(df["total_points"], errors="coerce")
    out["gw_minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0)

    out = out[out["GW"] >= config.MIN_GAMEWEEK]
    out = out[out[TARGET].notna()]
    # A player with no prior gameweek at all has no history to build features
    # from, and would contribute a block of identical all-zero rows.
    out = out[out["team_games_played"] > 0]
    columns = (
        list(FEATURES)
        + list(BASELINE_INPUTS)
        + list(KEYS)
        + [TARGET]
        + list(LABEL_EXTRAS)
    )
    return out[columns].reset_index(drop=True)


# ---------------------------------------------------------------- live path


def from_bootstrap(
    element: Mapping,
    team_games: int,
    fixture_context: Mapping,
    use_fpl_xp: bool = True,
) -> dict[str, float]:
    """The same features, from a live ``bootstrap-static`` element.

    ``fixture_context`` describes the *target* gameweek and must carry
    ``n_fixtures``, ``mean_fdr`` and ``home_share`` — all of which are published
    in the fixture list before the deadline.

    Returns a plain dict keyed by :data:`FEATURES`. Missing Opta fields come back
    as ``float('nan')`` so the live vector has the same missingness semantics the
    model was trained under.
    """
    nan = float("nan")

    minutes = _f(element.get("minutes"))
    games = max(1.0, float(team_games))
    total_points = _f(element.get("total_points"))

    # FPL publishes points_per_game as a string; recompute when it is absent so a
    # schema change degrades one feature rather than zeroing it.
    ppg = _f(element.get("points_per_game"))
    if ppg <= 0 and minutes > 0:
        ppg = total_points / max(1.0, minutes / 90.0)

    def per_90(total_key: str, rate_key: str | None = None, required: bool = False):
        if rate_key:
            rate = element.get(rate_key)
            if rate not in (None, ""):
                return _f(rate)
        raw = element.get(total_key)
        if required and raw in (None, ""):
            return nan
        if minutes <= 0:
            return nan if required else 0.0
        return _f(raw) * 90.0 / minutes

    position = _position_of(element)
    features = {
        "form": _f(element.get("form")),
        "points_per_game": ppg,
        "pts_per_90": (total_points * 90.0 / minutes) if minutes > 0 else 0.0,
        "minutes_per_team_game": minutes / games,
        "start_rate": _f(element.get("starts")) / games,
        "goals_per_90": per_90("goals_scored"),
        "assists_per_90": per_90("assists"),
        "cs_per_game": _f(element.get("clean_sheets")) / games,
        "gc_per_90": per_90("goals_conceded"),
        "saves_per_90": per_90("saves", "saves_per_90"),
        "xg_per_90": per_90("expected_goals", "expected_goals_per_90", required=True),
        "xa_per_90": per_90("expected_assists", "expected_assists_per_90", required=True),
        "xgc_per_90": per_90(
            "expected_goals_conceded",
            "expected_goals_conceded_per_90",
            required=True,
        ),
        "bps_per_90": per_90("bps"),
        "bonus_per_game": _f(element.get("bonus")) / games,
        "ict_per_game": _f(element.get("ict_index")) / games,
        "influence_per_game": _f(element.get("influence")) / games,
        "creativity_per_game": _f(element.get("creativity")) / games,
        "threat_per_game": _f(element.get("threat")) / games,
        "price": _f(element.get("now_cost")) / 10.0,
        "team_games_played": float(team_games),
        "is_gkp": 1.0 if position == "GKP" else 0.0,
        "is_def": 1.0 if position == "DEF" else 0.0,
        "is_mid": 1.0 if position == "MID" else 0.0,
        "is_fwd": 1.0 if position == "FWD" else 0.0,
        "n_fixtures": float(fixture_context.get("n_fixtures", 0)),
        "mean_fdr": _opt(fixture_context.get("mean_fdr")),
        "home_share": _opt(fixture_context.get("home_share")),
        "fpl_xp": _f(element.get("ep_next")) if use_fpl_xp else nan,
        "has_xstats": 1.0,
    }
    return {name: features[name] for name in FEATURES}


def vector(features: Mapping[str, float]) -> list[float]:
    """Features as a list in :data:`FEATURES` order. The only ordering allowed."""
    return [float(features.get(name, float("nan"))) for name in FEATURES]


def fixture_context_for(
    team_id: int,
    gameweek: int,
    fixture_index: Mapping[int, Mapping[int, Sequence[Mapping]]],
) -> dict:
    """Target-gameweek fixture context from :func:`engine.xpts.build_fixture_index`.

    A blank returns ``n_fixtures = 0`` with NaN difficulty and NaN home share,
    matching how blanks are encoded in training. Getting this wrong is how a
    model ends up projecting a full return for a player whose team isn't playing.
    """
    fixtures = list(fixture_index.get(int(team_id), {}).get(int(gameweek), ()))
    if not fixtures:
        return {"n_fixtures": 0, "mean_fdr": float("nan"), "home_share": float("nan")}
    return {
        "n_fixtures": len(fixtures),
        "mean_fdr": sum(float(f.get("fdr", 3)) for f in fixtures) / len(fixtures),
        "home_share": sum(1.0 for f in fixtures if f.get("home")) / len(fixtures),
    }


def _f(value, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _opt(value) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _position_of(element: Mapping) -> str:
    return {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}.get(
        int(element.get("element_type", 0)), "UNK"
    )


# ---------------------------------------------------------------- guard rails


def assert_no_lookahead(panel, frame, season: str, element: int, gameweek: int) -> None:
    """Re-derive one row by hand and check the accumulators exclude gameweek *g*.

    A unit test can assert on a fixture; this asserts on the real data, which is
    where an accidental un-shifted column would actually show up. Raises
    ``AssertionError`` on any mismatch.
    """
    prior = panel[
        (panel["season"] == season)
        & (panel["element"] == element)
        & (panel["GW"] < gameweek)
    ]
    row = frame[
        (frame["season"] == season)
        & (frame["element"] == element)
        & (frame["GW"] == gameweek)
    ]
    if row.empty:
        raise AssertionError(f"no feature row for {season}/{element}/GW{gameweek}")
    row = row.iloc[0]

    expected_games = float(len(prior))
    assert row["team_games_played"] == expected_games, (
        f"team_games_played {row['team_games_played']} != {expected_games}"
    )

    minutes = float(prior["minutes"].fillna(0).sum())
    expected_mpg = minutes / expected_games if expected_games else 0.0
    assert abs(row["minutes_per_team_game"] - expected_mpg) < 1e-6, (
        f"minutes_per_team_game {row['minutes_per_team_game']} != {expected_mpg}"
    )

    tail = prior.sort_values("GW").tail(FORM_WINDOW)["total_points"].fillna(0)
    expected_form = float(tail.mean()) if len(tail) else 0.0
    assert abs(row["form"] - expected_form) < 1e-6, (
        f"form {row['form']} != {expected_form}"
    )
