"""Assemble the cached CSVs into one tidy panel: a row per player-gameweek.

The output is the only thing the rest of the package reads, so every schema
difference between seasons is absorbed here rather than leaking into feature
code. The differences that matter:

============================  =====================================================
``position``                  Absent from ``merged_gw`` before 2020-21. Joined in
                              from ``players_raw.csv`` via ``element_type``.
``starts``                    Absent before 2022-23. Proxied by ``minutes >= 60``
                              *for every season*, including those where the real
                              column exists, so the feature means one thing
                              throughout. The real column is kept alongside for
                              auditing.
``expected_*``                Absent before 2022-23 (Opta data). Left as NaN with
                              an explicit ``has_xstats`` flag; gradient boosting
                              handles the missingness natively, and the flag lets
                              the model learn that those rows are a different
                              regime rather than one where every player has zero
                              xG.
``team``                      Named inconsistently across seasons. Always taken
                              as a team *id* from ``players_raw.csv``.
============================  =====================================================

Double gameweeks are the reason the panel is grouped rather than used raw:
``merged_gw`` has one row per *fixture*, so a player with two fixtures in
gameweek 29 appears twice. Points are summed to the gameweek, and the fixture
context collapses to ``n_fixtures`` / ``mean_fdr`` / ``home_share``. A blank
gameweek produces no row at all in the source, and is reinstated here with
``n_fixtures = 0`` — otherwise the model never sees a blank during training and
happily projects six points for a player who isn't playing.
"""

from __future__ import annotations

from typing import Iterable

from . import config, sources

# Outcome columns summed across the fixtures of a gameweek.
SUM_COLUMNS = (
    "total_points",
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_missed",
    "penalties_saved",
    "red_cards",
    "yellow_cards",
    "saves",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "xP",
)

# Columns taken from the first fixture of the gameweek (they don't add up).
FIRST_COLUMNS = ("value", "selected", "transfers_balance")

POSITION_BY_TYPE = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD", 5: "MGR"}
POSITION_ALIASES = {"GK": "GKP", "GKP": "GKP", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}

STARTER_MINUTES = 60


def build_season(season: str):
    """One season's panel. Returns a DataFrame or ``None`` if the data is absent."""
    import numpy as np
    import pandas as pd

    gw = sources.read_csv(sources.local_path(season, "gws/merged_gw.csv"))
    if gw is None or gw.empty:
        return None

    gw = gw.copy()
    gw.columns = [str(c).strip() for c in gw.columns]

    # `GW` and `round` name the same thing in different seasons.
    if "GW" not in gw.columns and "round" in gw.columns:
        gw["GW"] = gw["round"]
    gw["GW"] = pd.to_numeric(gw["GW"], errors="coerce")
    gw = gw.dropna(subset=["GW", "element"])
    gw["GW"] = gw["GW"].astype(int)
    gw["element"] = pd.to_numeric(gw["element"], errors="coerce").astype("Int64")

    for column in SUM_COLUMNS + FIRST_COLUMNS:
        if column in gw.columns:
            gw[column] = pd.to_numeric(gw[column], errors="coerce")
        else:
            gw[column] = np.nan

    gw["was_home"] = gw.get("was_home", False).astype(bool)
    gw["is_start"] = (gw["minutes"].fillna(0) >= STARTER_MINUTES).astype(float)
    if "starts" in gw.columns:
        gw["starts_reported"] = pd.to_numeric(gw["starts"], errors="coerce")
    else:
        gw["starts_reported"] = np.nan

    players = _players_lookup(season)
    # `merged_gw.csv` carries its own `position`/`team` columns in some seasons
    # (2020-21 onward), despite this module's own docstring claiming otherwise.
    # Drop them so the join below is the single source of truth and doesn't
    # silently rename both sides to `position_x`/`position_y`.
    gw = gw.drop(columns=[c for c in players.columns if c != "element" and c in gw.columns])
    gw = gw.merge(players, on="element", how="left")

    gw = _attach_fixture_difficulty(gw, season)

    grouped = _collapse_to_gameweek(gw)
    grouped["season"] = season
    grouped["has_xstats"] = grouped["expected_goals"].notna().astype(float)

    grouped = _reinstate_blanks(grouped)

    schedule = team_fdr_table(season)
    if schedule is not None:
        grouped = grouped.merge(schedule, on=["team_id", "GW"], how="left")
    else:
        grouped["next3_fdr"] = np.nan
    return grouped


def team_fdr_table(season: str):
    """``(team_id, GW) -> next3_fdr``: the FDR sum of a team's next 3 fixtures.

    This is FCPS's fixture term, and it is reconstructed here so the FCPS
    baseline in :mod:`ml.baselines` is the real FCPS rather than an approximation
    of it. Known before the gameweek's deadline, so it is safe to attach to the
    target row.
    """
    import numpy as np
    import pandas as pd

    fixtures = sources.read_csv(sources.local_path(season, "fixtures.csv"))
    if fixtures is None or fixtures.empty:
        return None

    fixtures = fixtures.copy()
    event = pd.to_numeric(fixtures.get("event"), errors="coerce")
    rows = []
    for side, team_column, difficulty_column in (
        ("h", "team_h", "team_h_difficulty"),
        ("a", "team_a", "team_a_difficulty"),
    ):
        if team_column not in fixtures.columns:
            return None
        rows.append(
            pd.DataFrame(
                {
                    "team_id": pd.to_numeric(fixtures[team_column], errors="coerce"),
                    "GW": event,
                    "difficulty": pd.to_numeric(
                        fixtures.get(difficulty_column, np.nan), errors="coerce"
                    ),
                }
            )
        )
    schedule = pd.concat(rows, ignore_index=True).dropna(subset=["team_id", "GW"])
    schedule["GW"] = schedule["GW"].astype(int)

    out = []
    for team_id, team_rows in schedule.groupby("team_id"):
        ordered = team_rows.sort_values("GW")
        gws = ordered["GW"].to_numpy()
        difficulties = ordered["difficulty"].fillna(3).to_numpy()
        for gw in range(1, 39):
            upcoming = difficulties[gws >= gw][:3]
            out.append(
                {
                    "team_id": team_id,
                    "GW": gw,
                    "next3_fdr": float(upcoming.sum()) if len(upcoming) else np.nan,
                    "next3_fixtures_counted": int(len(upcoming)),
                }
            )
    return pd.DataFrame(out)


def _players_lookup(season: str):
    """``element -> (position, team_id, web_name)``, from ``players_raw.csv``."""
    import pandas as pd

    raw = sources.read_csv(sources.local_path(season, "players_raw.csv"))
    if raw is None or raw.empty:
        return pd.DataFrame(
            {"element": pd.Series(dtype="Int64"), "position": [], "team_id": []}
        )

    raw = raw.copy()
    id_column = "id" if "id" in raw.columns else "element"
    out = pd.DataFrame(
        {
            "element": pd.to_numeric(raw[id_column], errors="coerce").astype("Int64"),
            "position": raw.get("element_type", pd.Series(index=raw.index)).map(
                POSITION_BY_TYPE
            ),
            "team_id": pd.to_numeric(
                raw.get("team", pd.Series(index=raw.index)), errors="coerce"
            ),
        }
    )
    return out.dropna(subset=["element"]).drop_duplicates("element")


def _attach_fixture_difficulty(gw, season: str):
    """Join FPL's own difficulty rating for the fixture the player featured in."""
    import numpy as np
    import pandas as pd

    fixtures = sources.read_csv(sources.local_path(season, "fixtures.csv"))
    if fixtures is None or fixtures.empty or "fixture" not in gw.columns:
        gw["fdr"] = np.nan
        return gw

    fixtures = fixtures.copy()
    key = "id" if "id" in fixtures.columns else "fixture"
    lookup = pd.DataFrame(
        {
            "fixture": pd.to_numeric(fixtures[key], errors="coerce"),
            "fdr_home": pd.to_numeric(
                fixtures.get("team_h_difficulty", np.nan), errors="coerce"
            ),
            "fdr_away": pd.to_numeric(
                fixtures.get("team_a_difficulty", np.nan), errors="coerce"
            ),
        }
    ).dropna(subset=["fixture"])

    gw["fixture"] = pd.to_numeric(gw["fixture"], errors="coerce")
    gw = gw.merge(lookup, on="fixture", how="left")
    gw["fdr"] = np.where(gw["was_home"], gw["fdr_home"], gw["fdr_away"])
    return gw.drop(columns=["fdr_home", "fdr_away"], errors="ignore")


def _collapse_to_gameweek(gw):
    """One row per (element, GW). Doubles are summed; context is aggregated."""
    import pandas as pd

    aggregations = {column: "sum" for column in SUM_COLUMNS}
    aggregations.update({column: "first" for column in FIRST_COLUMNS})
    aggregations.update(
        {
            "position": "first",
            "team_id": "first",
            "is_start": "max",
            "starts_reported": "sum",
            "fdr": "mean",
            "was_home": "mean",
            "opponent_team": "first",
            "fixture": "count",
        }
    )
    available = {k: v for k, v in aggregations.items() if k in gw.columns}

    grouped = gw.groupby(["element", "GW"], as_index=False).agg(available)
    grouped = grouped.rename(columns={"fixture": "n_fixtures", "was_home": "home_share"})
    grouped["position"] = (
        grouped["position"].map(lambda p: POSITION_ALIASES.get(str(p), None))
    )

    # `sum` over an all-NaN group yields 0 in pandas, which would turn "we have
    # no xG data for this season" into "this player recorded zero xG". Restore
    # the distinction — it is the difference between a missing regime and a bad
    # player, and gradient boosting can only exploit the former if it can see it.
    import numpy as np

    optional = (
        "expected_goals",
        "expected_assists",
        "expected_goals_conceded",
        "expected_goal_involvements",
        "xP",
    )
    present = [c for c in optional if c in gw.columns and c in grouped.columns]
    if present:
        observed = (
            gw.groupby(["element", "GW"], as_index=False)[present]
            .agg(lambda s: bool(s.notna().any()))
            .rename(columns={c: f"_seen_{c}" for c in present})
        )
        grouped = grouped.merge(observed, on=["element", "GW"], how="left")
        for column in present:
            grouped[column] = np.where(
                grouped[f"_seen_{column}"].fillna(False), grouped[column], np.nan
            )
        grouped = grouped.drop(columns=[f"_seen_{c}" for c in present])

    return grouped


def _reinstate_blanks(panel):
    """Add explicit zero-fixture rows for gameweeks a player's team didn't play.

    Built from the player's own observed gameweek span, so a player who joined in
    January isn't credited with fifteen blanks he wasn't in the league for.
    """
    import numpy as np
    import pandas as pd

    if panel.empty:
        return panel

    frames = [panel]
    blanks = []
    for element, rows in panel.groupby("element"):
        played = set(int(g) for g in rows["GW"])
        if not played:
            continue
        low, high = min(played), max(played)
        missing = [g for g in range(low, high + 1) if g not in played]
        if not missing:
            continue
        template = rows.iloc[0]
        for g in missing:
            blanks.append(
                {
                    "element": element,
                    "GW": g,
                    "position": template.get("position"),
                    "team_id": template.get("team_id"),
                    "season": template.get("season"),
                    "n_fixtures": 0,
                    "total_points": 0.0,
                    "minutes": 0.0,
                    "is_start": 0.0,
                    "home_share": np.nan,
                    "fdr": np.nan,
                    "value": template.get("value"),
                    "has_xstats": template.get("has_xstats", 0.0),
                }
            )
    if blanks:
        frames.append(pd.DataFrame(blanks))
    out = pd.concat(frames, ignore_index=True, sort=False)
    return out.sort_values(["element", "GW"]).reset_index(drop=True)


def build(seasons: Iterable[str] = config.SEASONS):
    """The full multi-season panel, sorted by (season, element, gameweek)."""
    import pandas as pd

    frames = []
    for season in seasons:
        season_panel = build_season(season)
        if season_panel is None:
            print(f"[panel] {season}: no data cached, skipping")
            continue
        print(f"[panel] {season}: {len(season_panel):,} player-gameweeks")
        frames.append(season_panel)

    if not frames:
        raise RuntimeError(
            "No seasons could be assembled. Run `python -m ml.sources` first."
        )
    panel = pd.concat(frames, ignore_index=True, sort=False)
    return panel.sort_values(["season", "element", "GW"]).reset_index(drop=True)


def cache_path():
    return config.DATA_DIR / "panel.parquet"


def build_cached(seasons: Iterable[str] = config.SEASONS, refresh: bool = False):
    """Build the panel once and reuse it. Assembly is the slow part, not training."""
    import pandas as pd

    path = cache_path()
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    panel = build(seasons)
    config.ensure_dirs()
    try:
        panel.to_parquet(path, index=False)
    except Exception:  # pyarrow not installed — the cache is an optimisation
        pass
    return panel


if __name__ == "__main__":
    df = build_cached(refresh=True)
    print(df.groupby("season").size())
