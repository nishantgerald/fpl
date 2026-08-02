"""Download and cache the historical data. Network lives here and nowhere else.

Source: `vaastav/Fantasy-Premier-League
<https://github.com/vaastav/Fantasy-Premier-League>`_, a per-gameweek archive of
the official FPL API going back to 2016-17. It is the standard corpus for this
problem, which matters: a result on it is comparable to published work rather
than to a private scrape.

Three files per season are used:

``gws/merged_gw.csv``
    One row per (player, gameweek) with that gameweek's *outcome* — points,
    minutes, goals, bps, and the price the player carried at the time. This is
    both the label source and, lagged, the feature source.
``fixtures.csv``
    Fixture list with FPL's own difficulty ratings and kickoff times. Needed for
    the target gameweek's context and for detecting blanks and doubles.
``teams.csv``
    Per-team strength ratings. Absent before 2020-21; callers must cope.

Everything is cached on disk after the first fetch. Re-running training does not
re-download, and the cache is what makes the backtest reproducible: a rerun three
months later sees the same bytes, not a mirror that has since been rewritten.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

from . import config

FILES = ("gws/merged_gw.csv", "fixtures.csv", "teams.csv", "players_raw.csv")

# Files that genuinely don't exist for some seasons. Their absence is data, not
# an error, and the feature builder degrades rather than failing.
OPTIONAL = ("teams.csv", "fixtures.csv", "players_raw.csv")


class DownloadError(RuntimeError):
    pass


def local_path(season: str, name: str) -> Path:
    return config.season_dir(season) / name.replace("/", "_")


def fetch(season: str, name: str, refresh: bool = False) -> Path | None:
    """Fetch one file for one season into the cache. Returns its local path."""
    target = local_path(season, name)
    if target.exists() and not refresh:
        return target

    import requests

    url = f"{config.VAASTAV_RAW}/{season}/{name}"
    response = requests.get(url, timeout=60)
    if response.status_code == 404:
        if name in OPTIONAL:
            return None
        raise DownloadError(f"{url} -> 404")
    if response.status_code != 200:
        raise DownloadError(f"{url} -> {response.status_code}")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(response.content)
    return target


def fetch_season(season: str, refresh: bool = False) -> dict[str, Path | None]:
    return {name: fetch(season, name, refresh) for name in FILES}


def fetch_all(
    seasons: tuple[str, ...] = config.SEASONS, refresh: bool = False
) -> dict[str, dict[str, Path | None]]:
    config.ensure_dirs()
    out = {}
    for season in seasons:
        print(f"[data] {season} ...", flush=True)
        out[season] = fetch_season(season, refresh)
    return out


def read_csv(path: Path | None):
    """Read a cached CSV, tolerating the encoding drift across seasons.

    Player names in the older files are latin-1 in places; pandas' default utf-8
    read raises on them, which is how you lose four seasons of data to one
    accented surname.
    """
    if path is None or not path.exists():
        return None
    import pandas as pd

    raw = path.read_bytes()
    for encoding in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise DownloadError(f"Could not decode {path}")


# ---------------------------------------------------------------- live snapshot


def snapshot_live(out_dir: Path | None = None) -> Path:
    """Save today's bootstrap + fixtures payloads.

    Not used by training. It exists so that when a prediction looks wrong in
    production you can reproduce the exact inputs that produced it, which is
    otherwise impossible against an API that mutates hourly.
    """
    import json

    import requests

    out_dir = out_dir or (config.DATA_DIR / "live")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, url in (
        ("bootstrap-static", "https://fantasy.premierleague.com/api/bootstrap-static/"),
        ("fixtures", "https://fantasy.premierleague.com/api/fixtures/"),
    ):
        response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        (out_dir / f"{name}.json").write_text(json.dumps(response.json()))
    return out_dir


if __name__ == "__main__":
    seasons = tuple(sys.argv[1:]) or config.SEASONS
    fetch_all(seasons)
    print(f"[data] cached under {config.DATA_DIR}")
