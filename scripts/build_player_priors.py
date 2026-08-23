"""Distil last season's per-90 rates into a small artifact the app can ship.

Why this exists
---------------
FPL resets every total at the season rollover, so from the first kick until a
player has ten full matches behind them the only per-90 rates on record are
computed from cameos. :func:`engine.xpts.shrink_rate` regresses those toward a
prior, and the prior it had was a positional median — which tells the £15.5m
striker and the £4.5m bench filler they will score alike, and early in a season
that *is* most of the projection.

The right prior for a player who has played 2736 Premier League minutes is the
rate he actually produced over them. This turns last season's totals into that
prior, keyed by FPL's ``code``, which is stable across seasons where ``id`` is
not.

Not every player has one. Promoted clubs and overseas signings genuinely have no
Premier League history, and they fall back to the price-band prior — a player
without a past should be judged by what the market thinks of him, not handed
somebody else's numbers.

Usage
-----
    python -m scripts.build_player_priors [season]

Reads ``ml/_data/<season>/players_raw.csv``, which the training pipeline already
downloads, and writes ``engine/data/player_priors.json``. The source data is
gitignored and 32 MB; the artifact is a few hundred kilobytes and is committed,
because the app needs it at runtime and Heroku does not run the trainer.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "engine" / "data" / "player_priors.json"

#: Below this, last season says too little to be anyone's prior. Roughly five
#: full matches — enough that a rate is a rate rather than an afternoon.
MIN_MINUTES = 450

#: Rates worth carrying, as (output key, totals column).
RATES = (
    ("expected_goals_per_90", "expected_goals"),
    ("expected_assists_per_90", "expected_assists"),
    ("expected_goals_conceded_per_90", "expected_goals_conceded"),
    ("saves_per_90", "saves"),
)


def _f(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build(season: str) -> dict:
    source = ROOT / "ml" / "_data" / season / "players_raw.csv"
    if not source.exists():
        raise SystemExit(
            f"{source} not found — run the training data download first "
            "(python -m ml.sources)."
        )

    players: dict[str, dict] = {}
    for row in csv.DictReader(source.open()):
        minutes = _f(row.get("minutes"))
        if minutes < MIN_MINUTES:
            continue
        code = row.get("code")
        if not code:
            continue

        entry = {"minutes": round(minutes)}
        for key, column in RATES:
            total = _f(row.get(column))
            if total > 0:
                entry[key] = round(total / minutes * 90.0, 4)
        # A player with no rate worth carrying is not worth an entry.
        if len(entry) > 1:
            players[str(int(code))] = entry

    return {"season": season, "min_minutes": MIN_MINUTES, "players": players}


def main() -> None:
    season = sys.argv[1] if len(sys.argv) > 1 else "2024-25"
    payload = build(season)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    print(
        f"{season}: {len(payload['players'])} players "
        f"-> {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB)"
    )


if __name__ == "__main__":
    main()
