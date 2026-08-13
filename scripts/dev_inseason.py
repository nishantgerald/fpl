"""Run the app as though the season were under way.

Three of this app's screens — Transfers, Actions, My Team — only have anything
to say once FPL has published picks for a gameweek. Before the first deadline
that is never, so the whole transfer path is unreachable in production for the
weeks when you would most like to look at it. That is not a bug to fix; it is
a fact about the upstream data, and it makes the code impossible to *see*.

This serves the real app, the real optimiser and the real bootstrap, with two
things faked at the narrowest seams that exist:

* ``fpl_client.season_state`` reports the upcoming gameweek as current.
* ``fpl_client.picks`` returns a synthetic but legal fifteen for any entry,
  built from the live player list, so the optimiser has a squad to reason about.

Everything downstream — projections, prices, legality, the optimiser itself —
is the shipping code path operating on live data.

    python scripts/dev_inseason.py            # http://127.0.0.1:5055
    open http://127.0.0.1:5055/dev/seed       # links a team, then goes to /app

Never imported by the app. Nothing here runs in production.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import redirect  # noqa: E402

import app as app_module  # noqa: E402
from engine import fpl_client, rules  # noqa: E402

PORT = int(os.environ.get("DEV_PORT", "5055"))

# Any id works — the picks below ignore it — but a real-looking one keeps the
# UI honest about what it is displaying.
DEV_ENTRY_ID = int(os.environ.get("DEV_ENTRY_ID", "1437667"))

# Roughly a mid-table budget, so the optimiser has something to trade with
# rather than a squad already at the ceiling.
BANK_TENTHS = 15


def _fake_season_state() -> dict:
    """The upcoming gameweek, reported as current."""
    data = fpl_client.bootstrap() or {}
    events = data.get("events") or []
    upcoming = next((e for e in events if e.get("is_next")), None) or (
        events[0] if events else {"id": 1, "name": "Gameweek 1"}
    )
    return {
        "started": True,
        "gameweek": int(upcoming.get("id", 1)),
        "gameweek_name": upcoming.get("name", "Gameweek 1"),
        "deadline": upcoming.get("deadline_time"),
    }


def _legal_fifteen() -> list[dict]:
    """A squad that satisfies the real rules, picked cheaply on purpose.

    Deliberately *not* the optimal fifteen: a squad that is already the best
    available leaves the transfer engine with nothing to suggest, which is the
    one state that would tell us nothing about the screen.
    """
    data = fpl_client.bootstrap() or {}
    elements = [e for e in data.get("elements", []) if e.get("status") == "a"]

    quota = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    by_position: dict[str, list[dict]] = {k: [] for k in quota}
    for e in elements:
        pos = rules.position_of(e)
        if pos in by_position:
            by_position[pos].append(e)

    squad: list[dict] = []
    per_club: dict[int, int] = {}
    for pos, need in quota.items():
        # Cheapest first, then by points, so the fifteen is affordable and dull
        # — which is what leaves room for the optimiser to have an opinion.
        pool = sorted(
            by_position[pos],
            key=lambda e: (e.get("now_cost", 0), -int(e.get("total_points", 0))),
        )
        taken = 0
        for e in pool:
            club = int(e.get("team", 0))
            if per_club.get(club, 0) >= 3:  # the real three-per-club rule
                continue
            squad.append(e)
            per_club[club] = per_club.get(club, 0) + 1
            taken += 1
            if taken == need:
                break

    return squad


def _fake_picks(entry_id: int, gameweek: int) -> dict:
    squad = _legal_fifteen()
    picks = []
    for i, element in enumerate(squad):
        picks.append(
            {
                "element": int(element["id"]),
                "position": i + 1,
                "multiplier": 1 if i < 11 else 0,
                "is_captain": i == 0,
                "is_vice_captain": i == 1,
            }
        )
    return {
        "picks": picks,
        "entry_history": {
            "event": gameweek,
            "bank": BANK_TENTHS,
            "value": sum(int(e.get("now_cost", 0)) for e in squad),
            "event_transfers": 0,
            "event_transfers_cost": 0,
        },
        "active_chip": None,
    }


def _fake_entry(entry_id: int) -> dict:
    return {
        "id": entry_id,
        "player_first_name": "Dev",
        "player_last_name": "Harness",
        "name": "In-Season Test",
        "summary_overall_rank": 250_000,
        "summary_overall_points": 0,
        "current_event": _fake_season_state()["gameweek"],
    }


fpl_client.season_state = _fake_season_state
fpl_client.picks = _fake_picks
fpl_client.entry = _fake_entry
# The optimiser reads the squad through the service, which reads it through
# these; `my_team` is the authenticated variant and would otherwise 401.
fpl_client.my_team = lambda entry_id, cookie=None: None
fpl_client.transfers = lambda entry_id: []

# A release bundle minifies Dart type names, so a runtime type error reads as
# "'minified:dm' is not a subtype of 'minified:iy'" and names nothing. Point
# this at a profile build to get the real names back.
#     flutter build web --profile -o /tmp/webprofile
#     DEV_BUNDLE=/tmp/webprofile python scripts/dev_inseason.py
_bundle = os.environ.get("DEV_BUNDLE")
if _bundle:
    app_module.FLUTTER_BUILD_DIR = _bundle

flask_app = app_module.app


@flask_app.route("/dev/seed")
def dev_seed():
    """Link a team in the browser, the way the real app stores it.

    `shared_preferences` on web is localStorage with a `flutter.` prefix and
    JSON-encoded values, so this writes exactly what the app would have written
    had someone typed the id into the form.
    """
    return f"""<!doctype html><meta charset="utf-8">
<title>Seeding a dev team…</title>
<body style="font:15px system-ui;padding:24px">
<p>Linking entry {DEV_ENTRY_ID} and opening the app…</p>
<script>
  localStorage.setItem('flutter.fpl_entry_id', JSON.stringify({DEV_ENTRY_ID}));
  localStorage.setItem('flutter.fpl_manager_name', JSON.stringify("Dev Harness"));
  localStorage.setItem('flutter.fpl_team_name', JSON.stringify("In-Season Test"));
  location.replace('/app/#/transfers');
</script>
</body>"""


@flask_app.route("/dev/state")
def dev_state():
    return {
        "season_state": _fake_season_state(),
        "entry_id": DEV_ENTRY_ID,
        "squad_size": len(_legal_fifteen()),
    }


if __name__ == "__main__":
    state = _fake_season_state()
    print(f"Dev in-season server on http://127.0.0.1:{PORT}")
    print(f"  pretending {state['gameweek_name']} is current")
    print(f"  seed a team: http://127.0.0.1:{PORT}/dev/seed")
    flask_app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False)
