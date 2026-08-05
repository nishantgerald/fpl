"""Synthetic FPL fixtures.

Everything the engine decides is a pure function of plain dicts, so the whole
test suite runs offline against payloads shaped like the real API's.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def _never_touch_real_llm_state(monkeypatch, tmp_path):
    """Keep every test out of the operator's real LLM budget and cache.

    Both live under ``~/.cache/fpl`` by default. Any test that reaches
    ``fcps_llm.call_model`` or ``narrative`` reserves against the global daily
    ceiling, so without this the suite silently spends the day's allowance and
    the counter climbs with no model calls behind it. Applied here rather than
    per-file so a new test file inherits the isolation instead of having to
    remember it.

    Imported inside the fixture: importing engine modules at conftest import
    time would run before ``sys.path`` is set up for some invocations.
    """
    from engine import fcps_llm, llm_budget, narrative

    monkeypatch.setattr(llm_budget, "STATE_DIR", tmp_path / "llm-budget")
    monkeypatch.setattr(fcps_llm, "CACHE_DIR", tmp_path / "fcps-cache")
    fcps_llm.clear_cache()
    narrative.clear_cache()


TEAM_IDS = list(range(1, 21))


def make_team(team_id):
    return {
        "id": team_id,
        "name": f"Team {team_id}",
        "short_name": f"T{team_id:02d}",
        "played": 12,
        "strength_attack_home": 1100 + team_id * 5,
        "strength_attack_away": 1050 + team_id * 5,
        "strength_defence_home": 1100 + team_id * 5,
        "strength_defence_away": 1050 + team_id * 5,
    }


def make_element(
    element_id,
    element_type,
    team,
    now_cost=50,
    *,
    status="a",
    starts=12,
    minutes=1080,
    xg90=0.1,
    xa90=0.1,
    xgc90=1.2,
    total_points=60,
    cost_change_start=0,
    chance=None,
    transfers_in=0,
    transfers_out=0,
    ownership="10.0",
    ep_next="4.0",
):
    return {
        "id": element_id,
        "code": 100000 + element_id,
        "first_name": f"First{element_id}",
        "second_name": f"Last{element_id}",
        "web_name": f"P{element_id}",
        "element_type": element_type,
        "team": team,
        "now_cost": now_cost,
        "cost_change_start": cost_change_start,
        "cost_change_event": 0,
        "status": status,
        "chance_of_playing_next_round": chance,
        "starts": starts,
        "minutes": minutes,
        "total_points": total_points,
        "form": "4.0",
        "ep_next": ep_next,
        "ict_index": "50.0",
        "selected_by_percent": ownership,
        "expected_goals_per_90": str(xg90),
        "expected_assists_per_90": str(xa90),
        "expected_goals_conceded_per_90": str(xgc90),
        "saves_per_90": "3.0" if element_type == 1 else "0.0",
        "defensive_contribution_per_90": "6.0",
        "bonus": 8,
        "yellow_cards": 2,
        "red_cards": 0,
        "goals_scored": 4,
        "assists": 3,
        "goals_conceded": 14,
        "transfers_in_event": transfers_in,
        "transfers_out_event": transfers_out,
        "news": "",
    }


@pytest.fixture
def teams():
    return [make_team(t) for t in TEAM_IDS]


@pytest.fixture
def element_types():
    return [
        {"id": 1, "singular_name_short": "GKP", "squad_select": 2},
        {"id": 2, "singular_name_short": "DEF", "squad_select": 5},
        {"id": 3, "singular_name_short": "MID", "squad_select": 5},
        {"id": 4, "singular_name_short": "FWD", "squad_select": 3},
    ]


@pytest.fixture
def events():
    return [
        {
            "id": gw,
            "name": f"Gameweek {gw}",
            "deadline_time": "2026-01-01T11:00:00Z",
            "finished": gw < 13,
            "is_current": gw == 13,
            "is_next": gw == 14,
            "most_captained": 41,
        }
        for gw in range(1, 39)
    ]


@pytest.fixture
def fixtures():
    """Every team plays every gameweek 13-20, paired 1v2, 3v4, ..."""
    out = []
    fixture_id = 1
    for gw in range(13, 21):
        for i in range(0, 20, 2):
            home, away = TEAM_IDS[i], TEAM_IDS[i + 1]
            if gw % 2 == 0:
                home, away = away, home
            out.append(
                {
                    "id": fixture_id,
                    "event": gw,
                    "team_h": home,
                    "team_a": away,
                    "team_h_difficulty": 2 + (home % 3),
                    "team_a_difficulty": 2 + (away % 3),
                    "finished": False,
                }
            )
            fixture_id += 1
    return out


@pytest.fixture
def elements():
    """A player universe: 2 GKP, 5 DEF, 5 MID, 3 FWD per team, ids 1..300."""
    out = []
    element_id = 1
    for team in TEAM_IDS:
        for element_type, count, cost in ((1, 2, 45), (2, 5, 50), (3, 5, 65), (4, 3, 70)):
            for n in range(count):
                out.append(
                    make_element(
                        element_id,
                        element_type,
                        team,
                        now_cost=cost + n * 5,
                        # Spread quality so the optimiser has real choices.
                        xg90=0.05 + (element_id % 7) * 0.06,
                        xa90=0.05 + (element_id % 5) * 0.04,
                        total_points=30 + (element_id % 11) * 8,
                    )
                )
                element_id += 1
    return out


@pytest.fixture
def squad(elements):
    """A legal 15: 2/5/5/3, at most 3 from any club."""
    by_type = {1: [], 2: [], 3: [], 4: []}
    club_count = {}
    for element in elements:
        et = element["element_type"]
        team = element["team"]
        needed = {1: 2, 2: 5, 3: 5, 4: 3}[et]
        if len(by_type[et]) >= needed:
            continue
        if club_count.get(team, 0) >= 3:
            continue
        by_type[et].append(element)
        club_count[team] = club_count.get(team, 0) + 1
    return by_type[1] + by_type[2] + by_type[3] + by_type[4]


@pytest.fixture
def bootstrap(elements, teams, element_types, events):
    return {
        "elements": elements,
        "teams": teams,
        "element_types": element_types,
        "events": events,
        "total_players": 10_000_000,
        "game_settings": {"squad_team_limit": 3},
    }
