#!/usr/bin/env python3

from flask import Flask, render_template, jsonify, request
import requests
import os

app = Flask(__name__)

# Status and Position mappings
STATUS_MAP = {
    "a": "Available",
    "i": "Injured",
    "d": "Doubtful",
    "s": "Suspended",
    "u": "Unavailable",
}
POSITION_MAP = {
    1: "GKP",
    2: "DEF",
    3: "MID",
    4: "FWD",
}


# Fetch data from FPL API
def fetch_player_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url)
    return response.json() if response.status_code == 200 else None


def fetch_teams():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {team["id"]: team["name"] for team in data["teams"]}
    return {}


def fetch_gameweek_picks(user_id, gameweek):
    url = (
        f"https://fantasy.premierleague.com/api/entry/{user_id}/event/{gameweek}/picks/"
    )
    response = requests.get(url)
    return response.json() if response.status_code == 200 else None


def fetch_current_gameweek():
    data = fetch_player_data()
    if not data:
        return None
    return next(event["id"] for event in data["events"] if event["is_current"])


def organize_team(picks, players):
    lineup = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
    bench = []
    for pick in picks:
        element_id = pick["element"]
        player = players.get(element_id, {})
        status = STATUS_MAP.get(player["status"], "Unknown")
        player_data = {
            "id": element_id,
            "name": f"{player.get('first_name', '')} {player.get('second_name', '')}",
            "photo": f"https://resources.premierleague.com/premierleague/photos/players/250x250/p{player['code']}.png",
            "position": POSITION_MAP[player["element_type"]],
            "is_captain": pick["is_captain"],
            "is_vice_captain": pick["is_vice_captain"],
            "status": status,
            "status_class": (
                "doubtful"
                if status == "Doubtful"
                else (
                    "injured"
                    if status in ["Injured", "Suspended", "Unavailable"]
                    else "available"
                )
            ),
        }
        if pick["multiplier"] > 0:
            lineup[player_data["position"]].append(player_data)
        else:
            bench.append(player_data)
    return lineup, bench


@app.route("/player-stats/<int:player_id>")
def player_stats(player_id):
    data = fetch_player_data()
    if not data:
        return jsonify({"error": "Failed to fetch player data"}), 500
    players = {p["id"]: p for p in data["elements"]}
    teams = fetch_teams()  # Fetch the team ID-to-name mapping
    player = players.get(player_id)
    if player:
        return jsonify(
            {
                "name": f"{player['first_name']} {player['second_name']}",
                "photo": f"https://resources.premierleague.com/premierleague/photos/players/250x250/p{player['code']}.png",
                "team": teams.get(
                    player["team"], "Unknown"
                ),  # Translate team ID to name
                "position": POSITION_MAP[player["element_type"]],
                "price": player["now_cost"] / 10,
                "total_points": player["total_points"],
                "form": player["form"],
                "selected_by_percent": player["selected_by_percent"],
                "status": STATUS_MAP.get(player["status"], "Unknown"),
            }
        )
    return jsonify({"error": "Player not found"}), 404

@app.route("/players")
def players_page():
    return render_template("player_search.html")

@app.route("/api/players")
def players_data():
    data = fetch_player_data()
    if not data:
        return jsonify({"error": "Failed to fetch player data"}), 500

    fixtures_response = requests.get("https://fantasy.premierleague.com/api/fixtures/")
    if fixtures_response.status_code != 200:
        return jsonify({"error": "Failed to fetch fixture data"}), 500
    fixtures = fixtures_response.json()

    current_gameweek = fetch_current_gameweek()
    if not current_gameweek:
        return jsonify({"error": "Failed to fetch current gameweek"}), 500

    # Fetch team abbreviations
    team_abbreviations = {team["id"]: team["short_name"] for team in data["teams"]}

    players = data["elements"]
    teams = {team["id"]: team["name"] for team in data["teams"]}

    # Group fixtures by team
    team_fixtures = {}
    for fixture in fixtures:
        if fixture["event"] is None or fixture["event"] < current_gameweek:
            continue
        team_h = fixture["team_h"]
        team_a = fixture["team_a"]
        if team_h not in team_fixtures:
            team_fixtures[team_h] = []
        if team_a not in team_fixtures:
            team_fixtures[team_a] = []
        team_fixtures[team_h].append({"difficulty": fixture["team_h_difficulty"], "gameweek": fixture["event"]})
        team_fixtures[team_a].append({"difficulty": fixture["team_a_difficulty"], "gameweek": fixture["event"]})

    # Filter players and add Next 3 FDR
    player_list = []
    for player in players:
        team_id = player["team"]
        # Filter fixtures that are strictly after the current gameweek
        future_fixtures = [
            fixture for fixture in team_fixtures.get(team_id, [])
            if fixture["gameweek"] > current_gameweek
        ]
        # Sort the fixtures by gameweek and take the next 3
        next_fixtures = sorted(future_fixtures, key=lambda x: x["gameweek"])[:3]
        
        # Calculate FDR for the next 3 fixtures
        next_3_fdr = sum(fixture["difficulty"] for fixture in next_fixtures)

        player_list.append({
            "id": player["id"],
            "name": f"{player['first_name']} {player['second_name']}",
            "team": team_abbreviations.get(team_id, "UNK"),
            "position": POSITION_MAP[player["element_type"]],
            "price": player["now_cost"] / 10,
            "total_points": player["total_points"],
            "form": player["form"],
            "selected_by_percent": player["selected_by_percent"],
            "status": STATUS_MAP.get(player["status"], "Unknown"),
            "next_3_fdr": next_3_fdr,
            "current_gw": current_gameweek,
        })

    return jsonify({"data": player_list})

@app.route("/api/fixtures")
def fixtures_data():
    url = "https://fantasy.premierleague.com/api/fixtures/"
    response = requests.get(url)
    if response.status_code != 200:
        return jsonify({"error": "Failed to fetch fixture data"}), 500

    fixtures = response.json()

    # Fetch team abbreviations from players data
    data = fetch_player_data()
    if not data:
        return jsonify({"error": "Failed to fetch player data for team abbreviations"}), 500

    team_abbreviations = {team["id"]: team["short_name"] for team in data["teams"]}

    # Extract required data
    fixture_list = [
        {
            "gameweek": fixture["event"],
            "home_team": team_abbreviations.get(fixture["team_h"], "UNK"),
            "away_team": team_abbreviations.get(fixture["team_a"], "UNK"),
            "team_h_difficulty": fixture["team_h_difficulty"],
            "team_a_difficulty": fixture["team_a_difficulty"],
        }
        for fixture in fixtures
    ]

    return jsonify(fixture_list)

@app.route("/")
def home():
    user_id = request.args.get("user_id", default=3022850, type=int)  # Default USER_ID
    gameweek = fetch_current_gameweek()
    if not gameweek:
        return "Failed to fetch current gameweek."
    data = fetch_player_data()
    if not data:
        return "Failed to fetch player data."
    players = {p["id"]: p for p in data["elements"]}
    picks = fetch_gameweek_picks(user_id, gameweek)
    if not picks:
        return "Failed to fetch gameweek picks."
    lineup, bench = organize_team(picks["picks"], players)
    return render_template(
        "formation_with_bench.html",
        starting_lineup=lineup,
        bench=bench,
        user_id=user_id,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))  # Default to 5001 if $PORT is not set
    app.run(host="0.0.0.0", port=port)
