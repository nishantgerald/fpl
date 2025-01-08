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

    # Get filter parameters from the request
    min_points = request.args.get("min_points", default=0, type=float)
    max_points = request.args.get("max_points", default=float("inf"), type=float)
    min_form = request.args.get("min_form", default=0, type=float)
    max_form = request.args.get("max_form", default=float("inf"), type=float)
    min_selected = request.args.get("min_selected", default=0, type=float)
    max_selected = request.args.get("max_selected", default=100, type=float)

    players = data["elements"]
    teams = fetch_teams()

    # Filter players based on the query parameters
    player_list = [
        {
            "id": player["id"],
            "name": f"{player['first_name']} {player['second_name']}",
            "team": teams.get(player["team"], "Unknown"),
            "position": POSITION_MAP[player["element_type"]],
            "price": player["now_cost"] / 10,
            "total_points": player["total_points"],
            "form": player["form"],
            "selected_by_percent": player["selected_by_percent"],
            "status": STATUS_MAP.get(player["status"], "Unknown"),
        }
        for player in players
        if min_points <= player["total_points"] <= max_points
        and min_form <= float(player["form"]) <= max_form
        and min_selected <= float(player["selected_by_percent"]) <= max_selected
    ]

    return jsonify({"data": player_list})  # Add "data" key

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
