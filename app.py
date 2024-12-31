#!/usr/bin/env python3

from flask import Flask, render_template, jsonify, request
import requests

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
    app.run(debug=True)
