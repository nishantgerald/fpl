#!/usr/bin/env python3

from flask import Flask, render_template, jsonify, request
import requests
import pandas as pd
import os

pd.set_option("display.max_rows", None)

cached_data = None  # Global variable to store FPL data
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

def get_cached_data():
    global cached_data
    if cached_data is None:
        print("Fetching data from FPL API...")
        cached_data = fetch_player_data()
    return cached_data

def calculate_fcps(dataframe, weights=None):
    """
    Calculate FCPS (Fantasy Composite Player Score) for a given DataFrame.
    """
    if weights is None:
        weights = {
            "total_points_weight": 0.2,
            "form_weight": 0.25,
            "fdr_weight": 0.25,
            "ict_index_weight": 0.3,
        }

    # Ensure numeric columns
    numeric_columns = ["total_points", "form", "next_3_fdr", "ict_index"]
    for col in numeric_columns:
        dataframe[col] = pd.to_numeric(dataframe[col], errors="coerce")  # Convert to numeric

    # Check for missing or invalid data
    if dataframe[numeric_columns].isnull().any().any():
        print("Warning: Some columns contain invalid or missing data. These will be filled with 0.")
        dataframe.fillna(0, inplace=True)  # Handle NaN values (optional: handle differently)

    # Normalize columns
    dataframe["total_points_norm"] = dataframe["total_points"] / dataframe["total_points"].max()
    dataframe["form_norm"] = dataframe["form"] / dataframe["form"].max()
    dataframe["fdr_norm"] = dataframe["next_3_fdr"] / dataframe["next_3_fdr"].max()
    dataframe["ict_index_norm"] = dataframe["ict_index"] / dataframe["ict_index"].max()

    # Calculate FCPS
    dataframe["fcps"] = (
        weights["total_points_weight"] * dataframe["total_points_norm"] +
        weights["form_weight"] * dataframe["form_norm"] +
        weights["fdr_weight"] * (1 - dataframe["fdr_norm"]) +  # Invert FDR
        weights["ict_index_weight"] * dataframe["ict_index_norm"]
    )

    # Drop intermediate normalization columns for cleaner output
    dataframe.drop(
        ["total_points_norm", "form_norm", "fdr_norm", "ict_index_norm"], 
        axis=1, 
        inplace=True
    )

    return dataframe


def calculate_next_3_fdr(team_id, team_fixtures, current_gameweek):
    """
    Calculates the sum of fixture difficulties for the next 3 fixtures of a team.
    """
    future_fixtures = [
        fixture for fixture in team_fixtures.get(team_id, [])
        if fixture["gameweek"] > current_gameweek
    ]
    next_fixtures = sorted(future_fixtures, key=lambda x: x["gameweek"])[:3]
    return sum(fixture["difficulty"] for fixture in next_fixtures)

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

def get_player_dataframe(player_ids, starting_eleven_ids):
    """
    Fetches player data for a list of player IDs using cached data
    and returns the data as a Pandas DataFrame, including FCPS,
    sorted in descending order of FCPS.
    """
    # Use cached data
    data = get_cached_data()
    if not data:
        print("Failed to fetch player data.")
        return pd.DataFrame()

    # Fetch fixtures data
    fixtures_response = requests.get("https://fantasy.premierleague.com/api/fixtures/")
    if fixtures_response.status_code != 200:
        print("Failed to fetch fixture data.")
        return pd.DataFrame()
    fixtures = fixtures_response.json()

    # Determine current gameweek
    current_gameweek = fetch_current_gameweek()
    if not current_gameweek:
        print("Failed to fetch current gameweek.")
        return pd.DataFrame()

    # Prepare mappings and necessary data
    players = {p["id"]: p for p in data["elements"]}
    team_abbreviations = {team["id"]: team["short_name"] for team in data["teams"]}
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

    # Filter the data for the given player IDs
    filtered_players = []
    for player_id in player_ids:
        player = players.get(player_id)
        if player:
            team_id = player["team"]
            next_3_fdr = calculate_next_3_fdr(team_id, team_fixtures, current_gameweek)

            filtered_players.append({
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
                "starting_eleven": player_id in starting_eleven_ids,
                "ict_index": player["ict_index"],
            })

    # Convert the filtered list into a DataFrame
    if filtered_players:
        df = pd.DataFrame(filtered_players)
        # Calculate FCPS using FCPS calc function
        df = calculate_fcps(df)
        # print("-----")
        # print("User's Team")
        # print(df)
        return df
    else:
        print("No player data fetched for the given IDs.")
        return pd.DataFrame()

def print_top_players():
    """
    Calculates FCPS (Fantasy Composity Player Score) for all players, sorts by position and FCPS, 
    and selects the top players for each position to form the top players DataFrame.
    """
    # Use cached data
    data = get_cached_data()
    if not data:
        print("Failed to fetch player data.")
        return

    # Fetch fixtures data
    fixtures_response = requests.get("https://fantasy.premierleague.com/api/fixtures/")
    if fixtures_response.status_code != 200:
        print("Failed to fetch fixture data.")
        return
    fixtures = fixtures_response.json()

    # Determine current gameweek
    current_gameweek = fetch_current_gameweek()
    if not current_gameweek:
        print("Failed to fetch current gameweek.")
        return

    # Prepare data mappings
    players = {p["id"]: p for p in data["elements"]}
    team_abbreviations = {team["id"]: team["short_name"] for team in data["teams"]}
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

    # Calculate FCPS for all players
    player_list = []
    for player in players.values():
        if player["status"] != "a":  # Only consider available players
            continue

        team_id = player["team"]
        next_3_fdr = calculate_next_3_fdr(team_id, team_fixtures, current_gameweek)

        player_list.append({
            "id": player["id"],
            "name": f"{player['first_name']} {player['second_name']}",
            "team": team_abbreviations.get(team_id, "UNK"),
            "position": POSITION_MAP[player["element_type"]],
            "price": player["now_cost"] / 10,
            "total_points": player["total_points"],
            "form": float(player["form"]),
            "next_3_fdr": next_3_fdr,
            "ict_index": player["ict_index"],
        })

    # Convert the full list into a DataFrame
    df = pd.DataFrame(player_list)

    # Ensure numeric conversion
    df["ict_index"] = pd.to_numeric(df["ict_index"], errors="coerce")

    # Calculate FCPS using FCPS calc function
    df = calculate_fcps(df)

    # Sort by FCPS
    df = df.sort_values(by="fcps", ascending=False)

    # Filter top players for each position
    top_gkps = df[df["position"] == "GKP"].head(5)
    top_defs = df[df["position"] == "DEF"].head(15)
    top_mids = df[df["position"] == "MID"].head(25)
    top_fwds = df[df["position"] == "FWD"].head(25)

    # Combine all top players into a single DataFrame
    top_players_df = pd.concat([top_gkps, top_defs, top_mids, top_fwds])

    # print("-----")
    # print("Top Players")
    # print(top_players_df)

    return top_players_df

@app.route("/players")
def players_page():
    return render_template("player_search.html")

@app.route("/api/players", methods=["GET"])
def players_data():
    data = fetch_player_data()
    if not data:
        return jsonify({"error": "Failed to fetch player data"}), 500

    # Fetch fixtures data
    fixtures_response = requests.get("https://fantasy.premierleague.com/api/fixtures/")
    if fixtures_response.status_code != 200:
        return jsonify({"error": "Failed to fetch fixture data"}), 500
    fixtures = fixtures_response.json()

    # Determine current gameweek
    current_gameweek = fetch_current_gameweek()
    if not current_gameweek:
        return jsonify({"error": "Failed to fetch current gameweek"}), 500

    # Prepare data mappings
    team_abbreviations = {team["id"]: team["short_name"] for team in data["teams"]}
    players = {p["id"]: p for p in data["elements"]}
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

    # Check if player_id is provided
    player_id = request.args.get("player_id", type=int)
    if player_id:
        # Return data for a single player
        player = players.get(player_id)
        if player:
            team_id = player["team"]
            next_3_fdr = calculate_next_3_fdr(team_id, team_fixtures, current_gameweek)

            return jsonify({
                "name": f"{player['first_name']} {player['second_name']}",
                "photo": f"https://resources.premierleague.com/premierleague/photos/players/250x250/p{player['code']}.png",
                "team": team_abbreviations.get(team_id, "UNK"),
                "position": POSITION_MAP[player["element_type"]],
                "price": player["now_cost"] / 10,
                "total_points": player["total_points"],
                "form": player["form"],
                "selected_by_percent": player["selected_by_percent"],
                "status": STATUS_MAP.get(player["status"], "Unknown"),
                "next_3_fdr": next_3_fdr,
            })
        return jsonify({"error": "Player not found"}), 404

    # Return data for all players
    player_list = []
    for player in players.values():
        team_id = player["team"]
        next_3_fdr = calculate_next_3_fdr(team_id, team_fixtures, current_gameweek)

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

    # Use cached data
    data = get_cached_data()
    if not data:
        return "Failed to fetch player data."
    
    players = {p["id"]: p for p in data["elements"]}
    picks = fetch_gameweek_picks(user_id, gameweek)
    if not picks:
        return "Failed to fetch gameweek picks."

    # Organize starting lineup and bench
    lineup, bench = organize_team(picks["picks"], players)

    # Extract player IDs from lineup and bench
    player_ids = [p["id"] for position in lineup.values() for p in position] + [p["id"] for p in bench]
    starting_eleven_ids = [p["id"] for position in lineup.values() for p in position]

    # Fetch player data as DataFrame
    user_players_df=get_player_dataframe(player_ids, starting_eleven_ids)

    # Print top players DataFrame
    top_players_df=print_top_players()

    # Render the template
    return render_template(
        "formation_with_bench.html",
        starting_lineup=lineup,
        bench=bench,
        user_id=user_id,
        user_players_df=user_players_df,
        top_players_df=top_players_df
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))  # Default to 5001 if $PORT is not set
    app.run(host="0.0.0.0", port=port)
