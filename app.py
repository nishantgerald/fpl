#!/usr/bin/env python3

import os
import requests
import pandas as pd
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI

app = Flask(__name__)
CORS(app)  # Allow Flutter web app to call this API
pd.set_option("display.max_rows", None)

# ------------------- Globals & Mappings -------------------
cached_data = None  # Global variable to store FPL data

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
    5: "MGR",
}

# ------------------- Fetch & Cache Data -------------------
def fetch_player_data():
    """
    Fetch overall bootstrap data from FPL API.
    """
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url)
    return response.json() if response.status_code == 200 else None

def get_cached_data():
    """
    Use a global cache so we don't repeatedly fetch the entire data set
    on every request. If cached_data is None, fetch from the API.
    """
    global cached_data
    if cached_data is None:
        print("Fetching data from FPL API...")
        cached_data = fetch_player_data()
    return cached_data

def fetch_fixtures_data():
    """
    Fetch all fixtures from the FPL API.
    Returns fixture list if success, else None.
    """
    url = "https://fantasy.premierleague.com/api/fixtures/"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()

def fetch_current_gameweek():
    """
    Identify the current gameweek from the FPL 'events' data.
    """
    data = fetch_player_data()
    if not data:
        return None
    return next(event["id"] for event in data["events"] if event["is_current"])

def build_team_fixtures(fixtures, current_gameweek):
    """
    Build a dictionary of upcoming fixtures keyed by team ID,
    each containing a list of its fixture difficulties and gameweeks.
    """
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
        team_fixtures[team_h].append({
            "difficulty": fixture["team_h_difficulty"],
            "gameweek": fixture["event"]
        })
        team_fixtures[team_a].append({
            "difficulty": fixture["team_a_difficulty"],
            "gameweek": fixture["event"]
        })
    return team_fixtures

def get_trade_recommendations(user_players_df, top_players_df):
    """
    Generate trade recommendations for the user's fantasy team
    using the GPT-4o-mini model.

    Args:
        user_players_df (pd.DataFrame): DataFrame containing the user's current team.
        top_players_df (pd.DataFrame): DataFrame containing the top players for the gameweek.

    Returns:
        str: Response from GPT-4o-mini with trade recommendations.
    """
    # Load environment variables
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in the .env file.")

    # Initialize the OpenAI client
    client = OpenAI(api_key=openai_api_key)

    # Prepare the prompt
    prompt = (
        f"""
        You are an expert fantasy premier league analyst. You will give the user the best transfer recommendations based on their current team, 
        and the top players for the upcoming weeks. Take into account all their stats including FCPS (fantasy composite player score = a composite 
        score that takes into account the difficulty rating for the next 3 games, the form, total score, and ICT index) and other stats. The user's 
        team is {user_players_df.to_dict(orient='records')} and the top players are {top_players_df.to_dict(orient='records')}. Give the user the best 
        trade recommendations for this gameweek.
        
        When considering trades, consider:
        - Max number of players from the same team can only be 3
        - Never suggest trade IN if the player is already in the user's current team
        - Never suggest trade OUT if the player is not in the user's current team
        - the trade out and trade in values to ensure that the trade is actually feasible in cost.
        
        Your response should be structured as follows (with sample data):
            # Fantasy Premier League Transfer Recommendations
            1. Goalkeeper
            * Out: Matz Sels (GKP, NFO)
            Price: 5.0
            Reason: While he has decent form and FCPS, there are other GKP options with lower prices and potential better returns.
            * In: Dean Henderson (GKP, CRY)

                Price: 4.5
                Total Points: 79
                Form: 4.8
                Next 3 FDR: 7
                FCPS: 426.0
                ICT Index: 51.6
            ---
            2. Defender
            * Out: ...
            Price: 6.4
            Reason: ...
            * In: Trent Alexander-Arnold (DEF, LIV)
            ...
            ...
            ...
            ---
            3. Midfielder
            * Out: ...
            Price: 5.1
            Reason: ...
            * In: Anthony Gordon (MID, NEW)
            ...
            ...
            ...
            ---
            4. Forward
            * Out: ...
            Price: 9.5
            Reason: ...
            * In: ...
            ...
            ...
            ...
            ---
            ### Summary of Recommendations
            Out Player	In Player	Position	Price Change	Notes
            Matz Sels (5.0)	Dean Henderson (4.5)	GKP	-0.5	Solid alternative with potential value.
            ...
            ...
            ...

            ## Conclusion
            Consider the suggested transfers to strengthen your lineup, which will not only save cost but also improve potential points from upcoming fixtures. Make sure to monitor injuries and form, as this can influence your strategy leading up to gameweek deadlines.

        Format your response in markdown. Use tables, headers, and bullet points.
        """
    )

    # Make the API call
    try:
        completion = client.chat.completions.create(
            # model="gpt-4o-mini",
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        # Extract and return the response
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating trade recommendations: {str(e)}"

# ------------------- Normalization & Scoring -------------------
def get_normalization_values():
    """
    Fetch global maximum values for normalization across all players
    (total_points, form, next_3_fdr, ict_index).
    """
    data = get_cached_data()
    if not data:
        return None

    players = data["elements"]
    return {
        "total_points": max(p["total_points"] for p in players),
        "form": max(float(p["form"]) for p in players),
        "next_3_fdr": 15,  # Hardcoded maximum for next 3 fixtures FDR
        "ict_index": max(float(p["ict_index"]) for p in players),
    }

def calculate_fcps(dataframe, weights=None, max_values=None):
    """
    Calculate FCPS (Fantasy Composite Player Score) for a given DataFrame.
    If max_values is provided, it uses them for normalization; otherwise,
    it calculates normalization values dynamically from the DataFrame.
    """
    if weights is None:
        weights = {
            "total_points_weight": 0.2,
            "form_weight": 0.4,
            "fdr_weight": 0.25,
            "ict_index_weight": 0.15,
        }

    numeric_columns = ["total_points", "form", "next_3_fdr", "ict_index"]
    for col in numeric_columns:
        dataframe[col] = pd.to_numeric(dataframe[col], errors="coerce")  # Convert to numeric

    # Fill missing data with 0
    if dataframe[numeric_columns].isnull().any().any():
        dataframe.fillna(0, inplace=True)

    # Use provided max values or calculate from dataframe
    if max_values:
        dataframe["total_points_norm"] = dataframe["total_points"] / max_values["total_points"]
        dataframe["form_norm"] = dataframe["form"] / max_values["form"]
        dataframe["fdr_norm"] = dataframe["next_3_fdr"] / max_values["next_3_fdr"]
        dataframe["ict_index_norm"] = dataframe["ict_index"] / max_values["ict_index"]
    else:
        dataframe["total_points_norm"] = dataframe["total_points"] / dataframe["total_points"].max()
        dataframe["form_norm"] = dataframe["form"] / dataframe["form"].max()
        dataframe["fdr_norm"] = dataframe["next_3_fdr"] / dataframe["next_3_fdr"].max()
        dataframe["ict_index_norm"] = dataframe["ict_index"] / dataframe["ict_index"].max()

    # Invert FDR because lower FDR is better (so we do 1 - normalized FDR)
    dataframe["fcps"] = (
        weights["total_points_weight"] * dataframe["total_points_norm"]
        + weights["form_weight"] * dataframe["form_norm"]
        + weights["fdr_weight"] * (1 - dataframe["fdr_norm"])
        + weights["ict_index_weight"] * dataframe["ict_index_norm"]
    )

    # Scale & round
    dataframe["fcps"] = (dataframe["fcps"].round(3) * 1000)

    # Drop intermediate columns
    dataframe.drop(
        ["total_points_norm", "form_norm", "fdr_norm", "ict_index_norm"],
        axis=1,
        inplace=True
    )
    return dataframe

def calculate_next_3_fdr(team_id, team_fixtures, current_gameweek):
    """
    Sum fixture difficulties for the next 3 fixtures of a team.
    """
    future_fixtures = [
        f for f in team_fixtures.get(team_id, []) if f["gameweek"] >= current_gameweek
    ]
    next_fixtures = sorted(future_fixtures, key=lambda x: x["gameweek"])[:3]
    return sum(fx["difficulty"] for fx in next_fixtures)

# ------------------- Picks & Organization -------------------
def fetch_gameweek_picks(user_id, gameweek):
    """
    Given an FPL user id and a gameweek, fetch the user's picks.
    """
    url = f"https://fantasy.premierleague.com/api/entry/{user_id}/event/{gameweek}/picks/"
    response = requests.get(url)
    return response.json() if response.status_code == 200 else None

def organize_team(picks, players):
    """
    Return a dictionary of { position : [list of player dicts] } for the starting lineup,
    and a separate bench list.
    """
    lineup = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
    bench = []

    for pick in picks:
        element_id = pick["element"]
        player = players.get(element_id, {})
        status = STATUS_MAP.get(player["status"], "Unknown")

        player_data = {
            "id": element_id,
            "name": f"{player.get('first_name', '')} {player.get('second_name', '')}",
            "photo": (
                f"{request.host_url}api/photo/{player['code']}"
            ),
            "position": POSITION_MAP[player["element_type"]],
            "is_captain": pick["is_captain"],
            "is_vice_captain": pick["is_vice_captain"],
            "status": status,
            "status_class": (
                "doubtful" if status == "Doubtful"
                else (
                    "injured" if status in ["Injured", "Suspended", "Unavailable"]
                    else "available"
                )
            ),
        }

        # If multiplier > 0, it's part of the starting lineup
        if pick["multiplier"] > 0:
            lineup[player_data["position"]].append(player_data)
        else:
            bench.append(player_data)

    return lineup, bench

# ------------------- DataFrame Builders -------------------
def get_player_dataframe(player_ids, starting_eleven_ids):
    """
    Build a DataFrame of player data for the given player_ids, including FCPS,
    and sort in descending order of FCPS. Distinguish who is in starting eleven.
    """
    data = get_cached_data()
    if not data:
        print("Failed to fetch player data.")
        return pd.DataFrame()

    # Fetch fixtures & current gameweek
    fixtures = fetch_fixtures_data()
    if fixtures is None:
        print("Failed to fetch fixture data.")
        return pd.DataFrame()

    current_gameweek = fetch_current_gameweek()
    if not current_gameweek:
        print("Failed to fetch current gameweek.")
        return pd.DataFrame()

    # Prepare data
    players = {p["id"]: p for p in data["elements"]}
    team_abbreviations = {team["id"]: team["short_name"] for team in data["teams"]}
    team_fixtures = build_team_fixtures(fixtures, current_gameweek)
    normalization_values = get_normalization_values()

    # Filter for the requested players
    filtered_players = []
    for pid in player_ids:
        player = players.get(pid)
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
                "starting_eleven": pid in starting_eleven_ids,
                "ict_index": player["ict_index"],
            })

    # Convert to DataFrame
    if not filtered_players:
        print("No player data fetched for the given IDs.")
        return pd.DataFrame()

    df = pd.DataFrame(filtered_players)
    df = calculate_fcps(df, max_values=normalization_values)
    return df

def print_top_players():
    """
    Calculate FCPS for all available players, sort them by FCPS descending,
    and select top GKP, DEF, MID, FWD. Return combined DataFrame.
    """
    data = get_cached_data()
    if not data:
        print("Failed to fetch player data.")
        return

    fixtures = fetch_fixtures_data()
    if fixtures is None:
        print("Failed to fetch fixture data.")
        return

    current_gameweek = fetch_current_gameweek()
    if not current_gameweek:
        print("Failed to fetch current gameweek.")
        return

    players = {p["id"]: p for p in data["elements"]}
    team_abbreviations = {team["id"]: team["short_name"] for team in data["teams"]}
    team_fixtures = build_team_fixtures(fixtures, current_gameweek)
    normalization_values = get_normalization_values()

    player_list = []
    for p in players.values():
        # Only consider 'Available' players in top players logic
        if p["status"] != "a":
            continue

        team_id = p["team"]
        next_3_fdr = calculate_next_3_fdr(team_id, team_fixtures, current_gameweek)

        player_list.append({
            "id": p["id"],
            "name": f"{p['first_name']} {p['second_name']}",
            "team": team_abbreviations.get(team_id, "UNK"),
            "position": POSITION_MAP[p["element_type"]],
            "price": p["now_cost"] / 10,
            "total_points": p["total_points"],
            "form": float(p["form"]),
            "next_3_fdr": next_3_fdr,
            "ict_index": p["ict_index"],
        })

    df = pd.DataFrame(player_list)
    df["ict_index"] = pd.to_numeric(df["ict_index"], errors="coerce")
    df = calculate_fcps(df, max_values=normalization_values)
    df = df.sort_values(by="fcps", ascending=False)

    # Filter top players by position
    top_gkps = df[df["position"] == "GKP"].head(5)
    top_defs = df[df["position"] == "DEF"].head(15)
    top_mids = df[df["position"] == "MID"].head(25)
    top_fwds = df[df["position"] == "FWD"].head(25)

    return pd.concat([top_gkps, top_defs, top_mids, top_fwds])

# ------------------- Flask Routes -------------------
@app.route("/")
def home():
    """
    Redirect root to the Flutter web app.
    """
    from flask import redirect
    return redirect("/app/", code=302)


@app.route("/legacy")
def home_legacy():
    """
    Old Flask HTML interface (kept for reference).
    """
    user_id = request.args.get("user_id", default=3022850, type=int)
    gameweek = fetch_current_gameweek()
    if not gameweek:
        return "Failed to fetch current gameweek."

    data = get_cached_data()
    if not data:
        return "Failed to fetch player data."

    players = {p["id"]: p for p in data["elements"]}
    picks = fetch_gameweek_picks(user_id, gameweek)
    if not picks:
        return "Failed to fetch gameweek picks."

    # Organize
    lineup, bench = organize_team(picks["picks"], players)

    # Extract player ids from lineup/bench
    player_ids = [
        p["id"] for position in lineup.values() for p in position
    ] + [p["id"] for p in bench]
    starting_eleven_ids = [
        p["id"] for position in lineup.values() for p in position
    ]

    # Build DataFrame
    user_players_df = get_player_dataframe(player_ids, starting_eleven_ids)
    top_players_df = print_top_players()

    # Render
    return render_template(
        "formation_with_bench.html",
        starting_lineup=lineup,
        bench=bench,
        user_id=user_id,
        user_players_df=user_players_df,
        top_players_df=top_players_df
    )

@app.route("/players")
def players_page():
    """
    Renders a page for searching players.
    """
    return render_template("player_search.html")

@app.route("/api/players", methods=["GET"])
def players_data():
    """
    API endpoint to return players (with FCPS).
    Query params:
      - player_id: get one specific player
      - min_fcps, max_fcps: filter by FCPS range
    """
    data = fetch_player_data()
    if not data:
        return jsonify({"error": "Failed to fetch player data"}), 500

    fixtures = fetch_fixtures_data()
    if fixtures is None:
        return jsonify({"error": "Failed to fetch fixture data"}), 500

    current_gameweek = fetch_current_gameweek()
    if not current_gameweek:
        return jsonify({"error": "Failed to fetch current gameweek"}), 500

    team_abbreviations = {team["id"]: team["short_name"] for team in data["teams"]}
    players = {p["id"]: p for p in data["elements"]}
    team_fixtures = build_team_fixtures(fixtures, current_gameweek)
    normalization_values = get_normalization_values()

    player_id = request.args.get("player_id", type=int)
    min_fcps = request.args.get("min_fcps", default=None, type=float)
    max_fcps = request.args.get("max_fcps", default=None, type=float)

    # If a single player_id is provided:
    if player_id:
        player = players.get(player_id)
        if player:
            team_id = player["team"]
            next_3_fdr = calculate_next_3_fdr(team_id, team_fixtures, current_gameweek)
            single_data = [{
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
                "ict_index": player["ict_index"],
                "photo": f"{request.host_url}api/photo/{player['code']}"
            }]
            df = pd.DataFrame(single_data)
            df = calculate_fcps(df, max_values=normalization_values)
            return jsonify({"data": df.to_dict(orient="records")})
        else:
            return jsonify({"error": "Player not found"}), 404

    # Otherwise, return data for all players
    player_list = []
    for pl in players.values():
        team_id = pl["team"]
        next_3_fdr = calculate_next_3_fdr(team_id, team_fixtures, current_gameweek)

        player_list.append({
            "id": pl["id"],
            "name": f"{pl['first_name']} {pl['second_name']}",
            "team": team_abbreviations.get(team_id, "UNK"),
            "position": POSITION_MAP[pl["element_type"]],
            "price": pl["now_cost"] / 10,
            "total_points": pl["total_points"],
            "form": pl["form"],
            "selected_by_percent": pl["selected_by_percent"],
            "status": STATUS_MAP.get(pl["status"], "Unknown"),
            "next_3_fdr": next_3_fdr,
            "ict_index": pl["ict_index"],
            "photo": f"{request.host_url}api/photo/{pl['code']}"
        })

    df = pd.DataFrame(player_list)
    numeric_columns = ["total_points", "form", "next_3_fdr", "ict_index"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = calculate_fcps(df, max_values=normalization_values)

    # FCPS Filters
    if min_fcps is not None:
        df = df[df["fcps"] >= min_fcps]
    if max_fcps is not None:
        df = df[df["fcps"] <= max_fcps]

    return jsonify({"data": df.to_dict(orient="records")})

@app.route("/api/fixtures")
def fixtures_data():
    """
    API endpoint to return fixture data.
    """
    fixtures = fetch_fixtures_data()
    if fixtures is None:
        return jsonify({"error": "Failed to fetch fixture data"}), 500

    data = fetch_player_data()
    if not data:
        return jsonify({"error": "Failed to fetch player data for team abbreviations"}), 500

    team_abbreviations = {team["id"]: team["short_name"] for team in data["teams"]}

    fixture_list = []
    for fixture in fixtures:
        fixture_list.append({
            "gameweek": fixture["event"],
            "home_team": team_abbreviations.get(fixture["team_h"], "UNK"),
            "away_team": team_abbreviations.get(fixture["team_a"], "UNK"),
            "team_h_difficulty": fixture["team_h_difficulty"],
            "team_a_difficulty": fixture["team_a_difficulty"],
        })

    return jsonify(fixture_list)

_photo_cache = {}  # in-memory cache: code -> bytes

@app.route("/api/photo/<int:code>")
def player_photo(code):
    """
    Proxy player photos from the PL CDN through our domain.
    This bypasses Safari ITP and other cross-origin restrictions on the client side.
    Images are cached in memory after the first fetch.
    """
    from flask import Response
    if code in _photo_cache:
        return Response(_photo_cache[code], mimetype="image/png",
                        headers={"Cache-Control": "public, max-age=86400"})
    try:
        url = f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{code}.png"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            _photo_cache[code] = r.content
            return Response(r.content, mimetype="image/png",
                            headers={"Cache-Control": "public, max-age=86400"})
    except Exception:
        pass
    return "", 404


@app.route("/api/entry/<int:entry_id>")
def get_entry(entry_id):
    """
    Proxy the FPL entry endpoint so Flutter can look up a manager's
    name/team without hitting FPL directly (avoids CORS in web builds).
    """
    try:
        url = f"https://fantasy.premierleague.com/api/entry/{entry_id}/"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 404:
            return jsonify({"error": "Entry not found"}), 404
        if resp.status_code != 200:
            return jsonify({"error": f"FPL API returned {resp.status_code}"}), 502
        data = resp.json()
        return jsonify({
            "id": data["id"],
            "manager_name": f"{data['player_first_name']} {data['player_last_name']}",
            "team_name": data.get("name", ""),
            "region": data.get("player_region_name", ""),
            "overall_points": data.get("summary_overall_points", 0),
            "overall_rank": data.get("summary_overall_rank", 0),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/team")
def team_data():
    """
    JSON API for the Flutter app: returns the user's current team
    (formation + bench) with full player stats and FCPS scores.
    """
    user_id = request.args.get("user_id", default=3022850, type=int)
    gameweek = fetch_current_gameweek()
    if not gameweek:
        return jsonify({"error": "Failed to fetch current gameweek"}), 500

    data = get_cached_data()
    if not data:
        return jsonify({"error": "Failed to fetch player data"}), 500

    players = {p["id"]: p for p in data["elements"]}
    picks = fetch_gameweek_picks(user_id, gameweek)
    if not picks:
        return jsonify({"error": "Failed to fetch gameweek picks. Check the user ID."}), 404

    # Organise into lineup/bench
    lineup, bench = organize_team(picks["picks"], players)

    # Get all player IDs for FCPS calculation
    starting_eleven_ids = [p["id"] for pos in lineup.values() for p in pos]
    bench_ids = [p["id"] for p in bench]
    all_ids = starting_eleven_ids + bench_ids

    # Build DataFrame with FCPS
    df = get_player_dataframe(all_ids, starting_eleven_ids)
    fcps_map = {}
    next_3_fdr_map = {}
    if not df.empty:
        fcps_map = dict(zip(df["id"], df["fcps"]))
        next_3_fdr_map = dict(zip(df["id"], df["next_3_fdr"]))

    def enrich(player_data):
        pid = player_data["id"]
        raw = players.get(pid, {})
        return {
            **player_data,
            "price": raw.get("now_cost", 0) / 10,
            "total_points": raw.get("total_points", 0),
            "form": float(raw.get("form", 0)),
            "selected_by_percent": float(raw.get("selected_by_percent", 0)),
            "ict_index": float(raw.get("ict_index", 0)),
            "next_3_fdr": int(next_3_fdr_map.get(pid, 0)),
            "team": next(
                (t["short_name"] for t in data["teams"] if t["id"] == raw.get("team")),
                "UNK",
            ),
            "fcps": round(fcps_map.get(pid, 0), 1),
        }

    enriched_lineup = {
        pos: [enrich(p) for p in players_list]
        for pos, players_list in lineup.items()
    }
    enriched_bench = [enrich(p) for p in bench]

    return jsonify({
        "gameweek": gameweek,
        "user_id": user_id,
        "lineup": enriched_lineup,
        "bench": enriched_bench,
    })


@app.route("/trade_recommendations", methods=["GET", "POST"])
def trade_recommendations():
    """
    Route to display trade recommendations for the user's fantasy team.
    """
    if request.method == "POST":
        try:
            # Fetch cached data and top players
            data = get_cached_data()
            if not data:
                return jsonify({"error": "Failed to fetch player data"}), 500

            top_players_df = print_top_players()
            user_id = request.form.get("user_id", 3022850)  # Default user ID if not provided
            gameweek = fetch_current_gameweek()
            if not gameweek:
                return jsonify({"error": "Failed to fetch current gameweek"}), 500

            picks = fetch_gameweek_picks(user_id, gameweek)
            if not picks:
                return jsonify({"error": "Failed to fetch gameweek picks"}), 500

            # Extract player data for the user's team
            player_ids = [pick["element"] for pick in picks["picks"]]
            starting_eleven_ids = [pick["element"] for pick in picks["picks"] if pick["multiplier"] > 0]
            user_players_df = get_player_dataframe(player_ids, starting_eleven_ids)

            # Call the recommendation function
            recommendation = get_trade_recommendations(user_players_df, top_players_df)

            # Return the recommendation as JSON
            return jsonify({"recommendation": recommendation})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Render the HTML template for GET request
    return render_template("trade_recommendations.html")


# ------------------- Flutter App Serving -------------------
import os as _os
from flask import send_from_directory as _send_from_directory

FLUTTER_BUILD_DIR = _os.path.join(_os.path.dirname(__file__), "flutter_web")

@app.route("/app")
@app.route("/app/")
def flutter_index():
    """Serve the Flutter web app index page."""
    return _send_from_directory(FLUTTER_BUILD_DIR, "index.html")

@app.route("/app/<path:filename>")
def flutter_static(filename):
    """Serve Flutter web static assets."""
    return _send_from_directory(FLUTTER_BUILD_DIR, filename)


# ------------------- Main Runner -------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))  # Default to 5001 if not set
    app.run(host="0.0.0.0", port=port)