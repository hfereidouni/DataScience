import requests
import json
import os

BASE_URL = "https://statsapi.web.nhl.com/api/v1/game/{}/feed/live/"
OUTPUT_DIR = "nhl_data"
#OUTPUT_DIR_PLAYOFFS = "nhl_data_playoffs"

# Ensure the output directories exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def generate_game_ids(season, playoffs=False):
    ids = []

    # Determine the number of games based on teams in the season
    if season == 2016:
        total_games = 1230  # 30 teams
    elif 2016 < season <= 2020:
        total_games = 1271  # 31 teams
    else:
        total_games = 1353  # 32 teams

    # Regular season game IDs
    if not playoffs:
        for game_num in range(1, total_games + 1):
            ids.append(f"{season}02{game_num:04d}")
    else:
        # Playoffs IDs
        for round_num in range(1, 5):  # There are 4 rounds in playoffs
            for matchup_num in range(1, 9):  # 8 matchups per round
                for game_num in range(1, 8):  # Max 7 games per matchup
                    ids.append(f"{season}030{round_num}{matchup_num}{game_num}")

    return ids

def download_data(game_id, playoffs=False):
    response = requests.get(BASE_URL.format(game_id))
    if response.status_code == 200:
        if not playoffs:
            output_dir = OUTPUT_DIR
        else:
            output_dir = OUTPUT_DIR
        with open(f"{output_dir}/{game_id}.json", 'w') as f:
            json.dump(response.json(), f)

if __name__ == "__main__":
    # Iterate over the desired seasons
    for season in range(2016, 2021):
        print(f"Downloading regular season data for {season}-{season+1} season...")
        for game_id in generate_game_ids(season, playoffs=False):
            download_data(game_id, playoffs=False)
        print(f"Finished downloading regular season data for {season}-{season+1} season!")

        print(f"Downloading playoff data for {season}-{season+1} season...")
        for game_id in generate_game_ids(season, playoffs=True):
            download_data(game_id, playoffs=True)
        print(f"Finished downloading playoff data for {season}-{season+1} season!")

print("All done!")
