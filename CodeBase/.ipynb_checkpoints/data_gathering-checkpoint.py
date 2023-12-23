import requests
import json
import os

BASE_URL = "https://statsapi.web.nhl.com/api/v1/game/{}/feed/live/"
OUTPUT_DIR = "nhl_data"
#OUTPUT_DIR = "nhl_data_playoffs"

# Ensure the output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def generate_game_ids(season):
    ids = []

    # Determine the number of games based on teams in the season
    if season == 2016:
        total_games = 1230  # 30 teams
    elif 2016 < season <= 2020:
        total_games = 1271  # 31 teams
    else:
        total_games = 1353  # 32 teams
    
    # Regular season game IDs
    for game_num in range(1, total_games + 1):
        ids.append(f"{season}02{game_num:04d}")
    
    '''
    # Playoffs IDs
    for round_num in range(1, 5):  # There are 4 rounds in playoffs
        for matchup_num in range(1, 9):  # 8 matchups per round
            for game_num in range(1, 8):  # Max 7 games per matchup
                ids.append(f"{season}030{round_num}{matchup_num}{game_num}")
    '''

    return ids

def download_data(game_id):
    response = requests.get(BASE_URL.format(game_id))
    if response.status_code == 200:
        with open(f"{OUTPUT_DIR}/{game_id}.json", 'w') as f:
            json.dump(response.json(), f)

if __name__ == "__main__":
    # Iterate over the desired seasons
    for season in range(2016, 2021):
        print(f"Downloading data for {season}-{season+1} season...")
        for game_id in generate_game_ids(season):
            download_data(game_id)
        print(f"Finished downloading data for {season}-{season+1} season!")

print("All done!")