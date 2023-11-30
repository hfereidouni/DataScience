import requests
import json
import os
from tqdm import tqdm

# The base URL and output directory
# BASE_URL = "https://statsapi.web.nhl.com/api/v1/game/{}/feed/live/"
BASE_URL = "https://api-web.nhle.com/v1/gamecenter/{}/play-by-play"
OUTPUT_DIR = "nhl_data"

# Ensure the output directories exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def generate_game_ids(season, playoffs=False):
    """
    Generate game IDs based on the given NHL season and whether it's for playoffs or regular season.

    Parameters:
    - season (int): The year of the NHL season. 
    - playoffs (bool): Whether the IDs are for playoffs. Default is False (for regular season).

    Returns:
    - list of str: A list of game IDs for the specified season and type (playoffs or regular season).
    """
    
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
    """
    Download game data using the given game ID and save it as a JSON file.

    Parameters:
    - game_id (str): The game ID for which data is to be downloaded.
    - playoffs (bool): Whether the game is a playoff game. Default is False (for regular season).

    Returns:
    None
    """
    
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
    for season in range(2016, 2022):
        print(f"Downloading regular season data for {season}-{season+1} season...")

        # Added tqdm to the loop for progress bar visualization
        for game_id in tqdm(generate_game_ids(season, playoffs=False), desc="Regular Season", unit="game"):
            download_data(game_id, playoffs=False)

        print(f"Finished downloading regular season data for {season}-{season+1} season!")

        print(f"Downloading playoff data for {season}-{season+1} season...")

        for game_id in tqdm(generate_game_ids(season, playoffs=True), desc="Playoffs", unit="game"):
            download_data(game_id, playoffs=True)

        print(f"Finished downloading playoff data for {season}-{season+1} season!")

print("All done!")
