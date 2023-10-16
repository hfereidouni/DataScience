---
layout: page
title: Data Acquisition
permalink: /milestone1/dataacquisition
---

## Tutorial: Downloading NHL Play-by-Play Data
Are you looking for a way to download the NHL play-by-play data for both the regular season and playoffs? Look no further! In this guide, we'll provide you with a Python-based tool to efficiently fetch this data from the NHL's API for any specified seasons. Let's get started!

### Understanding the NHL API:
The NHL API endpoint for retrieving play-by-play data is: https://statsapi.web.nhl.com/api/v1/game/[GAME_ID]/feed/live/. The GAME_ID is structured in a specific way to reflect the season, type of game (regular season, playoffs, etc.), and specific game number.

The format is YYYYTTGGGG:

YYYY: First 4 digits identify the season.
TT: Type of game (02 = regular season, 03 = playoffs).
GGGG: Specific game number.

### The Code:
We've created a Python script for this task. Here's a breakdown of the code:
- Setup: Import necessary libraries and setup constants.
- Game ID Generation: Based on the season and game type, generate the required game IDs.
- Data Download: Using the generated game IDs, fetch the data from the NHL API and save it as JSON.



```python

import requests
import json
import os
from tqdm import tqdm

# The base URL and output directory
BASE_URL = "https://statsapi.web.nhl.com/api/v1/game/{}/feed/live/"
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
    for season in range(2016, 2021):
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

```

### Code Description:

#### Purpose:
This code is a script to download NHL game data for both regular season and playoffs from a given API endpoint. It covers the seasons from 2016 to 2020.

#### Imports:
* `requests`: To make HTTP requests to the API.
* `json`: To parse and save the JSON response from the API.
* `os`: To interact with the filesystem, ensuring directories exist.
* `tqdm`: To display a progress bar when downloading data.

#### Constants:
`BASE_URL`: A string template of the API endpoint from where game data is fetched.

`OUTPUT_DIR`: The directory name where the downloaded JSON files are saved.

Ensure Directories:
The code checks if the `OUTPUT_DIR` exists, and if not, it creates the directory.

#### Functions:
`generate_game_ids(season, playoffs=False)`:

Generates game IDs based on the given NHL season and specifies if the data is for playoffs or the regular season.
Returns a list of game IDs for the specified season and type (playoffs or regular season).

`download_data(game_id, playoffs=False)`:

Fetches game data from the API using a given game ID and saves the data as a JSON file in the specified directory.
This function doesn't return any value.

#### Main Execution:
The `if __name__ == "__main__"` block does the following:

Iterates over the seasons from 2016 to 2020.
For each season:
- Announces the downloading of regular season data.
- Uses `tqdm` to display a progress bar while downloading regular season games data.
- Announces the completion of regular season data download.
- Announces the downloading of playoff data.
- Uses `tqdm` to display a progress bar while downloading playoff games data.
- Announces the completion of playoff data download.
- After iterating through all seasons, it prints "All done!" to indicate the end of the script's execution.

#### Observations:
The number of total games in the season is determined by the number of teams playing in that season.
Game IDs are constructed differently for regular seasons and playoffs.
`tqdm` is used to give a visual representation of the download progress for each season.
In the `download_data()` function, there's a redundant check for playoffs (`if not playoffs:... else:...`).
The script currently covers only seasons from 2016 to 2020, but it could be easily extended to other seasons if needed.

### How to Use:
Ensure you have the necessary libraries installed:

`pip install requests tqdm`

Save the above code in a Python script named `1DataAcquisition.py`

Run the script:

`python 1DataAcquisition.py`

Once executed, the data for each game will be saved as individual JSON files in a directory named `nhl_data`.

Note: It's crucial to ensure that you're not overloading the NHL API servers with numerous requests. Also, please do not store large amounts of data on Git repositories; it's not recommended.

And that's it! Now you have a straightforward tool to download NHL play-by-play data. Happy analyzing!

### Files:

**File name in the codebase:** `1DataAcquisition.py` and `1DataAcquisition.ipynb`

### References:
https://gitlab.com/dword4/nhlapi/-/blob/master/stats-api.md#game-ids

