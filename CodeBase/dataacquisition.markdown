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

Setup: Import necessary libraries and setup constants.
Game ID Generation: Based on the season and game type, generate the required game IDs.
Data Download: Using the generated game IDs, fetch the data from the NHL API and save it as JSON.

<pre>
```
import requests
import json
import os
from tqdm import tqdm

BASE_URL = "https://statsapi.web.nhl.com/api/v1/game/{}/feed/live/"
OUTPUT_DIR = "nhl_data"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def generate_game_ids(season, playoffs=False):
    """..."""  

def download_data(game_id, playoffs=False):
    """..."""  

if __name__ == "__main__":
    for season in range(2016, 2021):
        # Code for downloading regular season and playoff data
```
</pre>

### How to Use:
Ensure you have the necessary libraries installed:

`pip install requests tqdm`

Save the above code in a Python script named 1DataAcquisition.py

Run the script:

`python download_nhl_data.py`

Once executed, the data for each game will be saved as individual JSON files in a directory named nhl_data.

Note: It's crucial to ensure that you're not overloading the NHL API servers with numerous requests. Also, please do not store large amounts of data on Git repositories; it's not recommended.

And that's it! Now you have a straightforward tool to download NHL play-by-play data. Happy analyzing!