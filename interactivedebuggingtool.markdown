---
layout: page
title: Data Acquisition
permalink: /milestone1/dataacquisition
---

## Interactive Debugging Tool
Question: Implement an ipywidget that allows you to flip through all of the events, for every game of a given season, with the ability to switch between the regular season and playoffs. Draw the event coordinates on the provided ice rink image, similar to the example shown below (you can just print the event data when there are no coordinates). You may also print whatever information you find useful, such as game metadata/boxscores, and event summaries (but this is not required). Take a screenshot of the tool and add it to the blog post, accompanied with the code for the tool and a brief (1-2 sentences) description of what your tool does. You do not need to worry about embedding the tool into the blog post.

Answer: The task involves creating an interactive debugging tool using ipywidgets in a Jupyter notebook to explore NHL game data. The tool allows you to navigate through events in different games of a given season, including regular season and playoffs. It displays event coordinates on an ice rink image and optionally shows other relevant information such as game metadata and event summaries. The goal is to create a user-friendly interface for exploring NHL game data.

Therefore, to accomplish this task, we need to:

- Use ipywidgets to create a user interface for navigating through events.
- Load game data for the specified season, including regular season and playoff games.
- Implement logic to display event coordinates on an ice rink image for each event.
- Optionally, display additional information such as game metadata, box scores, and event summaries.

### The Code:

```python
# Import necessary libraries and modules
import ipywidgets as widgets
import numpy as np
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json

# Define the NHLExplorer class to explore and visualize NHL games data
class NHLExplorer:
    """
    Class to explore and visualize NHL game data using interactive widgets.
    
    Attributes:
    season (int): The NHL season year to explore.
    game_type (str): Type of the game ('Regular' or 'Playoffs').

    Methods:
    load_game_data(self, game_id)
    update_year(self, change)
    update_game_type(self, change)
    update_event_range(self)
    display_event(self, game_id, event_idx)
    display_widgets(self)
    browse_games(self)
    """

    def __init__(self, season):
        """
        Initialize the NHLExplorer with a specific season.
        
        Args:
        season (int): The NHL season year to explore.
        """
        
        # Initialize the NHLExplorer with a specific season
        self.season = season
        self.game_type = 'Regular'  # Default game type is Regular
        
        # Create a Dropdown widget for selecting the year
        self.year_dropdown = widgets.Dropdown(options=list(range(2016, 2021)), description='Year:')
        self.year_dropdown.observe(self.update_year, 'value')  

        # Create a Dropdown widget for selecting the game type (Regular or Playoffs)
        self.game_type_dropdown = widgets.Dropdown(options=['Regular', 'Playoffs'], description='Game Type:')
        self.game_type_dropdown.observe(self.update_game_type, 'value')

        # Create an IntSlider widget for selecting the game ID; Maximum is 1271 for our case
        self.game_slider = widgets.IntSlider(min=1, max=1271, description='Game ID:')
        
        # Create an IntSlider widget for selecting the event within the game
        self.event_slider = widgets.IntSlider(min=0, max=0, description='Event:')
        
        # Create a Textarea widget for displaying game information
        self.game_info_text = widgets.Textarea(value='', description='Game Info:', disabled=True, rows=10)

        # Display the widgets
        self.display_widgets()
        

    def load_game_data(self, game_id):
        """
        Load game data based on the selected year, game type, and game ID.
        
        Args:
        game_id (int): The ID of the game to be loaded.
        
        Returns:
        dict: A dictionary containing game data, or None if file not found.
        """
        
        # Load game data based on the selected year, game type, and game ID
        year_of_the_game = str(self.season)
        type_number = 20000 if self.game_type == 'Regular' else 30000  # Use 30000 for playoffs
        json_file = f'{year_of_the_game}0{int(type_number) + int(game_id)}'

        try:
            with open(f'nhl_data/{json_file}.json', 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            return None


    def update_year(self, change):
        """
        Update the selected year and event range when the year is changed.
        
        Args:
        change (object): Contains information about the change in the year.
        """
        
        # Update the selected year and event range when the year is changed
        self.season = change.new
        self.update_event_range()
        

    def update_game_type(self, change):
        """
        Update the selected game type and reset the game slider and event slider when game type changes.
        
        Args:
        change (object): Contains information about the change in the game type.
        """
        
        # Update the selected game type and reset the game slider and event slider when game type changes
        self.game_type = change.new
        self.game_slider.value = 1  # Reset game slider to 1 when game type changes
        self.event_slider.max = 0  # Reset event slider max to 0 when game type changes
        self.update_event_range()
        self.display_widgets()  # Update the widgets to reflect the new game type

        
    def update_event_range(self):
        """
        Update the maximum value of the event slider based on the selected game's data.
        """
        
        # Update the maximum value of the event slider based on the selected game's data
        if self.season == 2019 and self.game_slider.value > 1082 and self.game_type == 'Regular':
            self.event_slider.value = 0
            self.event_slider.max = 0
            return
        
        game_data = self.load_game_data(self.game_slider.value)
        if game_data is not None:
            all_plays = game_data['liveData']['plays']['allPlays']
            self.event_slider.max = len(all_plays) - 1
            
        else:
            self.event_slider.max = 0

            
    def display_event(self, game_id, event_idx):
        """
        Display the selected event within the game using plots.
        
        Args:
        game_id (int): The ID of the game.
        event_idx (int): The index of the event within the game.
        """
        # Display the selected event within the game
        if self.season == 2019 and game_id > 1082 and self.game_type == 'Regular':
            print("Data is unavailable!")
            self.game_info_text.value = "Data is unavailable for this game!"
            return
        

        game = self.load_game_data(game_id)
        if game is None:
            print("Game data not found.")
            return
        
        event = game['liveData']['plays']['allPlays'][event_idx]
        event_coord = event.get('coordinates', {})

        # Create a plot to display the event
        img = mpimg.imread('ice_rink.png')
        fig, ax = plt.subplots(figsize=(10, 4.25))
        ax.imshow(img, extent=[-100, 100, -42.5, 42.5])
        plt.ylim((-42.5, 42.5))
        my_y_ticks = np.arange(-42.5, 42.5, 21.25)
        plt.yticks(my_y_ticks)
        plt.xlabel('Feet')
        plt.ylabel('Feet')
        plt.title('Events: Shots(Blue)  &  Other Events(Red)')
        
        if 'x' in event_coord and 'y' in event_coord:
            x = event_coord['x']
            y = event_coord['y']
            if event['result']['event'] == 'Shot':
                ax.plot(x, y, 'bo', markersize=12)
            else:
                ax.plot(x, y, 'ro', markersize=12)
            ax.text(x + 2, y + 2, f'[{x}, {y}]', fontsize=12, color='black')
            
        # Add horizontal line at y=0
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        # Add vertical lines at x=89 and x=-89 for the nets
        ax.axvline(89, color='black', linestyle='--', linewidth=1)
        ax.axvline(-89, color='black', linestyle='--', linewidth=1)
        
        plt.show()

        print(event.get('result', {}).get('description', 'No event description available.'))
        game_info = game['liveData']['plays']['allPlays'][event_idx] if event_idx > 0 else game['gameData']
        game_info_text = json.dumps(game_info, indent=4)
        self.game_info_text.value = game_info_text

        
    def display_widgets(self):
        """
        Display all the interactive widgets.
        """
        
        # Display all the widgets
        clear_output(wait=True)
        display(widgets.VBox([self.year_dropdown, self.game_type_dropdown, self.game_slider, self.event_slider]))
        display(self.game_info_text)
        self.update_event_range()  # Update the event range when switching game type
        self.display_event(self.game_slider.value, self.event_slider.value)

        
    def browse_games(self):
        """
        Function to browse and display NHL games interactively.
        """
        
        # Function to browse and display NHL games
        def display_data(change):
            self.display_widgets()

        self.game_slider.observe(display_data, 'value')
        self.event_slider.observe(display_data, 'value')
        self.display_widgets()


# Create an instance of NHLExplorer and start browsing games for the 2016 season
explorer = NHLExplorer(2016)
explorer.browse_games()
```

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



<pre>

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

</pre>

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

### References:
https://gitlab.com/dword4/nhlapi/-/blob/master/stats-api.md#game-ids

