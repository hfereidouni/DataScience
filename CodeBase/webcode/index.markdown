---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---
<!-- main -->
<ul> 
<!-- main -->
<li style="font-size:30px;color: gray;"><strong>1. Data Acquisition</strong></li>
<p>
<li><strong>Understanding the NHL API</strong></li>
</p>
<p>
The NHL API endpoint for retrieving play-by-play data is: https://statsapi.web.nhl.com/api/v1/game/[GAME_ID]/feed/live/. The GAME_ID is structured in a specific way to reflect the season, type of game (regular season, playoffs, etc.), and specific game number.
</p>
<p>
The format is YYYYTTGGGG:
</p>
<p>
YYYY: First 4 digits identify the season. TT: Type of game (02 = regular season, 03 = playoffs). GGGG: Specific game number.
</p>

<li><strong>The Code:</strong></li>

<p>We've created a Python script for this task. Here's a breakdown of the code:</p>

<ul>
<li>Setup: Import necessary libraries and setup constants.</li>
<li>Game ID Generation: Based on the season and game type, generate the required game IDs.</li>
<li>Data Download: Using the generated game IDs, fetch the data from the NHL API and save it as JSON.</li>
</ul>
<p>


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

</p>
<p>
<li><strong>Code Description/purpose:</strong></li>
</p>
<p>
This code is a script to download NHL game data for both regular season and playoffs from a given API endpoint. It covers the seasons from 2016 to 2020.
</p>
<p>
Imports:<li><strong>Imports:</strong></li>
</p>
<ul>
<li>
'requests': To make HTTP requests to the API.
</li>
<li>
json: To parse and save the JSON response from the API.
</li>
<li>
os: To interact with the filesystem, ensuring directories exist.
</li>
<li>
tqdm: To display a progress bar when downloading data.
</li>
</ul>
<li><strong>
Constants:</strong>
</li>
<ul>
<li>
BASE_URL: A string template of the API endpoint from where game data is fetched.
</li>
<li>
OUTPUT_DIR: The directory name where the downloaded JSON files are saved.
</li>
<li>
Ensure Directories: The code checks if the 'OUTPUT_DIR' exists, and if not, it creates the directory.
</li>
</ul>
<li><strong>
Functions:</strong>
</li>
<ul>
<li>generate_game_ids(season, playoffs=False):
</li>
Generates game IDs based on the given NHL season and specifies if the data is for playoffs or the regular season.
Returns a list of game IDs for the specified season and type (playoffs or regular season).
<li>
download_data(game_id, playoffs=False):
</li>
Fetches game data from the API using a given game ID and saves the data as a JSON file in the specified directory.
This function doesn't return any value.
</ul>
<li><strong>
Main Execution:</strong>
</li>
The `if __name__ == "__main__"` block does the following:
For each season:
<ul>
<li>Announces the downloading of regular season data.</li>
<li>Uses `tqdm` to display a progress bar while downloading regular season games data.</li>
<li>Announces the completion of regular season data download.</li>
<li>Announces the downloading of playoff data.</li>
<li>Uses `tqdm` to display a progress bar while downloading playoff games data.</li>
<li>Announces the completion of playoff data download.</li>
<li>After iterating through all seasons, it prints "All done!" to indicate the end of the script's execution.</li>
</ul>
<li><strong>Observations:</strong></li>
The number of total games in the season is determined by the number of teams playing in that season.
Game IDs are constructed differently for regular seasons and playoffs.
`tqdm` is used to give a visual representation of the download progress for each season.
In the `download_data()` function, there's a redundant check for playoffs (`if not playoffs:... else:...`).
The script currently covers only seasons from 2016 to 2020, but it could be easily extended to other seasons if needed.
<p>
<li><strong>How to Use:</strong></li></p>
Ensure you have the necessary libraries installed:
<ul>
<li>pip install requests tqdm</li>
Save the above code in a Python script named '1DataAcquisition.py'
<li>python 1DataAcquisition.py</li>
Once executed, the data for each game will be saved as individual JSON files in a directory named `nhl_data`.

Note: It's crucial to ensure that you're not overloading the NHL API servers with numerous requests. Also, please do not store large amounts of data on Git repositories; it's not recommended.

And that's it! Now you have a straightforward tool to download NHL play-by-play data. Happy analyzing!

</ul>
<li><strong>
References:</strong>
</li>
https://gitlab.com/dword4/nhlapi/-/blob/master/stats-api.md#game-ids




<!-- main -->
</ul>
<!-- main -->



<!-- ========================================================================================================================================== -->



<!-- main -->
<ul> 
<!-- main -->
<li style="font-size:30px;color: gray;"><strong>2. Interactive Debugging Tool</strong></li>

<ul><li><strong>Question. Implement an ipywidget, draw the event coordinates on the provided ice rink image.</strong> </li>
<p><li><strong>Answer:</strong></li></p>

<!-- <p> -->
<!-- <figure>
<img src="tidy data snippet.png" alt = "Alt text">
<figcaption style="font-size: 11px; font-style: italic; text-align: center; font-weight: bold;">Small snippet of final data frame of season 2016-2017.</figcaption>
<figcaption style="text-align: center; font-weight: bold;"> Small snippet of final data frame of season 2016-2017 </figcaption>
<!-- </figure> -->
The task involves creating an interactive debugging tool using ipywidgets in a Jupyter notebook to explore NHL game data. The tool allows you to navigate through events in different games of a given season, including regular season and playoffs. It displays event coordinates on an ice rink image and optionally shows other relevant information such as game metadata and event summaries. The goal is to create a user-friendly interface for exploring NHL game data.</p>
<p>Therefore, to accomplish this task, we need to:</p>
<ul>
<li>Use ipywidgets to create a user interface for navigating through events.</li>
<li>Load game data for the specified season, including regular season and playoff games.</li>
<li>Implement logic to display event coordinates on an ice rink image for each event.</li>
<li>Optionally, display additional information such as game metadata, box scores, and event summaries.
</li>
</ul>
<li><strong>The Code:</strong></li>

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

<li><strong>Code Description/Purpose:</strong></li>
The purpose of the provided code is to create an interactive debugging and exploration tool for NHL (National Hockey League) game data using ipywidgets in a Jupyter notebook. This tool serves several key purposes:
<ul>
<li>**Data Exploration:** The primary purpose is to allow users to interactively explore and visualize NHL game data for a specific season. Users can select a season year and switch between regular season and playoff games.</li>
<li>**Event Visualization:** The code enables users to navigate through events within each game, displaying event coordinates on an ice rink image. Different event types are distinguished on the image.</li>
<li>**Game Information:** The tool provides additional information such as event descriptions, game metadata, box scores, and event summaries for the selected events.</li>
<li>**User-Friendly Interface:** It offers a user-friendly interface with dropdowns for selecting the season and game type (Regular or Playoffs), sliders for choosing the game and event, and a text area to display information.</li>
<li>**Flexibility:** Users can easily switch between different games and events within those games, making it a versatile tool for exploring various aspects of NHL games.</li>
<li>**Debugging and Analysis:** While the primary goal is exploration, the tool can also be used for debugging and analysis of NHL game data.</li>
</ul>
<li><strong>Imports:</strong></li>
<ul>
<li>ipywidgets as widgets: Imports the ipywidgets library and renames it as widgets. This library is used to create interactive widgets in Jupyter notebooks, facilitating user interaction.</li>
<li>numpy as np: Imports the NumPy library, aliased as np. NumPy is widely used for numerical and array operations, and it may be used for various calculations in the code.</li>
<li>from IPython.display import display, clear_output: Imports specific functions, display and clear_output, from the IPython.display module. These functions are used to display widgets and clear notebook output for updates.</li>
<li>import matplotlib.pyplot as plt`: Imports the pyplot module from the Matplotlib library, aliased as plt. Matplotlib is a popular library for creating various types of plots and visualizations.</li>
<li>import matplotlib.image as mpimg`: Imports the image submodule of Matplotlib, aliased as mpimg. This submodule is used for reading and displaying images within the Jupyter notebook.</li>
<li>import json: Imports the built-in json module, which is used for working with JSON data, including loading and parsing JSON files.
</li>
</ul>
<li><strong>Class and Functions:</strong></li>
<ul>
<li>Initialization:** The class is initialized with a specific NHL season year, and the default game type is set to "Regular." It creates several widgets, including Dropdowns for selecting the year and game type, IntSliders for selecting the game ID and event within the game, and a Textarea for displaying game information.</li>
<li>Loading Game Data:** The ‍`load_game_data` method loads game data based on the selected year, game type, and game ID. It uses JSON files containing NHL game data.</li>
<li>Updating Year and Game Type:** The `update_year` and `update_game_type` methods are event handlers for changing the year and game type Dropdowns. They update the selected year and game type, reset relevant sliders, and update the event range.</li>
<li>Updating Event Range:** The `update_event_range` method updates the maximum value of the event slider based on the selected game's data. It also handles special cases where data may be unavailable for specific games.</li>
<li>Displaying Events:** The `display_event` method displays the selected event within the game using `matplotlib` plots. It loads game data, extracts event coordinates, and plots them on an ice rink image. Event types such as "Shot" and "Other Events" are distinguished by blue and red markers. Event descriptions and game metadata are displayed as well.</li>
<li>Displaying Widgets:** The `display_widgets` method clears the notebook's output, displays all the interactive widgets, and updates the event range. It also calls `display_event` to show the selected event.</li>
<li>Browsing Games:** The `browse_games` method sets up interactivity by observing changes in the game and event sliders. It calls `display_widgets` to display the widgets.</li>
<li>Creating an Instance:** Finally, an instance of `NHLExplorer` is created for the 2016 season, and the `browse_games` method is called to initiate the interactive exploration of NHL games.
</li>
</ul>
<li><strong>Observations:</strong></li>
The tool is an interactive debugging and exploration tool for NHL (National Hockey League) game data. It is designed to be used within a Jupyter notebook and offers an intuitive interface for exploring NHL game data for a specific season, with the ability to switch between regular season and playoffs.
<p>
<figure>
<img src="integration1.png" alt = "Alt text">
<figcaption style="font-size: 11px; font-style: italic; text-align: center; font-weight: bold;">Fig. 1</figcaption>
</figure>
</p>
According to the above-mentioned (Fig. 1), at the beginning of each game, we can see the event 0 (including the game info) and then after swiping the event slider we can see the respective event info through the Textarea. 

<p>
<figure>
<img src="integration2.png" alt = "Alt text">
<figcaption style="font-size: 11px; font-style: italic; text-align: center; font-weight: bold;">Fig. 2.</figcaption>

</figure>
</p>
In the Fig. 2 mentioned above, we can see that the coordination of the events, the events descroption and so forth. 
<li><strong>How to use:</strong></li>
<ul>
<li>Setup Environment:</li>
<ul>
<li>Ensure you have Jupyter Notebook or Jupyter Lab installed on your system.</li>
<li>Install the required Python libraries, including `ipywidgets`, `numpy`, `matplotlib`, and `json`, if you haven't already.</li>
</ul>
<li>Download Data Files:</li>
<ul><li> Ensure you have the necessary NHL game data files available in a directory called `nhl_data`. These JSON files should contain the game data you want to explore.</li></ul>
<li>Copy and Paste the Code:</li>
<ul>
<li>Copy the entire code provided in the question and paste it into a Jupyter Notebook cell.</li>
<li>Copy the ice rink image from the codebase and put it in the same directory with the code.</li>
</ul>
<li>Run the Code:</li>
<ul><li>Execute the code cell in the Jupyter Notebook environment. This will initialize the `NHLExplorer` class and create an interactive tool for exploring NHL game data.</li></ul>
<li>Interact with the Tool:</li>
<ul>
<li>Once the code is executed, the tool's interface will appear in the notebook.</li>
<li>Use the dropdown menus to select the desired season and game type (Regular or Playoffs).</li>
<li>Adjust the game slider to choose a specific game ID.</li>
<li>Adjust the event slider to select an event within the chosen game.</li>
<li>The tool will display event coordinates on an ice rink image, along with event descriptions and game information.</li>
</ul>
<li>Explore NHL Game Data:</li>
<ul>
<li>Navigate through different events within games by moving the event slider.</li>
<li>Switch between games and explore event data interactively.</li>
<li> Observe event coordinates on the ice rink image and read event descriptions.</li>
</ul>
<li>Switch Season/Game Type:</li>
<ul>
<li>To explore NHL game data for a different season or game type, use the dropdown menus to make your selections.</li>
<li>The tool will dynamically update the available games and events based on your choices.</li>
</ul>
<li>Note on Data Availability:</li>
<ul><li>The tool includes error handling for cases where data for a specific game may be unavailable, particularly in the **2019 Regular season.</li></ul>
</ul>
<li>End Interaction:</li>
<ul><li>When you're done exploring NHL game data, you can close the Jupyter Notebook or clear the output of the code cell to end the tool's interaction.</li></ul>

</ul>
<li><strong>
References:</strong>
</li>
https://ipywidgets.readthedocs.io/en/latest/

https://www.nhl.com/gamecenter/tor-vs-wpg/2017/10/04/2017020001/playbyplay



<!-- main -->
</ul>
<!-- main -->





















<!-- ======================================================================================================================================================== -->


<!-- main -->
<ul> 
<!-- main -->
<li style="font-size:30px;color: gray;"><strong>4. Tidy Data</strong></li>
<p>
<ul><li><strong>Question 1:</strong> Include a small snippet of your final dataframe
<p><strong>Answer:</strong></p>
<p>
<figure>
<img src="tidy data snippet.png" alt = "Alt text">
<figcaption style="font-size: 11px; font-style: italic; text-align: center; font-weight: bold;">Small snippet of final data frame of season 2016-2017.</figcaption>
<!-- <figcaption style="text-align: center; font-weight: bold;"> Small snippet of final data frame of season 2016-2017 </figcaption> -->
</figure>
</p>
<p>
The tabular data in the image above contains 16 columns that provide us with a variety of information, including play type, game time, coordinate, shooter name, goalie name, and much more.
</p>
</li>
<li>
<strong>Question 2:</strong> How you could add the actual strength information to both shots and goals, given the other event types and features available.
<p><strong>Answer:</strong></p>
<p>
<ul type = "None">
<li><strong>Step 1:</strong> At the beginning of the game, set the strength to even</li>
<li><strong>Step 2:</strong> If a penalty happened in an event, save the penalty_minuite X as an integer, pen_team_ID as string.</li>
<li><strong>Step 3:</strong> For plays in the next X minuite, if the ID of the team who has done the shot is the same as pen_ID, then the strength attribute will be set as "short handed", otherwise the strength will be set as power "play".</li>
<li>
<li><strong> Step 4:</strong></li>
<ul>
<li>If the play was only a shot(not goal), for the next play we will do the same thing as step 3 until X minuites later or next goal.</li>
<li>If the play was a goal, we also do the same things as step 3, but then the plays afterwards will have a strength of even if no other penalties happen.</li>
</ul></li></ul></p><li>
<strong>Question 3:</strong> Discuss at least 3 additional features you could consider creating from the data available in this dataset.
<p><strong>Answer:</strong></p>
<p>
<p style="text-align: center; font-weight: bold;">Additional Features.</p>
<ol>
<li>Rebound: if another shot from a player from the same team happens within 3 seconds before a shot/goal, this can be considered as a rebound.</li>
<li>Shot off the rush: if a giveaway is happened 15 seconds before a shot, this shot can be considered as a Shot off the rush.</li>
<li>Shot distance: first to determinate which rinkside the shot team defend, then calculate the distances using the shot coordinates and the coordinates of the net on the opposite rinkside.</li></ol></p></li></li></ul>
</p>
<!-- main -->
</ul>
<!-- main -->

<!-- main -->
<ul> 
<!-- main -->
<li style="font-size:30px;color: gray;"><strong>5. Simple Visualizations</strong></li>

<p>
<ul><li><strong>Question 1:</strong> Produce a figure comparing the shot types over all teams (i.e. just aggregate all of the shots), in a season of your choosing.
<p><strong>Answer:</strong></p>
<p>
<figure>
<img src="q1_type_goal.png" alt = "Alt text">
<figcaption style="font-size: 11px; font-style: italic; text-align: center; font-weight: bold;">Figure comparing the shot types over all teams in season 2016-2017.</figcaption>
<!-- <figcaption style="text-align: center; font-weight: bold;"> Small snippet of final data frame of season 2016-2017 </figcaption> -->
</figure>
</p>
<p>
The data in the form of bar in the image above gives ralationship between type of shot and chances of scoring a goal with that shot.
</p>
<ol>
<li>
 What appears to be the most dangerous type of shot?
<p>Answer: For season 2016-2017, Deflected is the most dangerous one, its chance of shot has gotton 19.8% which is the highest among all.</p>
</li>
<li>
 The most common type of shot?
<p>Answer: Wrist shot is the most common type of shot.</p>
</li>
<li>
 Why did you choose this figure?
<p>Answer: This figure shows number of shots, goals and chance of goal for each type of shot.</p>
</li>
</ol>
</li>
<li>
<strong>Question 2:</strong> What is the relationship between the distance a shot was taken and the chance it was a goal? 
<p><strong>Answer:</strong></p>
<p>
<p>
<img src="q2_dist_goal_2018.png" alt = "Alt text">
<figcaption style="font-size: 11px; font-style: italic; text-align: center; font-weight: bold;">Snippet of shot distance and chance of goal for season 2018-2019.</figcaption>
</p>
<p>
<img src="q2_dist_goal_2019.png" alt = "Alt text">
<figcaption style="font-size: 11px; font-style: italic; text-align: center; font-weight: bold;">Snippet of shot distance and chance of goal for season 2019-2020.</figcaption>
</p>
<p>
<img src="q2_dist_goal_2020.png" alt = "Alt text">
<figcaption style="font-size: 11px; font-style: italic; text-align: center; font-weight: bold;">Snippet of shot distance and chance of goal for season 2020-2021.</figcaption>
</p>
</p>

<li>
<strong>Question 3:</strong>Produce a figure that shows the goal percentage (# goals / # shots) as a function of both distance from the net, and the category of shot types.
<p><strong>Answer:</strong></p>
<img src="q2_dist_type_goal.png" alt = "Alt text">
<figcaption style="font-size: 11px; font-style: italic; text-align: center; font-weight: bold;">Snippet showing relationship between shot distance, shot type and chance of goal for season 2018-2019.</figcaption>
<p>
<strong>Our Findings</strong>
</p>

<ul>
<li>We have used a heatmap to show the goal percentage as a function of both distance from the net, and the category of shot types， the y axis represents different range of shot distance, x axis shows different type of shots, each grid shows the chance of goals as function of a distance and a shot type(ex. if a Wrist Shot happens in a distance between 75 and 100, the chance of goal is 4.6%).</li>
<li>
The brighter the color of a grid is, the more chance of goal it will be. White grid represents NaN value, which means there's no such cases for us to calculate the chance(ex.0 Wrap-around shot at 125 to 150).
</li>
</ul>
<p>
<p>
<strong>What might be the most dangerous types of shots?</strong>
</p>
</p>
<ul>
<p>
<li>Generally speaking, the closer the distance is, the more chance of goal it will be regardless of the type of shots(the first row of the map is obviously brighter than others).</li>
<li>Wrap-around at a distance between 25 and 50(which is a relavently close distance to net) and Deflected at a distance between 150 and 175.</li>
</p>
</ul>
</li>
</li>
</ul>
</p>

<!-- main -->
</ul>
<!-- main -->

<!-- ============================================================================================================================================================== -->


<!-- main -->
<ul> 
<!-- main -->
<li style="font-size:30px;color: gray;"><strong>6. Advanced Visualizations: Shot Maps</strong></li>

<ul><li><strong>Question 1:</strong> Export the 4 plot offensive zone plots to HTML, and embed it into your blog post.</li>
<p><strong>Answer:</strong></p>
<p>
<figure>
<!-- <img src="q3Shot_Map_2016_2017.html" alt = "Alt text"> -->
<iframe src="q3Shot_Map_2016_2017.html" width="700" height="500" frameborder="0"></iframe>
<!-- <figcaption style="font-size: 11px; font-style: italic; text-align: center; font-weight: bold;">Shot_Map_2016_2017.</figcaption> -->
</figure>
</p>
<p>
<figure>
<!-- <img src="q3Shot_Map_2016_2017.html" alt = "Alt text"> -->
<iframe src="Shot_Map_2017_2018.html" width="700" height="500" frameborder="0"></iframe>
<!-- <figcaption style="font-size: 11px; font-style: italic; text-align: center; font-weight: bold;">Shot_Map_2016_2017.</figcaption> -->
</figure>
</p>
<p>
<figure>
<!-- <img src="q3Shot_Map_2016_2017.html" alt = "Alt text"> -->
<iframe src="Shot_Map_2018_2019.html" width="700" height="500" frameborder="0"></iframe>
<!-- <figcaption style="font-size: 11px; font-style: italic; text-align: center; font-weight: bold;">Shot_Map_2016_2017.</figcaption> -->
</figure>
</p>
<p>
<figure>
<!-- <img src="q3Shot_Map_2016_2017.html" alt = "Alt text"> -->
<iframe src="q3Shot_Map_2020_2021.html" width="700" height="500" frameborder="0"></iframe>
<!-- <figcaption style="font-size: 11px; font-style: italic; text-align: center; font-weight: bold;">Shot_Map_2016_2017.</figcaption> -->
</figure>
</p>

<li><strong>Question 2:</strong> Discuss (in a few sentences) what you can interpret from these plots.</li>
<li><strong>Answer:</strong></li>
<p>
<p>
We can interpret the following from the shot maps
<ul>
<li>Team Preference for using left or right side of the goal for attacking shots (higher excess shot rate per hour)</li>
<li>Range of the area used near the goal for the most attacking shots (highly density small area for attack shots v/s highly spread large area for attack shots in different teams)</li>
</ul></p>

<li><strong>Question 3:</strong> Consider the Colorado Avalanche; take a look at their shot map during the 2016-17 season. Discuss what you could say about the team during this season. Now look at the shot map for the Colorado Avalanche for the 2020-21 season, and discuss what you could conclude from these differences. Does this make sense?</li>
<li><strong>Answer:</strong></li>
<p>
<p>
Colorado Avalanche, 2016-2017 : The team prefers an attacking approach from the left side of the rink to shoot in the goal. The team avoids usage of center for attacking.
<p>
Colorado Avalanche, 2020-2021 along with difference : The team still prefers to attack from the left side, but now concentrates its attacking near the goal area, along with experimenting with the usage of center position for attack. In conclusion, the restriction of their attack space may highlight their passing efficiency to reach near the goal with much better goal success rate.
</p>

</p>

<li><strong>Question 4:</strong> Consider the Buffalo Sabres, which have been a team that has struggled over recent years, and compare them to the Tampa Bay Lightning, a team which has won the Stanley for two years in a row. Look at the shot maps for these two teams from the 2018-19, 2019-20, and 2020-21 seasons. Discuss what observations you can make. Is there anything that could explain the Lightning’s success, or the Sabres’ struggles? How complete of a picture do you think this paints?
</li>
<li><strong>Answer:</strong></li>
<p>
<p>
Despite being not able to get a complete picture of Tampa Bay Lightning's actual reason for success and Buffalo Sabres struggle only through the shot maps, we do notice a shift in shot dynamics in the rink between these two teams. The fact that Tampa Bay Lightning started to experiment with shots from center and shots from left and right side of the offensive zone in the 2019-2020 season and 2020-2021 season respectively, shows that they have gained much more control in using either of the sides for attacking. In a similar fashion, noticeable loss of this ability (attacking from both sides in offensive zone) has been shown by Buffalo Sabres from their shot maps in the seasons from 2018 till 2021. 
Such a control may be justified by the following:</p>
<ul>
<li>Change in attacking strategy</li>
<li>Balance/ Imbalance in the left-handed and right-handed players in a team</li>
</ul>





</ul>
<!-- main -->
</ul>
<!-- main -->
