---
layout: page
title: Data Acquisition
permalink: /milestone1/dataacquisition
---

## Interactive Debugging Tool
**Question:** Implement an ipywidget that allows you to flip through all of the events, for every game of a given season, with the ability to switch between the regular season and playoffs. Draw the event coordinates on the provided ice rink image, similar to the example shown below (you can just print the event data when there are no coordinates). You may also print whatever information you find useful, such as game metadata/boxscores, and event summaries (but this is not required). Take a screenshot of the tool and add it to the blog post, accompanied with the code for the tool and a brief (1-2 sentences) description of what your tool does. You do not need to worry about embedding the tool into the blog post.

**Answer:** The task involves creating an interactive debugging tool using ipywidgets in a Jupyter notebook to explore NHL game data. The tool allows you to navigate through events in different games of a given season, including regular season and playoffs. It displays event coordinates on an ice rink image and optionally shows other relevant information such as game metadata and event summaries. The goal is to create a user-friendly interface for exploring NHL game data.

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

### Code Description:

#### Purpose:

The purpose of the provided code is to create an interactive debugging and exploration tool for NHL (National Hockey League) game data using ipywidgets in a Jupyter notebook. This tool serves several key purposes:

1. **Data Exploration:** The primary purpose is to allow users to interactively explore and visualize NHL game data for a specific season. Users can select a season year and switch between regular season and playoff games.

2. **Event Visualization:** The code enables users to navigate through events within each game, displaying event coordinates on an ice rink image. Different event types are distinguished on the image.

3. **Game Information:** The tool provides additional information such as event descriptions, game metadata, box scores, and event summaries for the selected events.

4. **User-Friendly Interface:** It offers a user-friendly interface with dropdowns for selecting the season and game type (Regular or Playoffs), sliders for choosing the game and event, and a text area to display information.

5. **Flexibility:** Users can easily switch between different games and events within those games, making it a versatile tool for exploring various aspects of NHL games.

6. **Debugging and Analysis:** While the primary goal is exploration, the tool can also be used for debugging and analysis of NHL game data.



#### Imports:

* `ipywidgets as widgets`: Imports the ipywidgets library and renames it as widgets. This library is used to create interactive widgets in Jupyter notebooks, facilitating user interaction.
* `numpy as np`: Imports the NumPy library, aliased as np. NumPy is widely used for numerical and array operations, and it may be used for various calculations in the code.
* `from IPython.display import display, clear_output`: Imports specific functions, display and clear_output, from the IPython.display module. These functions are used to display widgets and clear notebook output for updates.
* `import matplotlib.pyplot as plt`: Imports the pyplot module from the Matplotlib library, aliased as plt. Matplotlib is a popular library for creating various types of plots and visualizations.
* `import matplotlib.image as mpimg`: Imports the image submodule of Matplotlib, aliased as mpimg. This submodule is used for reading and displaying images within the Jupyter notebook.
* `import json`: Imports the built-in json module, which is used for working with JSON data, including loading and parsing JSON files.


#### Class and Functions:

The code defines a Python class called `NHLExplorer`, which is designed to interactively explore and visualize NHL game data using ipywidgets. Here's how the code works:
* **Initialization:** The class is initialized with a specific NHL season year, and the default game type is set to "Regular." It creates several widgets, including Dropdowns for selecting the year and game type, IntSliders for selecting the game ID and event within the game, and a Textarea for displaying game information.
* **Loading Game Data:** The ‍`load_game_data` method loads game data based on the selected year, game type, and game ID. It uses JSON files containing NHL game data.
* **Updating Year and Game Type:** The `update_year` and `update_game_type` methods are event handlers for changing the year and game type Dropdowns. They update the selected year and game type, reset relevant sliders, and update the event range.
* **Updating Event Range:** The `update_event_range` method updates the maximum value of the event slider based on the selected game's data. It also handles special cases where data may be unavailable for specific games.
* **Displaying Events:** The `display_event` method displays the selected event within the game using `matplotlib` plots. It loads game data, extracts event coordinates, and plots them on an ice rink image. Event types such as "Shot" and "Other Events" are distinguished by blue and red markers. Event descriptions and game metadata are displayed as well.
* **Displaying Widgets:** The `display_widgets` method clears the notebook's output, displays all the interactive widgets, and updates the event range. It also calls `display_event` to show the selected event.
* **Browsing Games:** The `browse_games` method sets up interactivity by observing changes in the game and event sliders. It calls `display_widgets` to display the widgets.
* **Creating an Instance:** Finally, an instance of `NHLExplorer` is created for the 2016 season, and the `browse_games` method is called to initiate the interactive exploration of NHL games.

#### Observations:
The tool is an interactive debugging and exploration tool for NHL (National Hockey League) game data. It is designed to be used within a Jupyter notebook and offers an intuitive interface for exploring NHL game data for a specific season, with the ability to switch between regular season and playoffs.

### How to Use:

1. **Setup Environment:**
    - Ensure you have Jupyter Notebook or Jupyter Lab installed on your system.
    - Install the required Python libraries, including ipywidgets, numpy, matplotlib, and json, if you haven't already.

2. **Download Data Files:**
    - Ensure you have the necessary NHL game data files available in a directory called nhl_data. These JSON files should contain the game data you want to explore.

3. **Copy and Paste the Code:**
    - Copy the entire code provided in the question and paste it into a Jupyter Notebook cell.
    - Copy the ice rink image from the codebase and put it in the same directory with the code.

4. **Run the Code:**
    - Execute the code cell in the Jupyter Notebook environment. This will initialize the NHLExplorer class and create an interactive tool for exploring NHL game data.

5. **Interact with the Tool:**
    - Once the code is executed, the tool's interface will appear in the notebook.
    - Use the dropdown menus to select the desired season and game type (Regular or Playoffs).
    - Adjust the game slider to choose a specific game ID.
    - Adjust the event slider to select an event within the chosen game.
    - The tool will display event coordinates on an ice rink image, along with event descriptions and game information.

6. **Explore NHL Game Data:**
    - Navigate through different events within games by moving the event slider.
    - Switch between games and explore event data interactively.
    - Observe event coordinates on the ice rink image and read event descriptions.

7. **Switch Season/Game Type:**
    - To explore NHL game data for a different season or game type, use the dropdown menus to make your selections.
    - The tool will dynamically update the available games and events based on your choices.

8. **Note on Data Availability:**
    - The tool includes error handling for cases where data for a specific game may be unavailable, particularly in the 2019 Regular season.

9. **End Interaction:**
    - When you're done exploring NHL game data, you can close the Jupyter Notebook or clear the output of the code cell to end the tool's interaction.

### References:
https://gitlab.com/dword4/nhlapi/-/blob/master/stats-api.md#game-ids

