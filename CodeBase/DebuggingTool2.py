import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json

class NHLExplorer:
    def __init__(self, season):
        self.season = season
        self.game_type = 'Regular'  # Default to Regular games
        self.year_dropdown = widgets.Dropdown(options=list(range(2016, 2021)), description='Year:')
        self.year_dropdown.observe(self.update_year, 'value')  # Attach an observer to handle year changes
        self.game_type_dropdown = widgets.Dropdown(options=['Regular', 'Playoffs'], description='Game Type:')
        self.game_slider = widgets.IntSlider(min=1, max=1353, description='Game ID:')
        self.event_slider = widgets.IntSlider(min=0, max=0, description='Event:')
        self.game_info_text = widgets.Textarea(value='', description='Game Info:', disabled=True)
        
        self.update_event_range()  # Initial update of event slider
        
        # Initial display of widgets
        self.display_widgets()
        
    def load_game_data(self, game_id):
        year_of_the_game = str(self.season)
        type_number = 20000 if self.game_type == 'Regular' else 30000
        json_file = f'{year_of_the_game}0{int(type_number) + int(game_id)}'
        
        try:
            with open(f'nhl_data/{json_file}.json', 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            return None

    def update_year(self, change):
        self.season = change.new
        self.update_event_range()
        
    def update_event_range(self):
        game_data = self.load_game_data(self.game_slider.value)
        if game_data is not None:
            self.event_slider.max = len(game_data['liveData']['plays']['allPlays']) - 1
        else:
            self.event_slider.max = 0

    def display_event(self, game_id, event_idx):
        game = self.load_game_data(game_id)
        if game is None:
            print("Game data not found.")
            return

        event = game['liveData']['plays']['allPlays'][event_idx]
        event_coord = event.get('coordinates', {})

        img = mpimg.imread('Icehockeylayout.svg.png')
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(img, extent=[-100, 100, -42.5, 42.5])

        if 'x' in event_coord and 'y' in event_coord:
            x = event_coord['x']
            y = event_coord['y']
            ax.plot(x, y, 'ro')
            ax.text(x + 2, y + 2, f'({x}, {y})', fontsize=12, color='red')  # Display coordinates next to the marker

        plt.show()

        game_info = game['liveData']['plays']['allPlays'][event_idx - 1] if event_idx > 0 else game['gameData']
        game_info_text = json.dumps(game_info, indent=2)
        self.game_info_text.value = game_info_text  # Update the game info text widget

    def display_widgets(self):
        clear_output(wait=True)
        display(widgets.VBox([self.year_dropdown, self.game_type_dropdown, self.game_slider, self.event_slider]))
        display(self.game_info_text)
        self.display_event(self.game_slider.value, self.event_slider.value)
        
    def browse_games(self):
        def display_data(change):
            self.display_widgets()
            
        self.game_slider.observe(display_data, 'value')
        self.event_slider.observe(display_data, 'value')
        self.game_type_dropdown.observe(display_data, 'value')
        self.display_widgets()

explorer = NHLExplorer(2016)  # Default to 2016, replace with your desired initial year
explorer.browse_games()