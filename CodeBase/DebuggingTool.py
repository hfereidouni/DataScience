import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json

class NHLExplorer:
    def __init__(self, season):
        self.season = season
        self.games = []
        for game_id in self.generate_game_ids():
            try:
                with open(f"nhl_data/{game_id}.json", 'r') as file:
                    self.games.append(json.load(file))
            except FileNotFoundError:
                pass

    def generate_game_ids(self):
       game_ids = []
       
       # Determine the number of games based on teams in the season
       if self.season == 2016:
           total_games = 1230
       elif 2016 < self.season <= 2020:
           total_games = 1271
       else:
           total_games = 1353

       for game_num in range(1, total_games + 1):
           game_ids.append(f"{self.season}02{game_num:04d}")
       '''
       # Playoff game IDs
       matchups_per_round = [8, 4, 2, 1]
       for round_num in range(1, 5):
           for matchup_num in range(1, matchups_per_round[round_num - 1] + 1):
               for game_num in range(1, 8):
                   game_ids.append(f"{self.season}03{round_num}{matchup_num}{game_num}")
       '''
       return game_ids

    def display_event(self, game_idx, event_idx):
        game = self.games[game_idx]
        event = game['liveData']['plays']['allPlays'][event_idx]
        event_coord = event.get('coordinates', {})

        img = mpimg.imread('ice_rink_image.png')
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(img, extent=[-100, 100, -42.5, 42.5])

        if 'x' in event_coord and 'y' in event_coord:
            ax.plot(event_coord['x'], event_coord['y'], 'ro')
        plt.show()

        print(event.get('result', {}).get('description', 'No event description available.'))

    def browse_games(self):
        game_slider = widgets.IntSlider(min=0, max=len(self.games)-1, description='Game:')
        event_slider = widgets.IntSlider(min=0, max=len(self.games[0]['liveData']['plays']['allPlays'])-1, description='Event:')
        
        def update_event_range(*args):
            event_slider.max = len(self.games[game_slider.value]['liveData']['plays']['allPlays']) - 1
        
        game_slider.observe(update_event_range, 'value')
        
        def display_data(change):
            clear_output(wait=True)
            display(widgets.VBox([game_slider, event_slider]))
            self.display_event(game_slider.value, event_slider.value)
        
        game_slider.observe(display_data, 'value')
        event_slider.observe(display_data, 'value')
        
        display(widgets.VBox([game_slider, event_slider]))

explorer = NHLExplorer(2016)  # Replace with your desired season
explorer.browse_games()