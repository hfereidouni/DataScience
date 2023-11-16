# Part 4: Feature Engineering II

## Questions 1 to 4

**Features selected**:
`['game_time','period','x','y','shot_type','last_event_type',
    'x_coord_last_event', 'y_coord_last_event', 'Time_from_the_last_event',
    'Distance_from_the_last_event', 'Rebound', 'change_shot_angle', 'Speed',
    'shot_dist','angle_net', 'is_goal']`
    
### Question 1:
- **game_time**: The time (seconds) where this event happened in current game.
- **period**: The period that in which event happened in current game.
- **x**, **y**: The `x` and `y` coordinate of where this event happened in current game.
- **shot_type**: Type of shot of the this event.
- **shot_dist**: The distance of this event to the net of the attack rinkside.
- **angle_net**: The angle of this event to the net of the attack rinkside, calculated using `shot_dist`.

### Question 2:
- **last_event_type**: Type of event (play) of the previous event (of this event) in current game.
- **x_coord_last_event**, **y_coord_last_event**: The x and y coordinate of where the previous event happened in current game.
- **Time_from_the_last_event**: The time (seconds) where the previous event happened in current game.
- **Distance_from_the_last_event**: the distance between the coordinates of this event and the coordinates of the previous event.

### Question 3:
- **Rebound**: A boolean value of whether this event is a rebound (the previous event was also a shot) or not.
- **change_shot_angle**: The difference of angle of the shot of this event and that of the previous event, this is only a valid value if this event is a `rebound`.
- **Speed**: The average speed between the previous event and this event, calculated using `Distance_from_the_last_event` divided by `Time_from_the_last_event`.

Other:
- **is_goal**: A binary Label that if this event is a goal: 1 if it's a goal otherwise 0.

### Question 4:

**NOTE:** To check this part, please check the `FeatureEngineering2_Q4_Bonus.py`, `FeatureEngineering2_Q4_Bonus.ipynb` and `AdvancedModel_helpers2.py`


**Function: process_penalties(plays):**
```python
def process_penalties(plays):
    penalties = {'home': [], 'away': []}
    penalty_expirations = {'home': [], 'away': []}

    for play in plays:
        if play["result"]["event"] == "Penalty":
            team = play["team"]["triCode"]
            period = play["about"]["period"]
            period_time = play["about"]["periodTime"]
            penalty_minutes = play["result"]["penaltyMinutes"]

            game_time = (period - 1) * 20 + int(period_time.split(':')[0]) + penalty_minutes
            penalized_team = 'home' if team == 'away' else 'away'

            penalties[penalized_team].append(penalty_minutes)
            penalty_expirations[penalized_team].append(game_time)

            # Debugging
            print(f"Penalty detected: Team {penalized_team}, Game time {game_time} minutes")

    return penalties, penalty_expirations
```
This function analyzes a list of plays from a game, identifying and handling penalty events. It calculates when each penalty will expire based on the game's progress and logs these details. The function is primarily focused on tracking penalties for both the home and away teams, including the duration and expiration times of these penalties.

**Function: update_powerplay_status(penalties, penalty_expirations, current_game_time):**
```python
def update_powerplay_status(penalties, penalty_expirations, current_game_time):
    for team in ['home', 'away']:
        penalties[team] = [penalty for penalty, expiration in zip(penalties[team], penalty_expirations[team]) if expiration > current_game_time]
        penalty_expirations[team] = [expiration for expiration in penalty_expirations[team] if expiration > current_game_time]

    home_skaters = 5 - len(penalties['home'])
    away_skaters = 5 - len(penalties['away'])

    powerplay_team = 'home' if len(penalties['home']) < len(penalties['away']) else 'away'
    powerplay_time = 0
    if len(penalties[powerplay_team]) < len(penalties['home' if powerplay_team == 'away' else 'away']):
        powerplay_time = min(penalty_expirations[powerplay_team]) if penalty_expirations[powerplay_team] else 0

    return home_skaters, away_skaters, powerplay_time
```
update_powerplay_status is designed to update the current status of penalties and powerplays in a game based on the ongoing time. It removes expired penalties and calculates the number of players each team has on the ice. The function also identifies which team, if any, has a powerplay advantage and the remaining duration of that powerplay.

**Function: calculate_game_time(period, period_time):**
```python
def calculate_game_time(period, period_time):
    minutes, seconds = map(int, period_time.split(':'))
    return (period - 1) * 20 * 60 + minutes * 60 + seconds
```
This utility function converts the period and time within a period of a game into total game time in seconds. It's a straightforward calculation used to standardize time references in the game, allowing for easier comparison and assessment of time-dependent elements like penalties or play durations.

The list of features:

- **HomeSkaters**: The number of non-goalie skaters currently on the ice for the home team.
- **AwaySkaters**: The number of non-goalie skaters currently on the ice for the away team.
- **PowerplayTime**: The amount of time elapsed since the start of the current power-play, resetting to 0 when the power-play ends.


### Overview of filtered Dataframe
![df](./images/part%204/dataframe.jpg)


## Question 5

**The filtered DataFrame artifact for the specified game (wpg_v_wsh_2017021065):**
[Comet.ml](https://www.comet.com/hfereidouni/ift6758/41a02783a9d54633a47fd78acfb9a900?experiment-tab=panels&showOutliers=true&smoothing=0&xAxis=step)
