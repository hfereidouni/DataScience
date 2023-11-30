import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import math
from scipy.spatial import distance


#Rink_X = [-100,100]
#Rink_Y = [-42.5,42.5]
#Distance between nets and edges are 11 ft, 100 - 11 = 89
LEFT_NET_COOR = [-89,0]
RIGHT_NET_COOR = [89,0]


def shot_angle(y, distance, rink_side):
    # Check if y, distance, or rink_side is None, return None if any are missing
    if y is None or distance is None or distance == 0 or rink_side is None or y == 0:
        return 0

    # Calculate the angle and handle exceptions if any
    try:
        if rink_side == 'right':
            angle = np.arcsin(y/distance) * -180/math.pi
        else:
            angle = np.arcsin(y/distance) * 180/math.pi
    except ValueError:
        # In case y/distance is out of domain for arcsin function
        return None
    
    return angle

def process_penalties(plays, team_home_name, team_away_name):
    penalties = {'home': [], 'away': []}
    penalty_start_times = {'home': [], 'away': []}
    penalty_expirations = {'home': [], 'away': []}

    for play in plays:
        if play["result"]["event"] == "Penalty":
            team = play["team"]["triCode"]
            team_name = play["team"]["name"]
            period = play["about"]["period"]
            period_time = play["about"]["periodTime"]
            penalty_minutes = play["result"]["penaltyMinutes"]

            penalty_start = (period - 1) * 20 + int(period_time.split(':')[0])
            expiry_time = penalty_start + penalty_minutes
            penalized_team = 'home' if team_name == team_away_name else 'away'

            penalties[penalized_team].append(penalty_minutes)
            penalty_start_times[penalized_team].append(penalty_start)
            penalty_expirations[penalized_team].append(expiry_time)

            # Debugging
            print(f"Penalty detected: Team {penalized_team}, at Game time {penalty_start} minutes till {expiry_time} time")

    return penalties, penalty_start_times, penalty_expirations

def update_powerplay_status(penalties, penalty_start_times, penalty_expirations, current_game_time):
    for team in ['home', 'away']:
        # wrt current time, which are active penalties
        # pen < start < expiry
        active_penalties, active_penalty_start, active_penalty_expirations = {'home': [], 'away': []}, {'home': [], 'away': []}, {'home': [], 'away': []}
        for penalty, penalty_start, expiration in zip(penalties[team], penalty_start_times[team], penalty_expirations[team]):
            if penalty_start < current_game_time and expiration > current_game_time:

                active_penalties[team].append(penalty)
                active_penalty_expirations[team].append(expiration)
                active_penalty_start[team].append(penalty_start)
                

    home_skaters = 5 - len(active_penalties['home'])
    away_skaters = 5 - len(active_penalties['away'])

    # at a given time, the team with more penalties is not the owerplay team
    if len(active_penalties['home']) != len(active_penalties['away']):
        powerplay_team = 'home' if len(active_penalties['home']) < len(active_penalties['away']) else 'away'
        non_powerplay_team = 'home' if powerplay_team=='away' else 'away'
        powerplay_time = 0
        if len(active_penalties[powerplay_team]) < len(active_penalties[non_powerplay_team]):
            #powerplay is current time - time when powerplay started for a team
            powerplay_time = current_game_time - active_penalty_start[non_powerplay_team][-1]
    else:
        # both teams have equal penalties - no powerplay
        # chatgt query: what happens if both teams are in penalty in nhl, who is the powerplay?
        powerplay_time = 0
        
    return home_skaters, away_skaters, powerplay_time

def calculate_game_time(period, period_time):
    minutes, seconds = map(int, period_time.split(':'))
    return (period - 1) * 20 * 60 + minutes * 60 + seconds

def json_reader(json_path:str) -> pd.DataFrame:

    """This function will take a path of a json filse as input and tidy data inside,
    it will return tidied data in a pandas dataframe.
    The features we have included here are:
        {"game_id", "event_idx","play_type","shot_type","shot_dist", "game_time", 
    "goals_home", "goals_away",  "attack_team_name", "period", "period_time_rem", 
    "coordinate", "shooter_name", "goalie_name", "empty_Net","strength"} which most 
    of them can be extracted directly from json file, only "shot_dist" which represents
    the distance from the shot coordinate to the net coordinate needs to be calculated
    with help of rink_side attributes and team information. For those empty information
    or attributes, we replace them with None.
    
    Parameters:
    json_path(str): Path for the json file of a game

    Returns:
    Pandas Dataframe:Tidied data

   """
    with open(json_path) as f:
        
        game_json = json.load(f)
        
        game_time = game_json["gameData"]["datetime"]["dateTime"]
        game_id = game_json["gamePk"]
        teams = game_json["gameData"]["teams"]
        team_home = teams["home"]
        team_away = teams["away"]
        plays = game_json["liveData"]["plays"]["allPlays"]

        # Initialize penalty tracking
        team_home_name = game_json["gameData"]["teams"]["home"]["name"]
        team_away_name = game_json["gameData"]["teams"]["away"]["name"]
        penalties, penalty_start_times, penalty_expirations = process_penalties(plays, team_home_name, team_away_name)

        rows = []
        # gathering shots' and goals' informations
        for i,play in enumerate(plays):
            item_row = []
            if play["result"]["event"] == "Shot" or play["result"]["event"]=="Goal":

                play_idx = play["about"]["eventIdx"]
                period = play["about"]["period"]
                period_type = play["about"]["periodType"]
                period_time_rem = play["about"]["periodTimeRemaining"]
                goals_home = play["about"]["goals"]["home"]
                goals_away = play["about"]["goals"]["away"]
                attack_team = play["team"]
                attack_team_name = play["team"]["name"]
                play_type = play["result"]["event"]
                shot_type = None
                period_time = play["about"]["periodTime"]
                current_game_time = calculate_game_time(period, period_time)
                
                home_skaters, away_skaters, powerplay_time = update_powerplay_status(penalties, penalty_start_times, penalty_expirations, current_game_time)

                if "secondaryType" in play["result"]:
                    shot_type = play["result"]["secondaryType"]
                coordinate= play["coordinates"]
                players = play["players"]
                goalie_name = None
                shooter_name = None
                for player in players:
                    if player["playerType"]== "Goalie":
                        goalie_name = player["player"]["fullName"]
                    elif player["playerType"]== "Shooter" or player["playerType"]== "Scorer":
                        shooter_name = player["player"]["fullName"]
                empty_Net=None
                strength=None
                if play_type == "Goal":
                    if "emptyNet" in play["result"]:
                        empty_Net = play["result"]["emptyNet"]
                    strength = play["result"]["strength"]["code"]
                
                #find the rinkside that each team should attack/defend
                #and calculate the distance between play coordinates and net(attack rinkside) coordinate
                shot_dist = None
                if coordinate != {}:
                    if "y" not in coordinate:
                        coordinate["y"] = 0
                    elif "x" not in coordinate:
                        coordinate["x"] = 0
                    
                    #find rinkside
                    periods = game_json["liveData"]["linescore"]["periods"]
                    rink_side = None
                    if attack_team_name == team_home["name"]:
                        home_or_away = "home"
                        for p in periods:
                            if p["num"] == period:
                                if "rinkSide" in p[home_or_away]:
                                    rink_side = p[home_or_away]["rinkSide"]
                    else:
                        home_or_away = "away"
                        for p in periods:
                            if p["num"] == period:
                                if "rinkSide" in p[home_or_away]:
                                    rink_side = p[home_or_away]["rinkSide"]
                    
                    #calculate distance
                    if rink_side == "left":
                        shot_dist = np.linalg.norm(np.array([coordinate["x"],coordinate["y"]]) - np.array(RIGHT_NET_COOR))
                    elif rink_side == "right":
                        shot_dist = np.linalg.norm(np.array([coordinate["x"],coordinate["y"]]) - np.array(LEFT_NET_COOR))
                    else:
                        shot_dist = None
                
                            #data engineering 2:
                if i>0:
                    previous_event = plays[i-1]
                    last_event_type = previous_event['result']['event']

                    x = play['coordinates']['x'] if 'x' in play['coordinates'] else None
                    y = play['coordinates']['y'] if 'y' in play['coordinates'] else None

                    x_coord_last_event = previous_event['coordinates']['x'] if 'x' in previous_event['coordinates'] else None
                    y_coord_last_event = previous_event['coordinates']['y'] if 'y' in previous_event['coordinates'] else None

                    # time from the last event (seconds)
                    about = play['about']
                    period = about['period']
                    period_time = about['periodTime']
                    minutes_time = (period - 1) * 20
                    seconds_time = int(period_time.split(':')[0]) * 60 + int(period_time.split(':')[1])
                    game_time = minutes_time + seconds_time

                    prev_about = previous_event['about']
                    prev_period = previous_event['about']['period']
                    prev_period_time = prev_about['periodTime']
                    prev_minutes_time = (prev_period - 1) * 20
                    prev_seconds_time = int(prev_period_time.split(':')[0]) * 60 + int(prev_period_time.split(':')[1])
                    prev_gameTime = prev_minutes_time + prev_seconds_time

                    Time_from_the_last_event = game_time - prev_gameTime
                    # Distance from the last event
                    if all(coord is not None for coord in [x,y, x_coord_last_event, y_coord_last_event]):
                        Distance_from_the_last_event = distance.euclidean([x, y], [x_coord_last_event, y_coord_last_event])
                    else:
                        Distance_from_the_last_event = None

                    # Rebound
                    rebound = True if last_event_type == 'Shot' else False 

                    # change in shot angle
                    change_shot_angle=None
                    if rebound == True:
                        last_shot_dist = None
                        if x!=None and y!= None and x_coord_last_event!=None and y_coord_last_event!=None:
                            if rink_side == "left":
                                last_shot_dist = np.linalg.norm(np.array([x_coord_last_event,y_coord_last_event]) - np.array(RIGHT_NET_COOR))
                            elif rink_side == "right":
                                last_shot_dist = np.linalg.norm(np.array([x_coord_last_event,y_coord_last_event]) - np.array(LEFT_NET_COOR))
                            else:
                                last_shot_dist = None

                            last_shot_angle= None
                            if last_shot_dist != None:
                                last_shot_angle = shot_angle(y_coord_last_event,last_shot_dist,rink_side)
                            
                            if last_shot_angle!=None:
                                change_shot_angle= shot_angle(y, shot_dist, rink_side)-last_shot_angle


                    # Speed
                    speed = Distance_from_the_last_event / Time_from_the_last_event if Distance_from_the_last_event is not None and Time_from_the_last_event != 0 else None

                #list for each event/play
                item_row = [game_id,
                            play_idx,
                            play_type,
                            shot_type,
                            shot_dist,
                            game_time,
                            goals_home,
                            goals_away,
                            attack_team_name,
                            period,
                            period_time_rem,
                            coordinate,shooter_name,
                            goalie_name,
                            empty_Net,strength,
                            rink_side,last_event_type, 
                            x_coord_last_event,
                            y_coord_last_event,
                            Time_from_the_last_event,
                            Distance_from_the_last_event,
                            rebound,
                            change_shot_angle,
                            speed,x,y,
                            home_skaters, away_skaters, powerplay_time]


                #total list for all events/games
                rows.append(item_row)
        
        #list->pd.DataFreame
        df = pd.DataFrame(rows,columns=["game_id",
                                        "event_idx",
                                        "play_type",
                                        "shot_type",
                                        "shot_dist",
                                        "game_time",
                                        "goals_home",
                                        "goals_away",
                                        "attack_team_name",
                                        "period",
                                        "period_time_rem",
                                        "coordinate",
                                        "shooter_name",
                                        "goalie_name",
                                        "empty_Net",
                                        "strength",
                                        "rink_side",
                                        'last_event_type',
                                        'x_coord_last_event',
                                        'y_coord_last_event',
                                        'Time_from_the_last_event',
                                        'Distance_from_the_last_event',
                                        'Rebound',
                                        'change_shot_angle',
                                        'Speed','x','y',
                                        "HomeSkaters", "AwaySkaters", "PowerplayTime"])
        
        df['angle_net'] = df.apply(lambda row: shot_angle(row['coordinate'].get('y', 0), row['shot_dist'], row['rink_side']) if isinstance(row['coordinate'], dict) else 0, axis=1)
        df['is_goal'] = np.where(df['play_type'] == 'Goal', 1, 0)
        
    #print(df)
    return df

def read_a_season(path:str,start_year:int)->pd.DataFrame:
    """
    Parameters:
    path(str): Path for the json file of a game
    start_year(int): Start year of a selected season eg.2016 for season 2016-2017

    Returns:
    Pandas Dataframe:Tidied data for one particular season

   """
    if start_year>=2016 and start_year<=2020:
        res = os.listdir(path)
        result = pd.DataFrame()
        length = len(res)
        for i in tqdm(range(length)):
            if int(res[i][:4]) == start_year:
                path1 = path+res[i]
                temp = json_reader(path1)
                if result.empty:
                    # print(result)
                    result = temp
                else:
                    # pandas 2.1.1 has null error
                    # fix: https://stackoverflow.com/questions/77254777/alternative-to-concat-of-empty-dataframe-now-that-it-is-being-deprecated
                    result = pd.concat([result.astype(temp.dtypes),temp.astype(result.dtypes)],ignore_index=True)
    else:
        print("Please choose a season between 2016 and 2020")
        return None
    return result

def read_seasons(path:str,start_season:int,end_season:int)->pd.DataFrame:
    """This function will read multiple seasons from start_season to end_season

    Parameters:
    path(str): Path for the json file of a game
    start_season(int): Start year or start season
    end_season(int): Start year or end season

    Returns:
    Pandas Dataframe:Tidied data for selected seasons

   """
    years = list(range(start_season,end_season+1))
    result = pd.DataFrame()
    for year in years:
        temp = read_a_season(path,year)
        result = pd.concat([result,temp],ignore_index=True)
    return result


def read_all_game(path:str)->pd.DataFrame:
    """This function will read all game json files in path

    Parameters:
    path(str): Path for the json file of a game json directory

    Returns:
    Pandas Dataframe:Tidied data for all games

   """
    result = pd.DataFrame()
    res = os.listdir(path)
    length = len(res)
    print("Started...")
    for i in tqdm(range(length)):
            path1 = path+res[i]
            temp = json_reader(path1)
            result = pd.concat([result,temp],ignore_index=True)

    print("Finieshed.")
    return result
