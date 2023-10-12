import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import re

#Rink_X = [-100,100]
#Rink_Y = [-42.5,42.5]
#Distance between nets and edges are 11 ft, 100 - 11 = 89
LEFT_NET_COOR = [-89,0]
RIGHT_NET_COOR = [89,0]
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

        rows = []
        # gathering shots' and goals' informations
        for play in plays:
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
                    if "x" not in coordinate:
                        coordinate["x"] = 0
                    elif "y" not in coordinate:
                        coordinate["y"] = 0
                    
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
                            empty_Net,strength]
                
                #total list for all events/games
                rows.append(item_row)
        
        #list->pd.DataFreame
        df = pd.DataFrame(rows,columns=[["game_id",
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
                                        "strength"]])
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