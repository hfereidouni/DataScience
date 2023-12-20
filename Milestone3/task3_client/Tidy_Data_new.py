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
        
        game_time = game_json["startTimeUTC"]
        game_id = game_json["id"]
        team_home = game_json["awayTeam"]
        team_away = game_json["homeTeam"]
        home_name = team_home["name"]["default"]
        away_name = team_away["name"]["default"]
        plays = game_json["plays"]

        home_score = 0
        away_score = 0

        rows = []
        # gathering shots' and goals' informations
        for i,play in enumerate(plays):
            item_row = []
            if play["typeDescKey"] == "shot-on-goal" or play["typeDescKey"] == "missed-shot" or play["typeDescKey"] == "goal":

                play_idx = play["eventId"]

                period = play["period"]
                period_time_rem = play["timeRemaining"]
                play_type = None
                if play["typeDescKey"] == "shot-on-goal" or play["typeDescKey"] == "missed-shot":
                    play_type = "Shot"
                elif play["typeDescKey"] == "goal":
                    play_type = "Goal"
                    home_score = play["details"]["homeScore"]
                    away_score = play["details"]["awayScore"]

                shot_type = None
                shot_type_new = None

                if "shotType" in play["details"]:
                    shot_type_new = play["details"]["shotType"]

                #Convert shotTypes in new API to the same as the old API
                if shot_type_new == "snap" or shot_type_new == "slap" or shot_type_new == "wrist":
                    shot_type = shot_type_new.capitalize()+" Shot"
                elif shot_type_new == "backhand" or shot_type_new == "tip-in":
                    shot_type = shot_type_new.title()
                elif shot_type_new == "wrap-around":
                    shot_type = shot_type_new.capitalize()
                
                # coordinate= {"x":play["details"]["xCoord"],"y":play["details"]["yCoord"]}
                coordinate= {}
                if "xCoord" in play["details"]:
                    coordinate["x"]= play["details"]["xCoord"]
                if "yCoord" in play["details"]:                
                    coordinate["y"]= play["details"]["yCoord"]

                # players = play["players"]
                goalie_id = None
                shooter_id = None

                if play_type == "Shot":
                    shooter_id = play["details"]["shootingPlayerId"]
                    
                elif play_type == "Goal":
                    shooter_id = play["details"]["scoringPlayerId"]
                
                if "goalieInNetId" in play["details"]:
                    goalie_id = play["details"]["goalieInNetId"]

                situation_code = play["situationCode"]

                empty_Net=None

                if situation_code == None:
                    empty_Net = 1
                else:
                    if str(situation_code)[0]=='0' or str(situation_code)[-1]=='0':
                        empty_Net = 1
                    else:
                        empty_Net = 0

                shot_dist = None
                if coordinate != {}:
                    if "y" not in coordinate:
                        coordinate["y"] = 0
                    elif "x" not in coordinate:
                        coordinate["x"] = 0
                    
                    
                    home_defend_rinkside = play["homeTeamDefendingSide"]
                    rink_side = None
                    if home_defend_rinkside != None:
                        if home_defend_rinkside=="right":
                            attack_team_id = play["details"]["eventOwnerTeamId"]
                            if attack_team_id == team_home["id"]:
                                rink_side = "left"
                            else:
                                rink_side = "right"
                        elif home_defend_rinkside=="left":
                            attack_team_id = play["details"]["eventOwnerTeamId"]
                            if attack_team_id == team_home["id"]:
                                rink_side = "right"
                            else:
                                rink_side = "left"
                    
                    #calculate distance
                    if rink_side == "left":
                        shot_dist = np.linalg.norm(np.array([coordinate["x"],coordinate["y"]]) - np.array(RIGHT_NET_COOR))
                    elif rink_side == "right":
                        shot_dist = np.linalg.norm(np.array([coordinate["x"],coordinate["y"]]) - np.array(LEFT_NET_COOR))
                    else:
                        shot_dist = None
            

                #list for each event/play
                item_row = [game_id,
                            home_name,
                            away_name,
                            home_score,
                            away_score,
                            play_idx,
                            play_type,
                            shot_type,
                            shot_dist,
                            game_time,
                            period,
                            period_time_rem,
                            coordinate,
                            shooter_id,
                            goalie_id,
                            empty_Net,
                            rink_side,
                            ]


                #total list for all events/games
                rows.append(item_row)
        
        #list->pd.DataFreame
        df = pd.DataFrame(rows,columns=["game_id",
                                        "home_name",
                                        "away_name",
                                        "home_score",
                                        "away_score",
                                        "event_idx",
                                        "play_type",
                                        "shot_type",
                                        "shot_dist",
                                        "game_time",
                                        "period",
                                        "period_time_rem",
                                        "coordinate",
                                        "shooter_id",
                                        "goalie_id",
                                        "empty_Net",
                                        "rink_side",
                                        ])
        
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
    if start_year>=2020 and start_year<=2022:
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
                    result = pd.concat([result.astype(temp.dtypes,errors='ignore'),temp.astype(result.dtypes,errors='ignore')],ignore_index=True)
    else:
        print("Please choose a season between 2020 and 2022")
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
