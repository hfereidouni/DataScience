import pandas as pd
import json
import os
import numpy as np
from tqdm import tqdm
import requests
import re
import math
from scipy.spatial import distance
# from Tidy_Data_new import *

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

def load_game(game_id):
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
    response = requests.get(url)
    data = json.loads(response.text)
    
    return data

def create_game_df(game_data):
    game_df = json_reader_from_json_object(game_data)

    return game_df

def json_reader_from_json_object(game_json):

    game_time = game_json["startTimeUTC"]
    game_id = game_json["id"]
    team_home = game_json["awayTeam"]
    team_away = game_json["homeTeam"]
    home_name = team_home["name"]["default"]
    away_name = team_away["name"]["default"]
    plays = game_json["plays"]

    home_score = 0
    away_score = 0

    home_id = team_home["id"]
    away_id = team_away["id"]

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
            
            play_owner_ID = play["details"]["eventOwnerTeamId"]
            home_or_away = 'home' if play_owner_ID==home_id else 'away'

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
                        home_or_away,
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
                                    "home_or_away",
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
        
    return df


if __name__=="__main__":
    print("load game")
    game_id = "2022030411"
    game_df = load_game(game_id)
    print(game_df.keys())
    # print(game_df.head())
    print("===================================")

    game_df = create_game_df(game_df)
    print("shape df: ", game_df.shape)
    print("columns: ", game_df.columns)
    print(game_df.tail())

    print(game_df.iloc[-1]['event_idx'])
    

