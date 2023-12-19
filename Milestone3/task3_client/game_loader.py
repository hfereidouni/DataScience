import pandas as pd
import json
import os
import numpy as np
from tqdm import tqdm
import requests
from Tidy_Data_new import *



def load_game(game_id):
    url = f"https://api-web.nhle.com/ai/v1/game/{game_id}/feed/live"
    response = requests.get(url)
    data = json.loads(response.text)
    
    return game_df

def create_game_df(game_data):
    game_df = json_reader_from_json_object(game_data)

    return game_df

def json_reader_from_json_object(game_json):

    game_time = game_json["startTimeUTC"]
    game_id = game_json["id"]
    team_home = game_json["awayTeam"]
    team_away = game_json["homeTeam"]
    plays = game_json["plays"]

    # print(game_id)

    rows = []
    # gathering shots' and goals' informations
    for i,play in enumerate(plays):
        item_row = []
        # if play["result"]["event"] == "Shot" or play["result"]["event"]=="Goal":
        if play["typeDescKey"] == "shot-on-goal" or play["typeDescKey"] == "missed-shot" or play["typeDescKey"] == "goal":

            play_idx = play["eventId"]

            # print(play_idx)

            period = play["period"]
            period_time_rem = play["timeRemaining"]
            # goals_home = play["details"]["homeScore"]
            # goals_away = play["details"]["awayScore"]
            # attack_team_name = play["team"]["name"]
            play_type = None
            if play["typeDescKey"] == "shot-on-goal" or play["typeDescKey"] == "missed-shot":
                play_type = "Shot"
            elif play["typeDescKey"] == "goal":
                play_type = "Goal"

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

            #find the rinkside that each team should attack/defend
            #and calculate the distance between play coordinates and net(attack rinkside) coordinate
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
                        play_idx,
                        play_type,
                        shot_type,
                        shot_dist,
                        game_time,
                        # goals_home,
                        # goals_away,
                        # attack_team_name,
                        period,
                        period_time_rem,
                        coordinate,
                        shooter_id,
                        goalie_id,
                        empty_Net,
                        # strength,
                        rink_side,
                        ]


            #total list for all events/games
            rows.append(item_row)
        
    #list->pd.DataFreame
    df = pd.DataFrame(rows,columns=["game_id",
                                    "event_idx",
                                    "play_type",
                                    "shot_type",
                                    "shot_dist",
                                    "game_time",
                                    # "goals_home",
                                    # "goals_away",
                                    # "attack_team_name",
                                    "period",
                                    "period_time_rem",
                                    "coordinate",
                                    "shooter_id",
                                    "goalie_id",
                                    "empty_Net",
                                    # "strength",
                                    "rink_side",
                                    # 'last_event_type',
                                    # 'x_coord_last_event',
                                    # 'y_coord_last_event',
                                    # 'Time_from_the_last_event',
                                    # 'Distance_from_the_last_event',
                                    # 'Rebound',
                                    # 'change_shot_angle',
                                    # 'Speed',
                                    # 'x',
                                    # 'y'
                                    ])
        
    df['angle_net'] = df.apply(lambda row: shot_angle(row['coordinate'].get('y', 0), row['shot_dist'], row['rink_side']) if isinstance(row['coordinate'], dict) else 0, axis=1)
    df['is_goal'] = np.where(df['play_type'] == 'Goal', 1, 0)
        
    #print(df)
    return df



