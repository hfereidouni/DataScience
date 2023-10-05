import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import re

#for each game(only draft and test rn)

#Rink_X = [-100,100]
#Rink_Y = [-42.5,42.5]
#Distance between nets and edges are 11 ft, 100 - 11 = 89
LEFT_NET_COOR = [-89,0]
RIGHT_NET_COOR = [89,0]
def json_reader(json_path:str) -> pd.DataFrame:
    with open(json_path) as f:
        game_json = json.load(f)
        game_time = game_json["gameData"]["datetime"]["dateTime"]
        game_id = game_json["gamePk"]
        # print(game_id)
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
                # print(play_idx)
                period = play["about"]["period"]
                period_type = play["about"]["periodType"]
                # period_time = play["about"]["periodTime"]
                goals_home = play["about"]["goals"]["home"]
                goals_away = play["about"]["goals"]["away"]
                period_time_rem = play["about"]["periodTimeRemaining"]
                attack_team = play["team"]
                attack_team_name = play["team"]["name"]
                play_type = play["result"]["event"]
                shot_type = None
                if "secondaryType" in play["result"]:
                    shot_type = play["result"]["secondaryType"]
                coordinate= play["coordinates"]
                # print(coordinate) 
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
                
                #calculate the distance between play coordinates and net(that should be attacked) coordinate
                shot_dist = None
                if coordinate != {}:
                    if "x" not in coordinate:
                        coordinate["x"] = 0
                    elif "y" not in coordinate:
                        coordinate["y"] = 0
                    
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
                    
                    if rink_side == "left":
                        shot_dist = np.linalg.norm(np.array([coordinate["x"],coordinate["y"]]) - np.array(RIGHT_NET_COOR))
                    elif rink_side == "right":
                        shot_dist = np.linalg.norm(np.array([coordinate["x"],coordinate["y"]]) - np.array(LEFT_NET_COOR))
                    else:
                        shot_dist = None
                    
                
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
                rows.append(item_row)

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
    if start_year>=2016 and start_year<=2020:
        res = os.listdir(path)
        result = pd.DataFrame()
        length = len(res)
        for i in tqdm(range(length)):
            if int(res[i][:4]) == start_year:
                path1 = path+res[i]
                temp = json_reader(path1)
                result = pd.concat([result,temp],ignore_index=True)
    else:
        print("Please choose a season between 2016 and 2020")
        return None
    return result

def read_seasons(path:str,start_season:int,end_season:int)->pd.DataFrame:
    years = list(range(start_season,end_season+1))
    result = pd.DataFrame()
    for year in years:
        temp = read_a_season(path,year)
        result = pd.concat([result,temp],ignore_index=True)
    return result


def read_all_game(path:str)->pd.DataFrame:
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

# read_all_game("./nhl_data/").to_csv('./tidy_test_1.csv', sep=',',index=False) 
# read_a_season("./nhl_data/",2016).to_csv('./tidy_{season}.csv'.format(season=2016), sep=',',index=False)
