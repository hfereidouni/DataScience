import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

#for each game(only draft and test rn)
def json_reader(json_path:str) -> pd.DataFrame:
    # print(json_path)
    with open(json_path) as f:
        game_json = json.load(f)
        game_time = game_json["gameData"]["datetime"]["dateTime"]
        game_id = game_json["gamePk"]
        teams = game_json["gameData"]["teams"]
        team_home = teams["home"]
        team_away = teams["away"]

        plays = game_json["liveData"]["plays"]["allPlays"]

        # # gathering shots' and goals' indexes
        # this is useless, after testing i found that the json datas are not strictly
        # in the same order as "eventIdx"

        # goal_idxs = game_json["liveData"]["plays"]["scoringPlays"]
        # shot_idxs = []

        # for play in plays:
        #     if play["result"]["event"] == "Shot":
        #         shot_idxs.append(play["about"]["eventIdx"])
        
        # idxs = list(shot_idxs + goal_idxs)
        # idxs.sort()
        # print(idxs)

        rows = []
        # gathering shots' and goals' informations
        # for idx in idxs:
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

                item_row = [game_id,
                            play_idx,
                            play_type,
                            game_time,
                            team_home["name"],
                            team_away["name"],
                            goals_home,
                            goals_away,
                            attack_team_name,
                            period,
                            # period_time,
                            period_time_rem,
                            coordinate,shooter_name,
                            goalie_name,
                            empty_Net,strength]
                rows.append(item_row)

        df = pd.DataFrame(rows,columns=[["game_id",
                                        "event_idx",
                                        "play_type",
                                        "game_time",
                                        "team_home",
                                        "team_away",
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


def read_all_game(json_path:str)->pd.DataFrame:
    result = pd.DataFrame()
    res = os.listdir(json_path)
    length = len(res)
    print("Started...")
    for i in tqdm(range(length)):
            path = "./nhl_data/"+res[i]
            temp = json_reader(path)
            result = pd.concat([result,temp],ignore_index=True)

    print("Finieshed.")
    return result

# i = pd.DataFrame()
# res = os.listdir("./nhl_data/")
# count = 0
# for i in tqdm(range(len(res))):
#         path = "./nhl_data/"+res[i]
#         temp = json_reader(path)
#         i = pd.concat([i,temp],ignore_index=True)
#         count+=1
#         if count==1000:
#             break

# print(i)


# p = json_reader("./nhl_data/2016020001.json")
# q = json_reader("./nhl_data/2016020020.json")
# for 
# print(i.append(p))
# print(q)
print(read_all_game("./nhl_data/"))
    
