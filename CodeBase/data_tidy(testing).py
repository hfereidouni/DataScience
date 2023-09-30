import json
import os
import pandas as pd
import numpy as np

#for each game(only draft and test rn)
def json_reader(json_path:str) -> pd.DataFrame:
    with open(json_path) as f:
        game_json = json.load(f)
        # game time/period information,game ID,team information (which team took the shot),shot/goal,coordinates,
        # the shooter name,the goalie name, shot type(if it was on an empty net), and whether or not a goal was at even strength, shorthanded, or on the power play.
        game_time = game_json["gameData"]["datetime"]
        game_id = game_json["gamePk"]
        teams = game_json["gameData"]["teams"]
        team_home = teams["home"]
        team_away = teams["away"]

        plays = game_json["liveData"]["plays"]["allPlays"]

        # gathering shots' and goals' indexes
        goal_idxs = game_json["liveData"]["plays"]["scoringPlays"]
        shot_idxs = []

        for play in plays:
            if play["result"]["event"] == "Shot":
                shot_idxs.append(play["about"]["eventIdx"])

        rows = []
        # gathering shots' and goals' informations
        # shot
        for shot_idx in shot_idxs:
            item_row = []
            shot = plays[shot_idx]
            shot_period = shot["about"]["period"]
            shot_period_type = shot["about"]["periodType"]
            shot_period_time = shot["about"]["periodTime"]
            shot_period_time_rem = shot["about"]["periodTimeRemaining"]
            attack_team = shot["team"]
            attack_team_id = shot["team"]["id"]
            attack_team_name = shot["team"]["name"]
            attack_team_triCode = shot["team"]["triCode"]
            play_type = "Shot"
            coordinate= (shot["coordinates"]["x"],shot["coordinates"]["y"])
            players = shot["players"]
            for player in players:
                if player["playerType"]== "Goalie":
                    goalie = player["player"]
                    goalie_name = player["player"]["fullName"]
                elif player["playerType"]== "Shooter":
                    shooter = player["player"]
                    shooter_name = player["player"]["fullName"]
            empty_Net = None
            strength = None
            item_row = [play_type,game_id,game_time,team_home["name"],team_away["name"],
                        shot_period,shot_period_time,shot_period_time_rem,
                        attack_team_name,coordinate,shooter_name,goalie_name,
                        empty_Net,strength]

            rows.append(item_row)
        
        #goal
        for goal_idx in goal_idxs:
            item_row=[]
            goal = plays[goal_idx]
            goal_period = shot["about"]["period"]
            goal_period_type = shot["about"]["periodType"]
            goal_period_time = shot["about"]["periodTime"]
            goal_period_time_rem =shot["about"]["periodTimeRemaining"]
            attack_team = goal["team"]
            attack_team_id = goal["team"]["id"]
            attack_team_name = goal["team"]["name"]
            attack_team_triCode = goal["team"]["triCode"]
            play_type = "Goal"
            coordinate= (goal["coordinates"]["x"],goal["coordinates"]["y"])
            players = goal["players"]
            for player in players:
                if player["playerType"]== "Goalie":
                    goalie = player["player"]
                    goalie_name = player["player"]["fullName"]
                elif player["playerType"]== "Scorer":
                    shooter = player["player"]
                    shooter_name = player["player"]["fullName"]
            empty_Net = goal["result"]["emptyNet"]
            strength = goal["result"]["strength"]["code"]
            item_row = [play_type,game_id,game_time,team_home["name"],team_away["name"],
                        goal_period,goal_period_time,goal_period_time_rem,
                        attack_team_name,coordinate,shooter_name,goalie_name,
                        empty_Net,strength]

            
            rows.append(item_row)
            df = pd.DataFrame(rows,columns=[["play_type","game_id","game_time","team_home","team_away",
                        "goal_period","goal_period_time","goal_period_time_rem",
                        "attack_team_name","coordinate","shooter_name","goalie_name",
                        "empty_Net","strength"]])
    return df
p = json_reader("./nhl_data/2016020001.json")
print(p)

    
