import json
import os
import pandas

#for each game(only draft and test rn)
with open('./nhl_data/2016020001.json') as f:
    game_json = json.load(f)
    # game time/period information,game ID,team information (which team took the shot),shot/goal,coordinates,
    # the shooter name,the goalie name, shot type(if it was on an empty net), and whether or not a goal was at even strength, shorthanded, or on the power play.
    game_time = game_json["gameData"]["datetime"]
    game_id = game_json["gamePk"]

    plays = game_json["liveData"]["plays"]["allPlays"]

    # gathering shots' and goals' indexes
    goal_idx = game_json["liveData"]["plays"]["scoringPlays"]
    shot_idxs = []

    for play in plays:
        if play["result"]["event"] == "Shot":
            shot_idxs.append(play["about"]["eventIdx"])

    # gathering shots' and goals' informations(I'm trying to use dataframe directly here)
    # shot
    for shot_idx in shot_idxs:
        shot = plays[shot_idx]
        attack_team = shot["team"]
        attack_team_id = shot["team"]["id"]
        attack_team_name = shot["team"]["name"]
        attack_team_triCode = shot["team"]["triCode"]
        play_type = "Shot"
        coordinate= (shot["coordinates"]["x"],shot["coordinates"]["y"])
        players = shot["players"]
        for player in players:
            if player["playerType"]== "Goalie":
                goalie = player
                goalie_name = player["fullName"]
            elif player["playerType"]== "Shooter":
                shooter = player
                shooter_name = player["fullName"]
       
               # ...
        empty_Net = None
        strength = None
    
    #goal
    for shot_idx in shot_idxs:
        shot = plays[shot_idx]
        attack_team = shot["team"]
        attack_team_id = shot["team"]["id"]
        attack_team_name = shot["team"]["name"]
        attack_team_triCode = shot["team"]["triCode"]
        play_type = "Shot"
        coordinate= (shot["coordinates"]["x"],shot["coordinates"]["y"])
        players = shot["players"]
        for player in players:
            if player["playerType"]== "Goalie":
                goalie = player
                goalie_name = player["fullName"]
            elif player["playerType"]== "Shooter":
                shooter = player
                shooter_name = player["fullName"]
       
               # ...
        empty_Net = None
        strength = None

    
    #TODO
    #TODO
    #TODO
    #TODO
    #TODO
    #TODO#TODO
    #TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO
       

    
