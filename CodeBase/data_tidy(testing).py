import json
import os
import pandas

#for each game(only draft and test rn)
with open('./nhl_data/2016020001.json') as f:
    game_json = json.load(f)
    # game time/period information,game ID,team information (which team took the shot),shot/goal,coordinates,
    # the shooter name,the goalie name, shot type(if it was on an empty net, and whether or not a goal was at even strength, shorthanded, or on the power play).
    game_time = game_json["gameData"]["datetime"]
    game_id = game_json["gamePk"]

    plays = game_json["liveData"]["plays"]

    # gathering shots' and goals' indexes
    goal_idx = game_json["liveData"]["plays"]["scoringPlays"]
    shot_idxs = []
    for play in plays:
        if play["result"]["event"] == "Shot":
            shot_idxs.append(play["about"]["eventIdx"])

    # gathering shots' and goals' informations(I'm trying to use dataframe directly here)
    # shot
    for shot_idx in shot_idxs:
        pass
    
    #TODO
    #TODO
    #TODO
    #TODO
    #TODO
    #TODO#TODO
    #TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO
       

    
