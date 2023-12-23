---
layout: page
title: Tidy Data and Simple Visulization
permalink: /milestone1/
---
## Requirements for Tidy data and Simple visulization part
[requirements.txt](../CodeBase/requirements.txt)

## 4. Tidy Data
### Question 1. Small snippet of final dataframe of season 2016-2017
![title](../Images/tidy%20data%20snippet.png/)
### Question 2.  
1. At the beginning of the game, set the strength to *even*
2. If a penalty happened in an event, save the *penalty_minuite* X as an integer, *pen_team_ID* as string.
3. For plays in the next X minuite, if the ID of the team who has done the shot is the same as *pen_ID*, then the *strength* attribute will be set as "short handed", otherwise the *strength* will be set as "power play". 
4. - If the play was only a *shot*(not goal), for the next play we will do the same thing as 3) until X minuites later or next goal. 
    - If the play was a goal, we also do the same things as 3), but then the plays afterwards will have a strength of *even* if no other penalties happen.
### Question 3.
#### Additional features
1. Rebound: if another shot from a player from the same team happens within 3 seconds before a shot/goal, this can be considered as a **rebound**.
2. Shot off the rush: if a giveaway is happened 15 seconds before a shot, this shot can be considered as a **Shot off the rush**.
3. Shot distance: first to determinate which **rinkside** the shot team defend, then calculate the distances using the shot coordinates and the coordinates of the net on the opposite rinkside.

## 5. Simple visulization
### Question 1
![title](../Images/q1_type_goal.png/)
### Question 2
#### How did we determinate Shot Distance ?
- In our tidy_data.py, we have included an additional column which is shot distance. Then with this attribute, we can plot directly in simple visulization part.
```python
#Rink_X = [-100,100]
#Rink_Y = [-42.5,42.5]
#Distance between nets and edges are 11 ft, 100 - 11 = 89
LEFT_NET_COOR = [-89,0]
RIGHT_NET_COOR = [89,0]

[......]
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

```

![2018-2019_1](../Images/q2_dist_goal_2018.png)
![2019-2020_1](../Images/q2_dist_goal_2019.png)
![2018-2021_1](../Images/q2_dist_goal_2020.png)
####  Has there been much change over the past three seasons? 
- One significant change could be the chance of goal at long distances has decreased, in long distance, the high chance area has moved from [180,195] to [135,150].
####  Why did you choose this figure?
- Because the chance of goal in each area can well represent the relationship of chance of goal and distance if the gap of each interval is small.
- Bar plot can clearly show how different the chance of goals are for each interval of ditance.
### Question 3
![heat-map](../Images/q2_dist_type_goal.png)
#### Briefly discuss your findings
- We have used a heatmap to show the goal percentage as a function of both distance from the net, and the category of shot types， the y axis represents different range of shot distance, x axis shows different type of shots, each grid shows the chance of goals as function of a distance and a shot type(ex. if a Wrist Shot happens in a distance between 75 and 100, the chance of goal is 4.6%). 
- The brighter the color of a grid is, the more chance of goal it will be. White grid represents *NaN* value, which means there's no such cases for us to calculate the chance(ex.0 Wrap-around shot at 125 to 150).
#### what might be the most dangerous types of shots?
- Generally speaking, the closer the distance is, the more chance of goal it will be regardless of the type of shots(the first row of the map is obviously brighter than others).
- Wrap-around at a distance between 25 and 50(which is a relavently close distance to net) and Deflected at a distance between 150 and 175.