# Part 2: Feature Engineering 1

## Question 1

Using your training dataset create a tidied dataset for each SHOT/GOAL event, with the following columns (you can name them however you want):

- `Distance from net`: (shot_dist)
- `Angle from net`: (angle_net)
- `Is goal` (0 or 1): (is_goal)
- `Empty Net` (0 or 1; you can assume NaNs are 0): (empty_Net)

You can approximate the net as a single point (i.e., you don’t need to account for the width of the net when computing the distance or angle). You should be able to create this easily using the functionality you implemented for tidying data in Milestone 1, as you will only need the (x, y) coordinates for each shot/goal event. Create and include the following figures in your blog post and briefly discuss your observations (a few sentences):

- A histogram of shot counts (goals and no-goals separated), binned by distance
- A histogram of shot counts (goals and no-goals separated), binned by angle
- A 2D histogram where one axis is the distance and the other is the angle. You do not need to separate goals and no-goals.
  - **Hint**: check out jointplots.

As always, make sure all of your axes are labeled correctly, and you make the appropriate choice of axis scale.


### Question 1.1 (Histograms of shot counts (goals and no-goals separated), binned by distance)

![Question 1.1 (1)](images/FeatureEngineering1_Q2_1.png)
![Question 1.1 (2)](images/FeatureEngineering1_Q2_2.png)

Two histograms from the `2016-17` to `2019-20` regular season games data show shot counts with and without goals, highlighting their distance from the net. The data reveals a sharp decrease in shots on goal **beyond 70ft**, with scoring chances diminishing at greater distances. Additionally, shots from a team's own half-rink at the opponent's goal are rare and seldom successful. This trend may be due to the reduced apparent size of the goal from a distance and increased reaction time for the goaltender.



### Question 1.2 (Histograms of shot counts (goals and no-goals separated), binned by angle)

![Question 1.2 (1)](images/FeatureEngineering1_Q2_4.png)
![Question 1.2 (2)](images/FeatureEngineering1_Q2_5.png)

Two histograms from the `2016-17` to `2019-20` regular season data show shot counts with and without goals, based on their angle to the net. The data indicates that shots are commonly taken either straight at the net or at around an **abs(30-degree) angle**. Goals are most frequently scored from direct shots. Beyond an **abs(30) degrees**, the likelihood of scoring decreases.



### Question 1.3 (2D histogram where one axis is the distance and the other is the angle)

![Question 1.3](images/FeatureEngineering1_Q2_7.png)

The 2D histogram suggests a trend where shooters prefer wide-angle shots when near the goal and choose smaller angles from a distance. Key shooting zones are identified close to the goal, **around 60 feet away**, and at an **abs(20 to 30-degree angle)**, also **around 10 feet away**, and at an **abs(0 to 10-degree angle)**. The joint plot reveals a pattern of shots primarily taken from moderate distances with minimal angles. This decrease in shot frequency with greater distance or sharper angles is probably because shots become more challenging under these conditions.


## Question 2

Now, create two more figures relating the goal rate, i.e., `#goals / (#no_goals + #goals)`, to the distance, and goal rate to the angle of the shot. Include these figures in your blog post and briefly discuss your observations.

![Question 2 (1)](images/FeatureEngineering1_Q2_8.png)
![Question 2 (2)](images/FeatureEngineering1_Q2_9.png)

Two line charts show the goal rate based on the **angle** and **distance** to the net. They reveal that goals are more likely with closer proximity and smaller angles to the net. Interestingly, there's an increase in goal rate at longer distances, likely because players shoot from afar more often when the net is empty, enhancing the scoring chances.


## Question 3

Finally, let's do some quick checks to see if our data makes sense. Unfortunately, we don't have time to do automated anomaly detection, but we can use our "domain knowledge" for some quick sanity checks! The domain knowledge is that "it is incredibly rare to score a non-empty net goal on the opposing team from within your defensive zone." Knowing this, create another histogram, this time of goals only, binned by distance, and separate empty net and non-empty net events. Include this figure in your blog post and discuss your observations. Can you find any events that have incorrect features (e.g., wrong x/y coordinates)? If yes, prove that one event has incorrect features.

- **Hint**: the NHL gamecenter usually has video clips of goals for every game.

![Question 3 (1)](images/FeatureEngineering1_Q2_10.png)
![Question 3 (2)](images/FeatureEngineering1_Q2_11.png)

The histograms compare scoring with empty and non-empty nets, highlighting a clear difference in scoring distances. Goals against non-empty nets are typically scored from close range, whereas with empty nets, shooters have a higher chance of scoring from various distances across the rink.

![Question 3 (3)](images/FeatureEngineering1_Q2_13.png)
![Question 3 (4)](images/DebuggingTool.png)

This filtering aims to identify anomalous goal events in a hockey dataset where goals from a distance greater than 89 units (from opposing team) and not into an empty net are considered anomalies.
