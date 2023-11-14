# Part 2: Feature Engineering I

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

```python
# Function to Create and Save Histogram
def create_histogram(data, file_name, x_title, title, edge_color, color):
    # Create figure and calculate bins
    fig, ax = plt.subplots(figsize = (10, 5))
    num_bins = math.ceil((data.max() - data.min()) / 10)
    # Ensure the range covers the max value
    bins = np.arange(0, data.max() + 20, 10)
    
    # Create histogram
    ax.hist(data, bins = num_bins, color = color, edgecolor = edge_color)
    ax.grid(axis = 'y')
    ax.set_title(title)
    ax.set_xlabel('Net Distance')
    ax.set_ylabel(x_title)
    ax.set_xticks(bins)
    
    fig.savefig(file_name)
```
```python
# Plot Histogram for Goals
create_histogram(
    df_goals['shot_dist'], 
    'FeatureEngineering1_Q2_1.png',
    'Shot Counts (Goals)',
    'Shot Counts (Goals) based on Distance', 
    'white',
    '#00008B'
)
```

![Question 1.1 (1)](images/FeatureEngineering1_Q2_1.png)

```python
# Plot Histogram for No Goals
create_histogram(
    df_no_goals['shot_dist'],
    'FeatureEngineering1_Q2_2.png',
    'Shot Counts (No Goals)',
    'Shot Counts (No Goals) based on Distance',
    'white',
    '#DC143C'
)
```

![Question 1.1 (2)](images/FeatureEngineering1_Q2_2.png)

Two histograms from the `2016-17` to `2019-20` regular season games data show shot counts with and without goals, highlighting their distance from the net. The data reveals a sharp decrease in shots on goal **beyond 70 feet**, with scoring chances diminishing at greater distances. Additionally, shots from a team's own half-rink at the opponent's goal are rare and seldom successful. This trend may be due to the reduced apparent size of the goal from a distance and increased reaction time for the goaltender.



### Question 1.2 (Histograms of shot counts (goals and no-goals separated), binned by angle)

```python
# Function to plot histogram
def plot_histogram(data, filename, title, xlabel, ylabel, color, edge_color):
    # Calculate the number of bins
    bins = math.ceil((data.max() - data.min()) / 10)
    
    # Create figure and plot histogram
    fig, ax = plt.subplots(figsize = (10, 5))
    ax.hist(data, bins = bins, color = color, edgecolor = edge_color)
    ax.grid(axis = 'y')
    ax.set(title = title, xlabel = xlabel, ylabel = ylabel)
    ax.set_xticks(np.arange(-90, data.max() + 10, 10))
    
    fig.savefig(filename)
```

```python
# Plot Histogram for Goals
plot_histogram(
    df_goals['angle_net'],
    'FeatureEngineering1_Q2_4.png',
    'Shot Counts (Goals) based on Angle',
    'Shot Angle',
    'Shot Count (Goals)',
    '#00008B',
    'white'
)
```

![Question 1.2 (1)](images/FeatureEngineering1_Q2_4.png)

```python
# Plot Histogram for No Goals
plot_histogram(
    df_no_goals['angle_net'],
    'FeatureEngineering1_Q2_5.png',
    'Shot Counts (No Goals) based on Angle',
    'Shot Angle',
    'Shot Count (No Goals)',
    '#DC143C',
    'white'
)
```

![Question 1.2 (2)](images/FeatureEngineering1_Q2_5.png)

Two histograms from the `2016-17` to `2019-20` regular season data show shot counts with and without goals, based on their angle to the net. The data indicates that shots are commonly taken either straight at the net or at around an **abs(30-degree) angle**. Goals are most frequently scored from direct shots. Beyond an **abs(30) degrees**, the likelihood of scoring decreases.



### Question 1.3 (2D histogram where one axis is the distance and the other is the angle)

```python
# Create the joint-plot with the specified parameters
joint_plot = sns.jointplot(
    data = train_df,
    kind = "hist",
    x = "angle_net",
    y = "shot_dist",
    color = '#6F2DA8'
)
joint_plot.fig.set_size_inches(10, 10)
joint_plot.ax_joint.grid(True)
joint_plot.fig.suptitle('Shot Counts (2D Joint-Plot) based on Distance and Angle')
joint_plot.ax_joint.set_xlabel('Angle to Net')
joint_plot.ax_joint.set_ylabel('Shot Distance')
joint_plot.ax_joint.set_xticks(np.arange(-90, train_df['angle_net'].max() + 10, 10))
joint_plot.ax_joint.set_yticks(np.arange(0, train_df['shot_dist'].max() + 20, 10))

joint_plot.savefig("FeatureEngineering1_Q2_7.png")
```

![Question 1.3](images/FeatureEngineering1_Q2_7.png)

The 2D histogram suggests a trend where shooters prefer wide-angle shots when near the goal and choose smaller angles from a distance. Key shooting zones are identified close to the goal, **around 60 feet away**, and at an **abs(20 to 30-degree angle)**, also **around 10 feet away**, and at an **abs(0 to 10-degree angle)**. The joint plot reveals a pattern of shots primarily taken from moderate distances with minimal angles. This decrease in shot frequency with greater distance or sharper angles is probably because shots become more challenging under these conditions.


## Question 2

Now, create two more figures relating the goal rate, i.e., `#goals / (#no_goals + #goals)`, to the distance, and goal rate to the angle of the shot. Include these figures in your blog post and briefly discuss your observations.

```python
# Copying the dataframe and creating bins
df_copy = train_df.copy()
df_copy['angle_shot_categories'] = pd.cut(df_copy['angle_net'], bins = np.arange(-90, df_copy['angle_net'].max() + 10, 4), labels = (np.arange(-90, df_copy['angle_net'].max() + 10, 4)[:-1] + np.arange(-90, df_copy['angle_net'].max() + 10, 4)[1:]) / 2)
grouped = df_copy.assign(attempt = 1).groupby('angle_shot_categories').agg({'attempt': 'sum', 'is_goal': 'sum'}).eval('success_rate = is_goal / attempt')

# Line plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(grouped.index, grouped['success_rate'], color = '#6F2DA8', marker = 'o', linestyle = '-')
ax.set(title = 'Goal Rate based on Shot Angle', xlabel = 'Shot Angle', ylabel = 'Goal Rate')
ax.set_xticks(np.arange(-90, df_copy['angle_net'].max() + 10, 10))
ax.grid(True, which = 'major', linestyle = '-', linewidth = '0.5', color = 'grey')
ax.fill_between(grouped.index, grouped['success_rate'], color = '#6F2DA8', alpha = 0.3)

# Save the plot
plt.savefig("FeatureEngineering1_Q2_8.png")
```

![Question 2 (1)](images/FeatureEngineering1_Q2_8.png)

```python
# Creating bins and categorizing 'shot_dist'
df_copy['distance_bins'] = pd.cut(train_df['shot_dist'], bins = np.arange(0, 204, 4), labels = np.arange(2, 202, 4))
summary_data = df_copy.assign(shot_attempt = 1).groupby('distance_bins').agg({'shot_attempt': 'sum', 'is_goal': 'sum'}).eval('success_rate = is_goal / shot_attempt')

# Line plot
fig, ax = plt.subplots(figsize = (10, 5))
ax.plot(summary_data.index, summary_data['success_rate'], color = '#6F2DA8', marker = 'o', linestyle = '-')
ax.set(title = 'Goal Rate based on Shot Distance', xlabel = 'Shot Distance', ylabel = 'Goal Rate')
ax.set_xticks(np.arange(0, df_copy['shot_dist'].max() + 20, 10))
ax.grid(True, which = 'major', linestyle = '-', linewidth = '0.5', color = 'grey')
ax.fill_between(summary_data.index, summary_data['success_rate'], color = '#6F2DA8', alpha = 0.3)

# Save the plot
plt.savefig("FeatureEngineering1_Q2_9.png")
```

![Question 2 (2)](images/FeatureEngineering1_Q2_9.png)

Two line charts show the goal rate based on the **angle** and **distance** to the net. They reveal that goals are more likely with closer proximity and smaller angles to the net. Interestingly, there's an increase in goal rate at longer distances, likely because players shoot from afar more often when the net is empty, enhancing the scoring chances.


## Question 3

Finally, let's do some quick checks to see if our data makes sense. Unfortunately, we don't have time to do automated anomaly detection, but we can use our "domain knowledge" for some quick sanity checks! The domain knowledge is that "it is incredibly rare to score a non-empty net goal on the opposing team from within your defensive zone." Knowing this, create another histogram, this time of goals only, binned by distance, and separate empty net and non-empty net events. Include this figure in your blog post and discuss your observations. Can you find any events that have incorrect features (e.g., wrong x/y coordinates)? If yes, prove that one event has incorrect features.

- **Hint**: the NHL gamecenter usually has video clips of goals for every game.

```python
# Goal Success Analysis in Hockey: With vs. Without Goalies
goalie_net_goals = train_df.query('empty_Net == 0 and is_goal == 1')
empty_net_goals = train_df.query('empty_Net == 1 and is_goal == 1')

# Histogram Bin Calculation
bin_step = 10
bins_empty = np.arange(empty_net_goals['shot_dist'].min(), empty_net_goals['shot_dist'].max() + bin_step, bin_step)
bins_goalie = np.arange(goalie_net_goals['shot_dist'].min(), goalie_net_goals['shot_dist'].max() + bin_step, bin_step)

# Histogram Plotting Function
def plot_goals_hist(data, title, x_label, y_label, bins, color, save_path):
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=bins, edgecolor='white', color=color)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(np.arange(0, train_df['shot_dist'].max() + 20, 10))
    plt.grid(axis='y')
    plt.savefig(save_path)

# Histogram Plotting Function (Merged Function)
def plot_merged_goals_hist(data1, data2, title, x_label, y_label, bins1, bins2, color1, color2, legend_labels, save_path):
    plt.figure(figsize=(10, 5))
    plt.hist(data1, bins=bins1, edgecolor='white', color=color1, label=legend_labels[0])
    plt.hist(data2, bins=bins2, edgecolor='white', color=color2, label=legend_labels[1])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(np.arange(0, max(data1.max(), data2.max()) + 20, 10))
    plt.grid(axis='y')
    plt.legend()
    plt.savefig(save_path)

# The DataFrame's Shape
print('\n' + "Size of the goalie_net_goals: " + str(goalie_net_goals.shape))
print('\n' + "Size of the empty_net_goals: " + str(empty_net_goals.shape) + '\n')
```
```python
# Plot Histogram for Goals with Goalie in Net
plot_goals_hist(
    goalie_net_goals['shot_dist'],
    'Goal Counts (Goalie in Net) based on Distance',
    'Net Distance',
    'Shot Counts (Goals)',
    bins_goalie,
    '#00008B',
    'FeatureEngineering1_Q2_10.png'
)
```

![Question 3 (1)](images/FeatureEngineering1_Q2_10.png)

```python
# Plot Histogram for Goals with Empty Net
plot_goals_hist(
    empty_net_goals['shot_dist'],
    'Goal Counts (Empty Net) based on Distance',
    'Net Distance',
    'Shot Counts (Goals)',
    bins_empty,
    'silver',
    'FeatureEngineering1_Q2_11.png'
)
```

![Question 3 (2)](images/FeatureEngineering1_Q2_11.png)

```python
# Plot Histogram for Goals (Empty Net vs. Goalie in Net)
plot_merged_goals_hist(
    goalie_net_goals['shot_dist'],
    empty_net_goals['shot_dist'],
    'Goal Counts based on Distance (Goalie in Net vs Empty Net)',
    'Net Distance',
    'Shot Counts (Goals)',
    bins_goalie,
    bins_empty,
    '#00008B',
    '#DC143C',
    ['Goalie in Net', 'Empty Net'],
    'FeatureEngineering1_Q2_12.png'
)
```

![Question 3 (3)](images/FeatureEngineering1_Q2_12.png)

The histograms compare scoring with empty and non-empty nets, highlighting a clear difference in scoring distances. Goals against non-empty nets are typically scored from close range, whereas with empty nets, shooters have a higher chance of scoring from various distances across the rink.

```python
# Filtering the DataFrame for goals with unusual conditions
#goalie_net_goals = train_df.query('empty_Net == 0 and is_goal == 1')
#empty_net_goals = train_df.query('empty_Net == 1 and is_goal == 1')
def parse_coordinate(coord_str):
    return abs(ast.literal_eval(coord_str).get('x', 0))
df_goals_copy = df_goals.copy()
df_goals_copy['x_coord'] = df_goals_copy['coordinate'].apply(parse_coordinate)

# 89 is the "net" X-coordination and distance > 89 is for the opposing team
anomaly_condition = lambda row: (row['x_coord'] > 89) and (row['shot_dist'] > 89) and (row['empty_Net'] == 0)
#anomaly_condition = lambda row: (row['x_coord'] > 89) and (row['shot_dist'] > 89) and (row['empty_Net'] == 1)

# Apply the filter function to the DataFrame
anomalous_goals = df_goals_copy[df_goals_copy.apply(anomaly_condition, axis=1)]

# Save the resulting DataFrame as a PNG file
dfi.export(anomalous_goals, 'FeatureEngineering1_Q2_13.png')
print('\n' + "Size of the main (train_df) DataFrame: " + str(train_df.shape) + '\n')
print("Size of the anomalous_goals DataFrame: " + str(anomalous_goals.shape) + '\n')
anomalous_goals
```
![Question 3 (4)](images/FeatureEngineering1_Q2_13.png)
![Question 3 (5)](images/DebuggingTool.png)

This filtering aims to identify anomalous goal events in a hockey dataset where goals from a distance greater than 89 units (from opposing team) and not into an empty net are considered anomalies.
