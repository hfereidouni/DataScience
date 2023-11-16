# Part 1: Comet.ml
In subsequent tasks, you will be asked to reference experiment entries in your comet.ml workspace which are tied to the models that produced specific figures.


## Introduction to Comet.ml

`Comet.ml` is a powerful tool for tracking, managing, and optimizing your machine learning experiments. It provides a centralized platform for storing all experiment metadata, including configurations, hyperparameters, dataset hashes, and saved models. Comet.ml is especially beneficial for team projects, as it facilitates collaboration and ensures reproducibility of results.


## Setting Up Comet.ml

### Step 1: Create a Comet.ml Account
**For All Team Members**: Sign up for a `Comet.ml` account using your academic email via `Comet.ml` signup.

**Claim Your Free Academic Tier**: After signing up, claim your free academic tier benefits.

### Step 2: Establish a Shared Workspace
**Designate a Workspace Host**: One team member should create a workspace in `Comet.ml`.

**Add Team Members**: The **workspace** host should add all team members and the TA account `ift6758-2023` to the **workspace**. This is done in the workspace settings.

### Step 3: Install Comet.ml SDK
Install the `Comet.ml` Python SDK in your environment:

`pip install comet_ml`

## Integrating Comet.ml with Scikit-Learn

### Step 1: Import Comet.ml
Import `comet_ml`at the top of your `Python` script, before other libraries like `sklearn`.

```python
from comet_ml import Experiment
```

### Step 2: Configure Environment Variables
**Set API Key**: Store your `Comet.ml` API key in an environment variable named COMET_API_KEY.

**Access API Key**: Use the os package in Python to access the **API key**.

```python
import os
api_key = os.environ.get('COMET_API_KEY')
```

### Step 3: Create and Configure an Experiment
Initialize a new experiment with your API key, project name, and workspace:

```python
exp = Experiment(
    api_key=api_key,
    project_name='your_project_name',
    workspace='your_workspace_name'
)
```

### Step 4: Tracking Experiments
**Log Metrics**: Use exp.log_metrics() to log metrics like accuracy, loss, etc.

```python
exp.log_metrics({"accuracy": accuracy_score, "loss": loss_value})
```

**Log Parameters**: Log model parameters and hyperparameters.

```python
exp.log_parameters({"C": 1.0, "kernel": "linear"})
```

### Step 5: Saving and Registering Models
**Save Model Locally**: Save your `scikit-learn` model using joblib or pickle.

```python
from joblib import dump
dump(your_model, 'model.joblib')
```
**Log Model in Comet.ml**: Use `exp.log_model()` to log the model in `Comet.ml`.

```python
exp.log_model("model_name", "model.joblib")
```

### Step 6: Review and Analyze
**Accessing Dashboard**: Review your experiments on the `Comet.ml` dashboard.

**Collaboration**: Share experiment links with your team for collaborative analysis.


## Collaborative Use of Shared Workspace and API

Our team has set up a shared workspace in `Comet.ml` under the username `hfereidouni` to facilitate collaborative experiment tracking. All team members are encouraged to use this workspace for logging their experiments. To ensure secure and consistent access, we are utilizing a shared **API key**. This approach allows every team member to connect to the same workspace, enabling us to monitor, compare, and discuss our experiments in real-time. It’s important to remember not to hardcode the API key in your scripts or notebooks. Instead, store it as an environment variable on your machine and access it programmatically. This practice keeps our workspace secure and prevents unauthorized access. By using this centralized setup, we can maintain a cohesive and organized overview of our project's progress, making it easier to identify successful experiments and collaborate more effectively.


## Conclusion

By following these steps, our team can effectively use `Comet.ml` to track machine learning experiments using `scikit-learn`. This setup ensures that all experiments are logged and reproducible, facilitating a more organized and collaborative project environment. Remember to frequently consult Comet.ml's documentation for specific features and advanced usage.



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

**Train set**: from 2016-2017 to 2019-2020 - Including Regular games "02"

**Test set**: 2020-2021 - Including Regular games "02"

**NOTE** : First, I incorporated all the features mentioned above into the TidyData.py file, modifying the previous version used in Milestone 1. Then, based on this modified file, I created the plots shown below.

**Snippet**: *Histograms of shot counts (goals and no-goals separated), binned by distance*:
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

**Snippet**: *Histograms of shot counts (goals), binned by distance*:
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

![Question 1.1 (1)](images/part2/FeatureEngineering1_Q2_1.png)

**Snippet**: *Histograms of shot counts (no-goals), binned by distance*:
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

![Question 1.1 (2)](images/part2/FeatureEngineering1_Q2_2.png)

Two histograms from the `2016-17` to `2019-20` regular season games data show shot counts with and without goals, highlighting their distance from the net. The data reveals a sharp decrease in shots on goal **beyond 70 feet**, with scoring chances diminishing at greater distances. Additionally, shots from a team's own half-rink at the opponent's goal are rare and seldom successful. This trend may be due to the reduced apparent size of the goal from a distance and increased reaction time for the goaltender.



### Question 1.2 (Histograms of shot counts (goals and no-goals separated), binned by angle)

**Snippet**: *Histograms of shot counts (goals and no-goals separated), binned by angle*:
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

**Snippet**: *Histograms of shot counts (goals), binned by angle*:
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

![Question 1.2 (1)](images/part2/FeatureEngineering1_Q2_4.png)

**Snippet**: *Histograms of shot counts (no-goals), binned by angle*:
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

![Question 1.2 (2)](images/part2/FeatureEngineering1_Q2_5.png)

Two histograms from the `2016-17` to `2019-20` regular season data show shot counts with and without goals, based on their angle to the net. The data indicates that shots are commonly taken either straight at the net or at around an **abs(30-degree) angle**. Goals are most frequently scored from direct shots. Beyond an **abs(30) degrees**, the likelihood of scoring decreases.



### Question 1.3 (2D histogram where one axis is the distance and the other is the angle)

**Snippet**: *2D histogram where one axis is the distance and the other is the angle*:
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

![Question 1.3](images/part2/FeatureEngineering1_Q2_7.png)

The 2D histogram suggests a trend where shooters prefer wide-angle shots when near the goal and choose smaller angles from a distance. Key shooting zones are identified close to the goal, **around 60 feet away**, and at an **abs(20 to 30-degree angle)**, also **around 10 feet away**, and at an **abs(0 to 10-degree angle)**. The joint plot reveals a pattern of shots primarily taken from moderate distances with minimal angles. This decrease in shot frequency with greater distance or sharper angles is probably because shots become more challenging under these conditions.


## Question 2

Now, create two more figures relating the goal rate, i.e., `#goals / (#no_goals + #goals)`, to the distance, and goal rate to the angle of the shot. Include these figures in your blog post and briefly discuss your observations.

**Snippet**: *Goal rate to the angle of the shot*:
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

![Question 2 (1)](images/part2/FeatureEngineering1_Q2_8.png)

**Snippet**: *Goal rate to the distance of the shot*:
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

![Question 2 (2)](images/part2/FeatureEngineering1_Q2_9.png)

Two line charts show the goal rate based on the **angle** and **distance** to the net. They reveal that goals are more likely with closer proximity and smaller angles to the net. Interestingly, there's an increase in goal rate at longer distances, likely because players shoot from afar more often when the net is empty, enhancing the scoring chances.


## Question 3

Finally, let's do some quick checks to see if our data makes sense. Unfortunately, we don't have time to do automated anomaly detection, but we can use our "domain knowledge" for some quick sanity checks! The domain knowledge is that "it is incredibly rare to score a non-empty net goal on the opposing team from within your defensive zone." Knowing this, create another histogram, this time of goals only, binned by distance, and separate empty net and non-empty net events. Include this figure in your blog post and discuss your observations. Can you find any events that have incorrect features (e.g., wrong x/y coordinates)? If yes, prove that one event has incorrect features.

- **Hint**: the NHL gamecenter usually has video clips of goals for every game.

**Snippet**: *Histogram for Goals with Goalie in Net and for Goals with Empty Net*:
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

**Snippet**: *Histogram for Goals with Goalie in Net*:
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

![Question 3 (1)](images/part2/FeatureEngineering1_Q2_10.png)

**Snippet**: *Histogram for Goals with Empty Net*:
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

![Question 3 (2)](images/part2/FeatureEngineering1_Q2_11.png)

**Snippet**: *Histogram for Goals (Empty Net vs. Goalie in Net)*:
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

![Question 3 (3)](images/part2/FeatureEngineering1_Q2_12.png)

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
![Question 3 (4)](images/part2/FeatureEngineering1_Q2_13.png)
![Question 3 (5)](images/part2/DebuggingTool.png)

This filtering aims to identify anomalous goal events in a hockey dataset where goals from a distance greater than 89 units (from opposing team) and not into an empty net are considered anomalies.



# Part 3: Baseline Models

## Question 1
Using your dataset (remember, don’t touch the test set!), create a training and validation split however you feel is appropriate. Using only the distance feature, train a Logistic Regression classifier with the completely default settings, i.e.: 

clf = LogisticRegression()
clf.fit(X, y)

Evaluate the accuracy (i.e. correctly predicted / total) of your model on the validation set. What do you notice? Look at the predictions and discuss your findings. What could be a potential issue? Include these discussions in your blog post.

### Answer 1
Validation accuracy for Logistic regression using "shot_dist" feature only is around 0.9075
However, when we see the predictions, they are all referring to one class, "0" ("no goal" class).
This is due to the class imbalance in the dataset, where there are more examples for one class than the other. 
Here, looking at the distribution of Y_valid (for 80%-20% train_valid_split), we see the following frequencies for classes 0 and 1
```
np.bincount(Y_valid) -> [67211, 6999] (freq_class_0, freq_class_1)
```
This highlights the class imbalance in dataset.

## Question 2
Based on your findings in Q1, we should explore other ways of evaluating our model. The first thing to note is that we are not actually interested in the binary prediction of whether a shot is a goal or not - we are interested in the probability that the model assigns to it (recall that we’re interested in the notion of expected goals). Scikit-learn provides this to you via the predict_proba(...) method; make sure you take the probabilities of the class that you care about! You will produce four figures (one curve per model per plot) to probe our model’s performance. Make sure you are using the probabilities obtained on the validation set:
1. Receiver Operating Characteristic (ROC) curves and the AUC metric of the ROC curve. Include a random classifier baseline, i.e. each shot has a 50% chance of being a goal.
2. The goal rate (#goals / (#no_goals + #goals)) as a function of the shot probability model percentile, i.e. if a value is the 70th percentile, it is above 70% of the data.
3. The cumulative proportion of goals (not shots) as a function of the shot probability model percentile.
4. The reliability diagram (calibration curve). Scikit-learn provides functionality to create a reliability diagram in a few lines of code; check out the CalibrationDisplay API (specifically the .from_estimator() or .from_predictions() methods) for more information.

An example of what is expected for (b) and (c) are shown in Fig. 1. Do not include all of these in your blog post yet, as you will add a few more curves in the next section. 

### Answer 2
**Example of curves produced for shot_dist only**

**ROC Curve**

![Image](images/part3/Log_Reg_ROC_shot_dist_only.png)

**Goal Rate v/s Shot Probability percentile**

![Image](images/part3/Log_Reg_goalratepercentile_shot_dist_only.png)

**Cumulative Goal Rate v/s Shot Probability percentile**

![Image](images/part3/Log_Reg_cumulativegoalrate_shot_dist_only.png)

**Reliability Curve**

![Image](images/part3/Log_Reg_calibration_shot_dist_only.png)

# Question 3
Now train two more Logistic Regression classifiers using the same setup as above, but this time on the angle feature, and then both distance and angle. Produce the same three curves as described in the previous section for each model. Including the random baseline, you should have a total of 4 lines on each figure: 
 - Logistic Regression, trained on distance only (already done above)
 - Logistic Regression, trained on angle only
 - Logistic Regression, trained on both distance and angle
 - Random baseline: rather than training a classifier, the predicted probability is sampled from a uniform distribution, i.e. yiU(0,1)

Include these four figures (each with four curves) in your blogpost. In a few sentences, discuss your interpretation of these results. 

## Question 4
Next to the figures, include links to the three experiment entries in your comet.ml projects that produced these three models. Save the three models to the three experiments on comet.ml (example here) and register them with some informative tags, as you will need it for the final section. 

### Answer 3 & 4 COMBINED

### **Logistic Regression, trained on distance only**
![Image](images/part3/shot_dist_only_per-experiment-graph.png)

**Discussion**

The Logistic regresion classifier for "shot distances only" works well above the random classifier. However, from the "Goal Rate graph" and "Cumulative % of goals" we notice that even the higher shot probability percentiles (which means probability of goal being high) don't result in higher goal rates, indicating the presence of more #no_goals examples than #goal examples. The calibration curve tells us the real proportion of positive predictions for a given calculated probability of a positive class.

Experiment entry link: https://www.comet.com/hfereidouni/ift6758/ee98743fc69e489883e4e6f81a7b836e?assetId=86e8d61ab83a48f29fc8e8dfad3ec9c4&assetPath=models%2CLog_Reg_shot_dist_only&experiment-tab=assetStorage

Model Registry entry: https://www.comet.com/hfereidouni/model-registry/log_reg_shot_dist_only/1.15.0?tab=assets

### **Logistic Regression, trained on angle only**
![Image](images/part3/angle_only_per-experiment-graph.png)

**Discussion**

The Logistic regresion classifier for shot angles only works similarly to the random classifier. According to the ROC curve, we see that FPR > TPR till FPR=0.5 and after 0.5 value TPR > FPR. Increased FPR points to low precision model. From the Goal Rate and Cumulative % of goals graphs, we notice that the goal rate aproximately stays the same irrepsective of the shot probability model percentile (for both high and low model probabilities). This implies that the model fails to accumulate/ group high probabilities with goal examples. The calibration curve show that this model is poorly calibrated with actual data.

Experiment entry link: https://www.comet.com/api/experiment/redirect?assetId=31c269d20c2449828cc40d37767e7fe8&assetPath=models,Log_Reg_angle_only&experiment-tab=assetStorage&experimentKey=46d8c6c8f29244018fc7adb0aa290e3f

Model Registry link: https://www.comet.com/hfereidouni/model-registry/log_reg_angle_only/1.9.0?tab=assets


## **Logistic Regression, trained on both distance and angle**
![Image](images/part3/shot_dist_and_angle_per-experiment-graph.png)

**Discussion**

The Logistic regresion classifier for both shot angle and distance features works well above to the random classifier. Similar to "shot distances only" classifier, we notice the presence of class imbalance from "Goal Rate" and "Cumulative % of goals" graphs. We also see that a larger proportion of positive class population (#goal) lies within a small range of model predicted probabilities, similar to "shot ditance only" Logistic Rgression classifier.


Experiment Entry Link: https://www.comet.com/api/experiment/redirect?assetId=1540e95509844ca8a533c7dd025c8500&assetPath=models,Log_Reg_shot_dist_and_angle&experiment-tab=assetStorage&experimentKey=335a9643c0cd42288290f1c02de30661

Model Registry link: https://www.comet.com/hfereidouni/model-registry/log_reg_shot_dist_and_angle/1.7.0?tab=assets


## **Random baseline: rather than training a classifier, the predicted probability is sampled from a uniform distribution**
![Image](images/part3/random_classifier_per-experiment-graph.png)

**Discussion**

The Random classifier follows uniform distribution and hence gives AUC of 0.5 . 
From the "Goal Rate" and "Cumulative% of goals" graph, we see that there is no effect on goal rate for any shot probabilities (high or low), which is because the random classifier doesn't differentiate between a high and low model probability (high model probability shoud accumulate more examples of goals).
Furthermore, the reliability curve shows that the random classifier places the same frequency of goals dsitribution for each probability bin. This shows the class imbalance in data (same freq. of goals) and poor calibration of random classifier.


Experiment Entry Link: https://www.comet.com/api/experiment/redirect?assetId=12623ff25c504b37b3c853922ecb4590&assetPath=models,Random_classifer_random_classifier&experiment-tab=assetStorage&experimentKey=38ca0870199140b685b9f5fccb79befb

Model Registry link: https://www.comet.com/hfereidouni/model-registry/random_classifer_random_classifier/1.5.0?tab=assets



# Part 4: Feature Engineering II

## Questions 1 to 4

**Features selected**:
`['game_time','period','x','y','shot_type','last_event_type',
    'x_coord_last_event', 'y_coord_last_event', 'Time_from_the_last_event',
    'Distance_from_the_last_event', 'Rebound', 'change_shot_angle', 'Speed',
    'shot_dist','angle_net', 'is_goal']`
    
### Question 1:
- **game_time**: The time (seconds) where this event happened in current game.
- **period**: The period that in which event happened in current game.
- **x**, **y**: The `x` and `y` coordinate of where this event happened in current game.
- **shot_type**: Type of shot of the this event.
- **shot_dist**: The distance of this event to the net of the attack rinkside.
- **angle_net**: The angle of this event to the net of the attack rinkside, calculated using `shot_dist`.

### Question 2:
- **last_event_type**: Type of event (play) of the previous event (of this event) in current game.
- **x_coord_last_event**, **y_coord_last_event**: The x and y coordinate of where the previous event happened in current game.
- **Time_from_the_last_event**: The time (seconds) where the previous event happened in current game.
- **Distance_from_the_last_event**: the distance between the coordinates of this event and the coordinates of the previous event.

### Question 3:
- **Rebound**: A boolean value of whether this event is a rebound (the previous event was also a shot) or not.
- **change_shot_angle**: The difference of angle of the shot of this event and that of the previous event, this is only a valid value if this event is a `rebound`.
- **Speed**: The average speed between the previous event and this event, calculated using `Distance_from_the_last_event` divided by `Time_from_the_last_event`.

Other:
- **is_goal**: A binary Label that if this event is a goal: 1 if it's a goal otherwise 0.

### Question 4:

**NOTE:** To check this part, please check the `FeatureEngineering2_Q4_Bonus.py`, `FeatureEngineering2_Q4_Bonus.ipynb` and `AdvancedModel_helpers2.py`


**Function: process_penalties(plays):**
```python
def process_penalties(plays):
    penalties = {'home': [], 'away': []}
    penalty_expirations = {'home': [], 'away': []}

    for play in plays:
        if play["result"]["event"] == "Penalty":
            team = play["team"]["triCode"]
            period = play["about"]["period"]
            period_time = play["about"]["periodTime"]
            penalty_minutes = play["result"]["penaltyMinutes"]

            game_time = (period - 1) * 20 + int(period_time.split(':')[0]) + penalty_minutes
            penalized_team = 'home' if team == 'away' else 'away'

            penalties[penalized_team].append(penalty_minutes)
            penalty_expirations[penalized_team].append(game_time)

            # Debugging
            print(f"Penalty detected: Team {penalized_team}, Game time {game_time} minutes")

    return penalties, penalty_expirations
```
This function analyzes a list of plays from a game, identifying and handling penalty events. It calculates when each penalty will expire based on the game's progress and logs these details. The function is primarily focused on tracking penalties for both the home and away teams, including the duration and expiration times of these penalties.

**Function: update_powerplay_status(penalties, penalty_expirations, current_game_time):**
```python
def update_powerplay_status(penalties, penalty_expirations, current_game_time):
    for team in ['home', 'away']:
        penalties[team] = [penalty for penalty, expiration in zip(penalties[team], penalty_expirations[team]) if expiration > current_game_time]
        penalty_expirations[team] = [expiration for expiration in penalty_expirations[team] if expiration > current_game_time]

    home_skaters = 5 - len(penalties['home'])
    away_skaters = 5 - len(penalties['away'])

    powerplay_team = 'home' if len(penalties['home']) < len(penalties['away']) else 'away'
    powerplay_time = 0
    if len(penalties[powerplay_team]) < len(penalties['home' if powerplay_team == 'away' else 'away']):
        powerplay_time = min(penalty_expirations[powerplay_team]) if penalty_expirations[powerplay_team] else 0

    return home_skaters, away_skaters, powerplay_time
```
update_powerplay_status is designed to update the current status of penalties and powerplays in a game based on the ongoing time. It removes expired penalties and calculates the number of players each team has on the ice. The function also identifies which team, if any, has a powerplay advantage and the remaining duration of that powerplay.

**Function: calculate_game_time(period, period_time):**
```python
def calculate_game_time(period, period_time):
    minutes, seconds = map(int, period_time.split(':'))
    return (period - 1) * 20 * 60 + minutes * 60 + seconds
```
This utility function converts the period and time within a period of a game into total game time in seconds. It's a straightforward calculation used to standardize time references in the game, allowing for easier comparison and assessment of time-dependent elements like penalties or play durations.

The list of features:

- **HomeSkaters**: The number of non-goalie skaters currently on the ice for the home team.
- **AwaySkaters**: The number of non-goalie skaters currently on the ice for the away team.
- **PowerplayTime**: The amount of time elapsed since the start of the current power-play, resetting to 0 when the power-play ends.


### Overview of filtered Dataframe
![df](./images/part%204/dataframe.jpg)


## Question 5

**The filtered DataFrame artifact for the specified game (wpg_v_wsh_2017021065):**
[Comet.ml](https://www.comet.com/hfereidouni/ift6758/41a02783a9d54633a47fd78acfb9a900?experiment-tab=panels&showOutliers=true&smoothing=0&xAxis=step)


# Part 5: Advanced Models

## Question 1:
### Splitting data
We have used a rate of 75/25 training-validation split with our around 30000 datas. Comparing to commonly used 80/20 split, we decide to have a larger validation set to better evaluate our models.

In this part, a xgboost classifier that is trained with only *shot distance* and *shot angle* has been introduced:

[Comet.ml Experiment](https://www.comet.com/hfereidouni/ift6758/e0a5aa92457347cb8d4f817f7697747b?experiment-tab=panels&showOutliers=true&smoothing=0&xAxis=step)

### ROC AUC figure
![ROC AUC figure](./images/roc_auc_5.1.png)
### Goal rate vs Shot probability percentile
![Goal Rate](./images/goal_rate_5.1.png)
### Cumulative % of Goals vs Shot probability percentile
![Cumulative sum](./images/cumsum_5.1.png)
### Calibration Curve
![Cali plot](./images/cali_plot_5.1.png)

### Comparaison with Baseline model
Compared to Logistic Regression model that is trained on the same features, Xgboost has a slightly better performance, AUC of 0.71 comparing to 0.69 of Logistic Regression.


## Question 2:
[Comet.ml Experiment](https://www.comet.com/hfereidouni/ift6758/cd4cdf2f3d6a47dc8a58b25261ed7c27?experiment-tab=panels&showOutliers=true&smoothing=0&xAxis=step)
- ### One-hot encoding
  For `shot_type` and `last_event_type`, we have chosen to use one-hot encoding to represents them instead of label encoding, this is to avoid an arbitrary order to categorical values. The dimension of the datas are thus **36**.
- ### Hyperparameter tuning
  references: 
    ######  https://aiinpractice.com/xgboost-hyperparameter-tuning-with-bayesian-optimization/
    ###### https://zhuanlan.zhihu.com/p/131216861
  #### Method: Bayesian Optimization
  We have choose to use **Bayesian Optimization** to tune hyperparameters for xgboost model since it's more efficient and faster comparing to regular **Grid Search** method. We have used *bayes_opt* python library to complete this part.
  #### Tuning iterations
  ![tuning](./images/hp_tune_5.2.jpg)
- ### Model
    In this part, a xgboost classifier that is trained with all features from Q4 and hyperparameter tuned has been introduced: 
    ### ROC AUC figure
    ![ROC AUC figure](./images/roc_auc_5.2.png)
    ### Goal rate vs Shot probability percentile
    ![Goal Rate](./images/goal_rate_5.2.png)
    ### Cumulative % of Goals vs Shot probability percentile
    ![Cumulative sum](./images/cumsum_5.2.png)
    ### Calibration Curve
    ![Cali plot](./images/cali_plot_5.2.png)
    ### Comparing to XGboost trained only on two parameters and without hyperparameter tuning
    The ROC_AUC score has progressed even more compared to changing models from Logisic Regression to XGboost: 0.71 -> 0.76;
    The Calibration Plot is much more converged to the oerfectkt calibrated line.
## Question 3:
- ### Feature Selection
    #### reference: https://scikit-learn.org/stable/modules/feature_selection.html
    
    ### Strategies:
    We have used three methods of feature selection
    - Filter Method: Removing features with low Variance using `VarianceThreshold` function of `sklearn` library.
    - Wrapper Method: `Recursive Feature Elimination(RFE)` function of `sklearn` library.
    - Embedded Method: L1-based feature selection using `SelectFromModel` function of `sklearn` library.
    These three methods has respectively lower the dimention of input datas from **36** to **14**,**14** and **13**, the second method has taken significantly more time than other two methods.
    From validating new models trained using selected features from these three methods and tuned hyperparameters from Q2, we have discovered that the features selected using **SelectFromModel** and **RFE** have the best performance overall(best auc roc score), but it's only slightly better than Variance Filter method, considering that the **SelectFromModel** method is much faster than the second **RFE** method and they have almost the same roc auc score, we have chosen the features selected from the second method.
    ### The features selected:
    `['game_time', 'period', 'y', 'Time_from_the_last_event', 'shot_dist',
       'shot_type_Backhand', 'shot_type_Deflected', 'shot_type_Wrap-around',
       'shot_type_Wrist Shot', 'last_event_type_Hit',
       'last_event_type_Missed Shot', 'last_event_type_Shot',
       'last_event_type_Stoppage']`

    ### The ROC/AUC curve of comparaison:
    ![ROC AUC figure](./images/roc_auc_compare.png)
- ### Models
  We have trained two xgboost models using selected features, one using hyper-parameters from part 5.2, one with newly tuned hyperparameters, to note that using selected features, the hyperparameter tuning process is nearly twice faster than using all features.
  - ### Model without hyperparameters tuning
      [Comet.ml Experiment](https://www.comet.com/hfereidouni/ift6758/f0d489c19d4b471288372809a4cd8ff6?experiment-tab=panels&showOutliers=true&smoothing=0&xAxis=step)
      In this part, a xgboost classifier that is trained with feature selection (no hyperparameters tuned) has been introduced: 
      ### ROC AUC figure
      ![ROC AUC figure](./images/roc_auc_5.3_1.png)
      ### Goal rate vs Shot probability percentile
      ![Goal Rate](./images/goal_rate_5.3_1.png)
      ### Cumulative % of Goals vs Shot probability percentile
      ![Cumulative sum](./images/cumsum_5.3_1.png)
      ### Calibration Curve
      ![Cali plot](./images/cali_plot_5.3_1.png)
  - ### Model with hyperparameters tuning(Best)
      [Comet.ml Experiment](https://www.comet.com/hfereidouni/ift6758/caa6b53d724c41c9a001be6440449b52?experiment-tab=panels&showOutliers=true&smoothing=0&xAxis=step)
      In this part, a xgboost classifier that is trained with feature selection (no hyperparameters tuned) has been introduced: 
      ### ROC AUC figure
      ![ROC AUC figure](./images/roc_auc_5.3_2.png)
      ### Goal rate vs Shot probability percentile
      ![Goal Rate](./images/goal_rate_5.3_2.png)
      ### Cumulative % of Goals vs Shot probability percentile
      ![Cumulative sum](./images/cumsum_5.3_2.png)
      ### Calibration Curve
      ![Cali plot](./images/cali_plot_5.3_2.png)



# Part 6: Give it your best shot!

## Model 1: Decision Tree
#### First we have load our code with all features from part 4, then like in Part 5 we have applied one-hot encoding for String features.
### Step 1: Data Pre-processing
1. Drop `change_shot_angle` column where the majority of datas are NaN, then drop NaN values.
2. Using Variance filter method to filter low variance features using `VarianceThreshold`
### Step 2: Hyperparameters tuning
Hyperparameters to tune:
```'class_weight':[{0:1,1:1},{0:3,1:1},{0:6,1:1},{0:12,1:1},{0:50,1:1},{0:100,1:1},{0:1000,1:1}],
    'criterion':['gini','entropy'],
    'max_depth':[1,5,10,15,20,40],
    'min_samples_split':[2,8,16,32,64],
    'min_samples_leaf':[1,2,4,16,32,64]
```
As we can see there are lots of possible combinations, so we have chosen **Randomized Search** to optimize our Algorithme.
### Step 3: Feature selection
After applying wrapper method and optimizing algorithm, we apply a Wrapper method(RFE) to then reduce the half of the features to 6.

### Model
After all these processes, we have trained a Decision Tree model using the tuned hyperparameters and selected features.
Comparing to a non-tuned Decision Tree that is trained with all datas, the tuned model has a better performance in terms of roc auc score.
### ROC AUC Compare
![ROC_AUC_compare](./images/part%206/ROC_dt_compare.png)
### ROC AUC figure
![ROC_AUC](./images/part%206/ROC_dt_fine.png)
### Goal Rate vs Shot probability percentile
![Goal_rate](./images/part%206/../part%206/goal_rate_dt_fine.png)
### Cumulative % of Goals vs Shot probability percentile
![Cumulative sum](./images/part%206/cumulative_dt_fine.png)
### Calibration Curve
![Cali plot](./images/part%206/cali_plot_dt_fine.png)



## Model 2: Random Forest
#### First we have load our code with all features from part 4, then like in Part 5 we have applied one-hot encoding for String features.
### Step 1: Data Pre-processing
1. Drop rows where NaN values are found. 
### Step 2: Hyperparameters tuning
We have used the following hyper-paramter-value pairs instead of tuning the Hyperparameters:
```
    n_estimators=200,
    random_state=42,
    class_weight="balanced", 
```
Note that we have used ***class_weight*** so that the random forest model accounts for the class imbalance present in the dataset.
### Step 3: Feature selection
Using SelectKBest method along with mutual_info_classifi to filter K best features using 'Mutual Information' with Label/Class as a hueristic. We used the alue of k=10, which means we selected best 10 feeatures based on Mutual information with label.

### Model
After all these processes, we have trained a Random Forest model using the above hyperparameters and selected features.
### ROC AUC figure
![ROC_AUC](./images/part%206/Random_Forest_KBest_MI_ROC_KBest_MI_random_forest_no_tuning.png)
### Goal Rate vs Shot probability percentile
![Goal_rate](./images/part%206/Random_Forest_KBest_MI_goalratepercentile_KBest_MI_random_forest_no_tuning.png)
### Cumulative % of Goals vs Shot probability percentile
![Cumulative sum](./images/part%206/Random_Forest_KBest_MI_cumulativegoalrate_KBest_MI_random_forest_no_tuning.png)
### Calibration Curve
![Cali plot](./images/part%206/Random_Forest_KBest_MI_calibration_KBest_MI_random_forest_no_tuning.png)

## Model 3: Logistic Regression with L1 regulizer

### Step 1: Hyperparameters tuning
Based on our implementation of logistic regression in part 3, we did some feature engineering and added L1 regulizer as an added optimizer in this part.

### Step 2: Feature selection
For this part we have selected fetaures ['game_time','period','x','y','shot_type','last_event_type', 'x_coord_last_event', 'y_coord_last_event', 'Time_from_the_last_event', 'Distance_from_the_last_event', 'Rebound', 'change_shot_angle', 'Speed', 'shot_dist','angle_net'] to train our model.

### Model
After all these processes, we have trained a Logistic Regression model using the above hyperparameters and selected features.

### ROC Curve
![ROC Curve](./images/part%206/Log_Reg_L1_i4.png)

### Goal Rate
![Goal Rate](./images/part%206/Log_Reg_L1_i3.png)

### Cumulative % of goals
![Cumulative % of goals](./images/part%206/Log_Reg_L1_i2.png)

### Reliability Curve
![Reliability Curve](./images/part%206/Log_Reg_L1_i1.png)



### Best Models:

### ROC AUC Comparison
![ROC_AUC_compare](./images/part%206/ROC_dt_fine.png )|![ROC_AUC](./images/part%206/Random_Forest_KBest_MI_ROC_KBest_MI_random_forest_no_tuning.png)|![ROC Curve](./images/part%206/Log_Reg_L1_i4.png) \
By comparing the roc_auc score of the two models: The Decision tree has a slightly better performance than Random Forest.

### Calibration Comparison
![Cali plot](./images/part%206/cali_plot_dt_fine.png)|![Cali plot](./images/part%206/Random_Forest_KBest_MI_calibration_KBest_MI_random_forest_no_tuning.png)|![Reliability Curve](./images/part%206/Log_Reg_L1_i1.png) \
When it comes to reliability curve, Random Forest has a better curve that is more close to the perfectly calibrated line.

### Precision and recall
#### Decision Tree
- precision: 0.229
- recall: 0.002
#### Random Forest
- precision: 0.0359
- recall: 0.0642165342748
#### Logistic Regression
- precision: 0.184
- recall: 0.70

Here we have decided to use **Random Forest** as out best model after comparison:
- Its reliability curve is closer to the perfect classfier one
- Its Precision and recall are more reasonable compared to Decision Tree and Logistic Regression
- Although the AUC-ROC is not as good as the others, but they are very close



# Part 7: Evaluate on Test set
## Random Forest Downloading
Since this model is too big (~200MB), so we decided to keep it in comet.ml.
To download, run:
```python
from comet_ml import API

api = API(api_key="v5q8O8LftZtvOcoXlVM8Ku8fH")

experiment = api.get_model(workspace='hfereidouni',model_name='random_forest_kbest_mi_random_forest_no_tuning')

experiment.download("1.0.0","./models/",expand=True)
```

## Question 1
Test your 5 models on the untouched 2020/21 regular season dataset. In your blogpost, include the four figures described above. Discuss your results and observations on the test set. Do your models perform as well on the test set as you did on your validation set when building your models? Do any models perform better or worse than you expected?

### Answer 1

### ROC/AUC Curve 

![Image](./images/part%207/ROC_REGULAR.png)

### Goal Rate vs probabaility percentile

![Image](./images/part%207/goal_rate_regular.png)

### Cumulative proportion of goals vs probability percentile

![Image](./images/part%207/cumulative_regular.png)

### Reliability curve

![Image](./images/part%207/cali_regular.png)

### Discussion
XGBoost classifier gives the best model performance whereas Logistic regrssion wih angle feature gives the worst model performance. XGBoost is also well calibrated when proportion of goal examples are compared with the class probabilities. 
The models slightly under-perform on the test set than in the validation set while building models. This can be becuase the models trained on the training set may have slightly different data distributions than the test set, which brings generalization into perspective. Random Forest model with Mutual Information (AUC=0.66) based feature selection wasn't expected to perform similar/ under-perform (small difference in AUC) when compared to Logistic regression models (AUC = 0.68) with only one feature. This example helps to understand Occam's razor, that the best model is usually simple in nature. We also see that random forest and logistic regressors give similar curves in Goal rate and Cumulative % of goals graph.
From the reliability curve, we notice that XGBoost and Random Forest are better calibrated with goals in dataset than Logistic Regressors. Also, both XGBoost and Random Forest are able to assign highr class proababilities to goal examples when compared to Logistic regressors, even with the presence of class imbalance in the dataset.

## Question 2
Test your 5 models on the untouched 2019/20 2020/21 playoff games. In your blogpost, include the four figures described above. Discuss your results and observations on this test set. Are there any differences to the regular season test set or do you get similar ‘generalization’ performance?

### Answer 2

### ROC/AUC Curve 

![Image](./images/part%207/ROC_OFF.png)

### Goal Rate vs probabaility percentile

![Image](./images/part%207/goal_rate_off.png)

### Cumulative proportion of goals vs probability percentile

![Image](./images/part%207/cumulative_off.png)

### Reliability curve

![Image](./images/part%207/cali_off.png)

### Discussion
XGBoost still remains the best performing model both in terms of model performance shown by ROC curve and AUC metric & also through the reliability curve, making it a good model to use further for predictions for the task if required.
In the Goal rate graph, we notice that for high model probabilities (left side of the graph), Random Forest and XGBoost show higher goal rates than Logistic regression models, which is diffrent compared to the previous tests done on regular data set. This can be due to the change in data distributions betwen regular and playoff seasons. Therefore, this situation highlights the higher robustness of Random Forest and XGBoost when compared to Logistic regression models.
Another example of such robustness along with handling of class imbalance can be seen in the reliability curve. Here, both Random Forest and XGBoost try to assign/relate  higher probabilties to actual goal exmaples whereas Logistic regression is capable of relating actual goal exmaple in the lower probability region.

### Reference
https://www.geeksforgeeks.org/probability-calibration-curve-in-scikit-learn/ \
https://aiinpractice.com/xgboost-hyperparameter-tuning-with-bayesian-optimization/ \
https://zhuanlan.zhihu.com/p/131216861 \
https://scikit-learn.org/stable/modules/feature_selection.html \
https://www.geeksforgeeks.org/cumulative-percentage-of-a-column-in-pandas-python/