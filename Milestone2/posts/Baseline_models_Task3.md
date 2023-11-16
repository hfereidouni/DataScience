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

![Image](Baseline_figures_mile_2/Log_Reg_ROC_shot_dist_only.png)

**Goal Rate v/s Shot Probability percentile**

![Image](Baseline_figures_mile_2/Log_Reg_goalratepercentile_shot_dist_only.png)

**Cumulative Goal Rate v/s Shot Probability percentile**

![Image](Baseline_figures_mile_2/Log_Reg_cumulativegoalrate_shot_dist_only.png)

**Reliability Curve**

![Image](Baseline_figures_mile_2/Log_Reg_calibration_shot_dist_only.png)

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
![Image](Baseline_figures_mile_2/shot_dist_only_per-experiment-graph.png)

**Discussion**

The Logistic regresion classifier for "shot distances only" works well above the random classifier. However, from the "Goal Rate graph" and "Cumulative % of goals" we notice that even the higher shot probability percentiles (which means probability of goal being high) don't result in higher goal rates, indicating the presence of more #no_goals examples than #goal examples. The calibration curve tells us the real proportion of positive predictions for a given calculated probability of a positive class.

Experiment entry link: https://www.comet.com/hfereidouni/ift6758/ee98743fc69e489883e4e6f81a7b836e?assetId=86e8d61ab83a48f29fc8e8dfad3ec9c4&assetPath=models%2CLog_Reg_shot_dist_only&experiment-tab=assetStorage

Model Registry entry: https://www.comet.com/hfereidouni/model-registry/log_reg_shot_dist_only/1.15.0?tab=assets

### **Logistic Regression, trained on angle only**
![Image](Baseline_figures_mile_2/angle_only_per-experiment-graph.png)

**Discussion**

The Logistic regresion classifier for shot angles only works similarly to the random classifier. According to the ROC curve, we see that FPR > TPR till FPR=0.5 and after 0.5 value TPR > FPR. Increased FPR points to low precision model. From the Goal Rate and Cumulative % of goals graphs, we notice that the goal rate aproximately stays the same irrepsective of the shot probability model percentile (for both high and low model probabilities). This implies that the model fails to accumulate/ group high probabilities with goal examples. The calibration curve show that this model is poorly calibrated with actual data.

Experiment entry link: https://www.comet.com/api/experiment/redirect?assetId=31c269d20c2449828cc40d37767e7fe8&assetPath=models,Log_Reg_angle_only&experiment-tab=assetStorage&experimentKey=46d8c6c8f29244018fc7adb0aa290e3f

Model Registry link: https://www.comet.com/hfereidouni/model-registry/log_reg_angle_only/1.9.0?tab=assets


## **Logistic Regression, trained on both distance and angle**
![Image](Baseline_figures_mile_2/shot_dist_and_angle_per-experiment-graph.png)

**Discussion**

The Logistic regresion classifier for both shot angle and distance features works well above to the random classifier. Similar to "shot distances only" classifier, we notice the presence of class imbalance from "Goal Rate" and "Cumulative % of goals" graphs. We also see that a larger proportion of positive class population (#goal) lies within a small range of model predicted probabilities, similar to "shot ditance only" Logistic Rgression classifier.


Experiment Entry Link: https://www.comet.com/api/experiment/redirect?assetId=1540e95509844ca8a533c7dd025c8500&assetPath=models,Log_Reg_shot_dist_and_angle&experiment-tab=assetStorage&experimentKey=335a9643c0cd42288290f1c02de30661

Model Registry link: https://www.comet.com/hfereidouni/model-registry/log_reg_shot_dist_and_angle/1.7.0?tab=assets


## **Random baseline: rather than training a classifier, the predicted probability is sampled from a uniform distribution**
![Image](Baseline_figures_mile_2/random_classifier_per-experiment-graph.png)

**Discussion**

The Random classifier follows uniform distribution and hence gives AUC of 0.5 . 
From the "Goal Rate" and "Cumulative% of goals" graph, we see that there is no effect on goal rate for any shot probabilities (high or low), which is because the random classifier doesn't differentiate between a high and low model probability (high model probability shoud accumulate more examples of goals).
Furthermore, the reliability curve shows that the random classifier places the same frequency of goals dsitribution for each probability bin. This shows the class imbalance in data (same freq. of goals) and poor calibration of random classifier.


Experiment Entry Link: https://www.comet.com/api/experiment/redirect?assetId=12623ff25c504b37b3c853922ecb4590&assetPath=models,Random_classifer_random_classifier&experiment-tab=assetStorage&experimentKey=38ca0870199140b685b9f5fccb79befb

Model Registry link: https://www.comet.com/hfereidouni/model-registry/random_classifer_random_classifier/1.5.0?tab=assets

