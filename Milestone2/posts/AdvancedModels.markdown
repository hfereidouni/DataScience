# Part 5: Advanced Models

## Question 1:
### Splitting data
We have used a rate of 75/25 training-validation split with our around 30000 datas. Comparing to commonly used 80/20 split, we decide to have a larger validation set to better evaluate our models.

In this part, a xgboost classifier that is trained with only *shot distance* and *shot angle* has been introduced:

[Comet.ml Experiment](https://www.comet.com/hfereidouni/ift6758/934d6e4884f046b59408de6197dfa32d?experiment-tab=panels&showOutliers=true&smoothing=0&xAxis=step)

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
[Comet.ml Experiment](https://www.comet.com/hfereidouni/ift6758/1fd819b207ce41e4a09e89e74e45c16e?experiment-tab=panels&showOutliers=true&smoothing=0&xAxis=step)

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
      [Comet.ml Experiment](https://www.comet.com/hfereidouni/ift6758/3f2c8e5488b04654881315c1c7dca01d?experiment-tab=panels&showOutliers=true&smoothing=0&xAxis=step)
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
      [Comet.ml Experiment](https://www.comet.com/hfereidouni/ift6758/66775ec9a7af4091b1e29ad6a83b3099?experiment-tab=panels&showOutliers=true&smoothing=0&xAxis=step)
      In this part, a xgboost classifier that is trained with feature selection (no hyperparameters tuned) has been introduced: 
      ### ROC AUC figure
      ![ROC AUC figure](./images/roc_auc_5.3_2.png)
      ### Goal rate vs Shot probability percentile
      ![Goal Rate](./images/goal_rate_5.3_2.png)
      ### Cumulative % of Goals vs Shot probability percentile
      ![Cumulative sum](./images/cumsum_5.3_2.png)
      ### Calibration Curve
      ![Cali plot](./images/cali_plot_5.3_2.png)

### References:
https://aiinpractice.com/xgboost-hyperparameter-tuning-with-bayesian-optimization/ \
https://zhuanlan.zhihu.com/p/131216861 \
https://scikit-learn.org/stable/modules/feature_selection.html