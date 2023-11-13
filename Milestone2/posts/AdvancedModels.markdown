#  Advanced Models

## Question 1:
### Splitting data
We have used a rate of 75/25 training-validation split with our around 30000 datas. Comparing to commonly used 80/20 split, we decide to have a larger validation set to better evaluate our models.

In this part, a xgboost classifier that is trained with only *shot distance* and *shot angle* has been introduced: \

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
    These three methods has respectively lower the dimention of input datas from **36** to **13**,**13** and **9**, the second method has taken significantly more time than other two methods.
    From validating new models trained using selected features from these three methods and tuned hyperparameters from Q2, we have discovered that the features selected using **RFE** has the best performance overall(best auc roc score), but it's only slightly better than Variance Filter method which took way less time and resources, the third method has the worst performance overall. So we decided to move for ward with the features selected from the first method.  \
    ### The features selected:
    `['game_time', 'period', 'x', 'y', 'x_coord_last_event',
       'y_coord_last_event', 'Time_from_the_last_event',
       'Distance_from_the_last_event', 'change_shot_angle', 'Speed',
       'angle_net', 'shot_type_Wrist Shot', 'last_event_type_Faceoff']`

    ### The ROC/AUC curve of comparaison:
    ![ROC AUC figure](./images/roc_auc_compare.png)
- ### Models
  We have trained two xgboost models using selected features, one using hyper-parameters from part 5.2, one with newly tuned hyperparameters:
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
  - ### Model with hyperparameters tuning
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