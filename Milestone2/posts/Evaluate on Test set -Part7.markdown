# Part 7: Evaluate on Test set

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
