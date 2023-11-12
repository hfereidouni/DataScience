import math
import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from FeatureEngineering_2 import *
from sklearn import metrics
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.calibration import CalibrationDisplay
from sklearn.preprocessing import OneHotEncoder


from joblib import dump, load

def split_data(train_val_data,features,test_size,dropna):
    #drop NAN if necessary
    if dropna:
        train_val_data = train_val_data[train_val_data['shot_dist'].notna() & train_val_data['angle_net'].notna()]

    #Split training and validation data
    train, val = train_test_split(train_val_data, test_size=test_size,random_state=0)

    #Split X and Y
    train_X = train[features]
    val_X =  val[features]
    train_Y = train[['is_goal']]
    val_Y =  val[['is_goal']]

    return train_X,train_Y,val_X,val_Y
    
def roc_auc_plot(val_Y,val_res,model_name,sub_title):
    fpr,tpr,threshold = metrics.roc_curve(val_Y,val_res[:,1])

    auc = metrics.auc(fpr,tpr)

    plt.title(f'ROC/AUC for {model_name} trained on {sub_title}')
    plt.plot(fpr,tpr,'y',label=f"{sub_title}, AUC =%0.2f"%auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.grid(True)
    plt.legend(loc="lower right")

    return fpr,tpr,threshold,auc

def helper_df(val_Y,val_res):
    df = val_Y.copy()
    df['goal_proba']=val_res[:,1]
    df = df.sort_values(by=['goal_proba'], ascending=False)
    df['#goal+#shot'] = range(1,len(df)+1)
    # shot_and_goal = np.array(probas_and_label.loc[:,'is_goal'])

    # Calculate # of goals for each row
    goal = np.zeros(len(df['is_goal']))
    for i in range(len(df['is_goal'])):
        goal[i] = df['is_goal'][:i].sum()
    df['#goal'] =  goal

    df['#goal/#goal+#shot'] = 100*df['#goal']/df['#goal+#shot']

    #get percentile using rank
    df['rank'] = df['goal_proba'].rank(pct=True)*100

    # Calculate cumulative sum and sum for #goal/#goal+#shot
    df['goal_prob_sum'] = df['#goal/#goal+#shot'].sum()
    df['goal_prob_cumulative_sum'] = df['#goal/#goal+#shot'].cumsum()
    df['cum_percent'] = 100*(df['goal_prob_cumulative_sum'] / df['goal_prob_sum'])

    return df

def onehot_generator(column):
    #ref:https://stackoverflow.com/questions/69302224/how-to-one-hot-encode-a-dataframe-column-in-python
    one_hot_encoder = OneHotEncoder()
    values = one_hot_encoder.fit_transform(column)

    df = pd.DataFrame(values.toarray(),columns=one_hot_encoder.get_feature_names_out()).astype(int)
    return df