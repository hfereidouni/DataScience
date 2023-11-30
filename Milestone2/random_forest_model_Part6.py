import matplotlib.pyplot as plt

from comet_ml import Experiment
from joblib import dump, load

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, RocCurveDisplay, auc, roc_curve, precision_score, recall_score
from sklearn.calibration import calibration_curve, CalibrationDisplay

from dataloader import *
from baseline_models import *

import argparse
import os


def random_forest_model(X_train, X_valid, Y_train, Y_valid, features_list, BASELINE_FOLDER_PATH, BASELINE_MODEL_DIR, experiment, name_placeholder="train_cols", titlename="train_cols"):

    modelname="Random_Forest"
    # experiment.add_tag(f"{modelname}_{name_placeholder}")
    
    
    clf = RandomForestClassifier(
                                    n_estimators=200,
                                    n_jobs=10,
                                    random_state=42,
                                    class_weight="balanced", #account for class imbalance
                                )
    clf = clf.fit(X_train, Y_train)
    
    #saving model
    model_path = os.path.join(BASELINE_MODEL_DIR, f"{modelname}_{name_placeholder}.joblib")
    # print(model_path)
    
    # dump(clf, model_path)
    # experiment.log_model(f"{modelname}_{name_placeholder}", model_path)
    # experiment.register_model(f"{modelname}_{name_placeholder}", public=True)
    
    #compute accuracy on Validation set
    valid_accuracy = accuracy_score(clf.predict(X_valid), Y_valid)
    print(f"Validation Accuracy: ", valid_accuracy)
    
    #probability of each class
    class_probabilities = clf.predict_proba(X_valid)
    print(f"Class prob: {class_probabilities[0]}" )
    # print(np.array(class_probabilities).shape)

    #AUC score
    fpr, tpr, thresholds = roc_curve(Y_valid.to_numpy(), class_probabilities[:,1])
    AUC_score = auc(fpr, tpr)
    print(f"AUC score: {AUC_score}")

    #calculate importances
    # https://towardsdatascience.com/improving-random-forest-in-python-part-1-893916666cd
    importances = list(clf.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(X_train.columns, importances)]
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    for (feat, importance) in feature_importances:
        print(f"Variable: {feat} Importance: {importance}")
    
    #add metrics to experiment
    metrics ={
                "class_probabilities": class_probabilities,
                "validation_accuracy": valid_accuracy,
                "AUC_score": AUC_score,
             }
    # experiment.log_metrics(metrics, step=1)

    
    #display figures
    #ROC curve
    # Y_pred = clf.predict(X_valid)
    display_roc(class_probabilities, Y_valid, BASELINE_FOLDER_PATH, titlename, modelname, name_placeholder)
    
    # Goal rate v/s Shot probability percentile
    calc_percentile_goal_rate_data_display(class_probabilities, Y_valid, BASELINE_FOLDER_PATH, titlename, modelname, name_placeholder)

    #Calibration Display   
    display_calibration(clf, X_valid, Y_valid, BASELINE_FOLDER_PATH, titlename, modelname, name_placeholder)
    print(f"-----Experiment with {features_list} using Logistic Regression COMPLETED!-------")

    return clf




def random_forest_model_with_FS(X_train, X_valid, Y_train, Y_valid, features_list, BASELINE_FOLDER_PATH, BASELINE_MODEL_DIR, experiment, name_placeholder="train_cols", titlename="train_cols"):

    modelname="Random_Forest"
    experiment.add_tag(f"{modelname}_{name_placeholder}")
    
    # Using KBest Feature Selection with Mutual Information estimation
    from sklearn.feature_selection import mutual_info_classif, SelectKBest
    features_MI = mutual_info_classif(X_train, Y_train, n_neighbors=3, random_state=42)
    print("MI of features: ", list(zip(X_train.columns, features_MI)))
    kbest_FS = SelectKBest(mutual_info_classif, k=10)
    kbest_FS = kbest_FS.fit(X_train, Y_train)
    # X_train = .fit_transform(X_train, Y_train)
    new_features_list = kbest_FS.get_feature_names_out(X_train.columns)
    
    print("Selected new features: ", new_features_list)

    #slecting new features in validation set
    X_train = X_train.loc[:, new_features_list]
    X_valid = X_valid.loc[:, new_features_list]
    print("new X shape: ", X_train.shape)
    
    #training with new slected features
    clf = RandomForestClassifier(
                                    n_estimators=200,
                                    random_state=42,
                                    class_weight="balanced", #account for class imbalance
                                )
    clf = clf.fit(X_train, Y_train)

    #saving model
    model_path = os.path.join(BASELINE_MODEL_DIR, f"{modelname}_{name_placeholder}.joblib")
    # print(model_path)
    
    dump(clf, model_path)
    experiment.log_model(f"{modelname}_{name_placeholder}", model_path)
    experiment.register_model(f"{modelname}_{name_placeholder}", public=True)
    
    #compute accuracy on Validation set
    valid_accuracy = accuracy_score(clf.predict(X_valid), Y_valid)
    print(f"Validation Accuracy: ", valid_accuracy)
    
    #probability of each class
    class_probabilities = clf.predict_proba(X_valid)
    print(f"Class prob: {class_probabilities[0]}" )
    # print(np.array(class_probabilities).shape)

    #AUC score
    fpr, tpr, thresholds = roc_curve(Y_valid.to_numpy(), class_probabilities[:,1])
    AUC_score = auc(fpr, tpr)
    print(f"AUC score: {AUC_score}")

    #precision
    precision = precision_score(clf.predict(X_valid), Y_valid, average="binary")
    print("Precision : ", precision)

    #recall
    recall = recall_score(clf.predict(X_valid), Y_valid, average="binary")
    print("Recall : ", recall)
    
    #calculate importances
    # https://towardsdatascience.com/improving-random-forest-in-python-part-1-893916666cd
    importances = list(clf.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(X_train.columns, importances)]
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    for (feat, importance) in feature_importances:
        print(f"Variable: {feat} Importance: {importance}")
    
    #add metrics to experiment
    metrics ={
                "class_probabilities": class_probabilities,
                "validation_accuracy": valid_accuracy,
                "AUC_score": AUC_score,
                "precision": precision,
                "recall": recall,
             }
    experiment.log_metrics(metrics, step=1)

    
    #display figures
    #ROC curve
    # Y_pred = clf.predict(X_valid)
    display_roc(class_probabilities, Y_valid, BASELINE_FOLDER_PATH, titlename, modelname, name_placeholder, experiment)
    
    # Goal rate v/s Shot probability percentile
    calc_percentile_goal_rate_data_display(class_probabilities, Y_valid, BASELINE_FOLDER_PATH, titlename, modelname, name_placeholder, experiment)

    #Calibration Display   
    display_calibration(clf, X_valid, Y_valid, BASELINE_FOLDER_PATH, titlename, modelname, name_placeholder, experiment)
    print(f"-----Experiment with {features_list} using Logistic Regression COMPLETED!-------")


def experiment_setup(features_for_model, model_code, name_placeholder, titlename, BASELINE_FOLDER_PATH, BASELINE_MODEL_DIR):

    #Start experiment
    import api_key_gen
    api_key_comet = api_key_gen.get_api_key()
    # os.getenv('COMET_ML')
    # print(os.environ['COMET_ML'])
    experiment = Experiment(
    api_key=api_key_comet,
    project_name="ift6758",
    workspace="hfereidouni",
    )
    #Get required data
    X_train, X_valid, Y_train, Y_valid = prepare_train_valid_dataset(features_for_model)

    # Available Models
    models = {'log_reg': log_reg_model,
             'random_classifier': random_uniform_classifier,
              'random_forest': random_forest_model,
              'random_forest_FS': random_forest_model_with_FS,
             }

    # Train Classifier
    models[model_code](X_train,
                       X_valid,
                       Y_train,
                       Y_valid,
                       name_placeholder=name_placeholder,
                       titlename=titlename,
                       features_list=features_for_model,
                       BASELINE_FOLDER_PATH=BASELINE_FOLDER_PATH,
                       BASELINE_MODEL_DIR=BASELINE_MODEL_DIR,
                       experiment=experiment,
                       # experiment=None,
                      )
    


def random_forest_execution():

    #Setup file structures
    BASELINE_FOLDER_NAME = "Baseline_figures_mile_2"
    BASELINE_FOLDER_PATH = os.path.join(os.getcwd(), BASELINE_FOLDER_NAME)
    if os.path.isdir(BASELINE_FOLDER_PATH):
        print("Folder already present")
    else:
        os.makedirs(BASELINE_FOLDER_PATH)
    BASELINE_MODEL_DIR_NAME = "models"
    BASELINE_MODEL_DIR = os.path.join(os.getcwd(), BASELINE_MODEL_DIR_NAME)
    if os.path.isdir(BASELINE_MODEL_DIR):
        print("Folder already present")
    else:
        os.makedirs(BASELINE_MODEL_DIR)


    #Using Random Forest, run the following models
    # features from feat 4
    features_for_model = ['game_time','period','x','y','shot_type','last_event_type',
       'x_coord_last_event', 'y_coord_last_event', 'Time_from_the_last_event',
       'Distance_from_the_last_event', 'Rebound', 'change_shot_angle', 'Speed',
       'shot_dist','angle_net']
    
    # experiment_setup(features_for_model=features_for_model,
    #                  model_code='random_forest',
    #                  name_placeholder="vanilla_random_forest",
    #                  titlename="all features",
    #                  BASELINE_FOLDER_PATH=BASELINE_FOLDER_PATH,
    #                  BASELINE_MODEL_DIR=BASELINE_MODEL_DIR,
    #                 )


    experiment_setup(features_for_model=features_for_model,
                     model_code='random_forest_FS',
                     name_placeholder="KBest_MI_random_forest_no_tuning",
                     titlename="selected features",
                     BASELINE_FOLDER_PATH=BASELINE_FOLDER_PATH,
                     BASELINE_MODEL_DIR=BASELINE_MODEL_DIR,
                    )

if __name__=="__main__":
    random_forest_execution()