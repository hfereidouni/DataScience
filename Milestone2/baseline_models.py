import matplotlib.pyplot as plt

from comet_ml import Experiment
from joblib import dump, load

from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, RocCurveDisplay, auc, roc_curve, precision_score, recall_score
from sklearn.calibration import calibration_curve, CalibrationDisplay

from dataloader import *
import argparse
import os




def calc_percentile_goal_rate_data_display(class_probabilities, Y_valid, BASELINE_FOLDER_PATH, titlename, modelname, name_placeholder, experiment):

    #code from Task 6
    probas_and_label = pd.DataFrame(Y_valid.copy())
    probas_and_label['goal_proba']=class_probabilities[:,1]
    probas_and_label = probas_and_label.sort_values(by='goal_proba', ascending=False)
    probas_and_label['#goal+#shot'] = range(1,len(probas_and_label)+1)

    shot_and_goal = np.array(probas_and_label.loc[:,'is_goal'])

    #cumulative sum of goals based on percentile
    goal = np.zeros(len(shot_and_goal))
    for i in range(len(shot_and_goal)):
        goal[i] = shot_and_goal[:i].sum()
    probas_and_label['#goal'] =  goal

    #adding other params
    probas_and_label['#goal/#goal+#shot'] = 100*probas_and_label['#goal']/probas_and_label['#goal+#shot']
    probas_and_label['rank'] = probas_and_label['goal_proba'].rank(pct=True)*100
    probas_and_label['goal_prob_sum'] = probas_and_label['#goal/#goal+#shot'].sum()
    probas_and_label['goal_prob_cumulative_sum'] = probas_and_label['#goal/#goal+#shot'].cumsum()
    probas_and_label['cum_percent'] = 100*(probas_and_label['goal_prob_cumulative_sum'] / probas_and_label['goal_prob_sum'])
    
    plot_x_values = probas_and_label['rank'].iloc[int(len(probas_and_label)*0.01):]
    plot_y_values = probas_and_label['#goal/#goal+#shot'].iloc[int(len(probas_and_label)*0.01):]
    
    goal_rate = plt.plot(plot_x_values,plot_y_values,label=name_placeholder)
    
    plt.plot(plot_x_values,plot_y_values,label=name_placeholder)
    plt.xlim([105,-5])
    plt.ylim([-5,105])
    plt.grid(True)
    plt.ylabel("Goals/(Shots+Goals) (%)")
    plt.xlabel("Shot probability model percentile")
    plt.title(f"Goal Rate for {titlename} with {modelname}")
    plt.legend()
    plt.savefig(os.path.join(BASELINE_FOLDER_PATH, f"{modelname}_goalratepercentile_{name_placeholder}.png"))
    experiment.log_figure(figure_name=f"{modelname}_goalratepercentile_{name_placeholder}")
    plt.close()
    print("Saving goal rate vs shot percentile curve!")

    cum_goal_rate = plt.plot(probas_and_label['rank'],probas_and_label['cum_percent'],label=name_placeholder)
    
    plt.plot(probas_and_label['rank'],probas_and_label['cum_percent'],label=name_placeholder)
    plt.xlim([105,-5])
    plt.ylim([-5,105])
    plt.grid(True)
    plt.ylabel("Proportion (%)")
    plt.xlabel("Shot probability model percentile")
    plt.title(f"Cumulative % of goals for {titlename} with {modelname}")
    plt.legend()
    plt.savefig(os.path.join(BASELINE_FOLDER_PATH, f"{modelname}_cumulativegoalrate_{name_placeholder}.png"))
    experiment.log_figure(figure_name=f"{modelname}_cumulativegoalrate_{name_placeholder}")
    plt.close()
    print("Saving cumulative goal rate vs shot percentile curve!")

    return goal_rate, cum_goal_rate


def display_calibration(clf, X_valid, Y_valid, BASELINE_FOLDER_PATH, titlename, modelname, name_placeholder, experiment):

    # print(np.bincount(Y_valid.to_numpy()), class_probabilities[:,1])
    # prob_true, prob_pred = calibration_curve(Y_valid.to_numpy(), class_probabilities[:,1])
    # print(prob_true, "----\n", prob_pred)
    # disp = CalibrationDisplay.from_predictions(Y_valid.to_numpy(), class_probabilities[:,1])
    displ = CalibrationDisplay.from_estimator(clf, X_valid, Y_valid, n_bins=15)
    plt.title(f"Reliability curve for {titlename} with {modelname}")
    plt.grid(True)
    plt.show()
    plt.savefig(os.path.join(BASELINE_FOLDER_PATH, f"{modelname}_calibration_{name_placeholder}.png"))
    experiment.log_figure(figure_name=f"{modelname}_calibration_{name_placeholder}")
    plt.close()
    print("Saving Calibration curve!")

    return displ

def display_roc(class_probabilities, Y_valid, BASELINE_FOLDER_PATH, titlename, modelname, name_placeholder, experiment):

    fpr,tpr,threshold = roc_curve(Y_valid,class_probabilities[:,1])

    auc_score = auc(fpr,tpr)
    auc_score = round(auc_score, 2)
    
    # RocCurveDisplay.from_predictions(Y_valid.to_numpy(),
                                     # Y_pred,
                                    # name=f"{name_placeholder}")
    roc_plot = plt.plot(fpr,tpr,'b',label=f"{name_placeholder}, AUC ={auc_score}")
    plt.plot(fpr,tpr,'b',label=f"{name_placeholder}, AUC ={auc_score}")
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve for {modelname} with {titlename}")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(os.path.join(BASELINE_FOLDER_PATH, f"{modelname}_ROC_{name_placeholder}.png"))
    experiment.log_figure(figure_name=f"{modelname}_ROC_{name_placeholder}")
    print("Saving ROC curve!")
    plt.close()

    return roc_plot


def log_reg_model(X_train, X_valid, Y_train, Y_valid, features_list, BASELINE_FOLDER_PATH, BASELINE_MODEL_DIR, experiment, name_placeholder="train_cols", titlename="train_cols"):

    modelname="Log_Reg"
    experiment.add_tag(f"{modelname}_{name_placeholder}")
    
    
    clf = LogisticRegression()
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
    Y_pred = clf.decision_function(X_valid)
    roc_plot = display_roc(class_probabilities, Y_valid, BASELINE_FOLDER_PATH, titlename, modelname, name_placeholder, experiment)
    
    # Goal rate v/s Shot probability percentile
    goal_rate_plot, cum_goal_rate_plot = calc_percentile_goal_rate_data_display(class_probabilities, Y_valid, BASELINE_FOLDER_PATH, titlename, modelname, name_placeholder, experiment)

    #Calibration Display   
    cali_plot = display_calibration(clf, X_valid, Y_valid, BASELINE_FOLDER_PATH, titlename, modelname, name_placeholder, experiment)
    print(f"-----Experiment with {features_list} using Logistic Regression COMPLETED!-------")

    return roc_plot, goal_rate_plot, cum_goal_rate_plot, cali_plot, AUC_score


def random_uniform_classifier(X_train, X_valid, Y_train, Y_valid, features_list, BASELINE_FOLDER_PATH, BASELINE_MODEL_DIR, experiment, name_placeholder="train_cols", titlename="train_cols"):

    modelname="Random_classifer"
    experiment.add_tag(modelname)
    
    #set seed
    # np.random.seed(42)

    dummy_clf = DummyClassifier(strategy="uniform")
    dummy_clf = dummy_clf.fit(X_train, Y_train)
    
    Y_pred = dummy_clf.predict(X_valid)

    #saving model
    model_path = os.path.join(BASELINE_MODEL_DIR, f"{modelname}_{name_placeholder}.joblib")

    dump(dummy_clf, model_path)
    experiment.log_model(f"{modelname}_{name_placeholder}", model_path)
    experiment.register_model(f"{modelname}_{name_placeholder}", public=True)

    
    #compute accuracy on Validation set
    valid_accuracy = accuracy_score(Y_pred, Y_valid)
    print(f"Validation Accuracy: ", valid_accuracy)

    #probability of each class
    class_probabilities = dummy_clf.predict_proba(X_valid)
    
    #AUC score
    fpr, tpr, thresholds = roc_curve(Y_valid.to_numpy(), class_probabilities[:,1])
    AUC_score = auc(fpr, tpr)
    print(f"AUC score: {AUC_score}")

    #precision
    precision = precision_score(dummy_clf.predict(X_valid), Y_valid, average="binary")
    print("Precision : ", precision)

    #recall
    recall = recall_score(dummy_clf.predict(X_valid), Y_valid, average="binary")
    print("Recall : ", recall)
    
    #add metrics to experiment
    metrics ={
                "class_probabilities": class_probabilities,
                "validation_accuracy": valid_accuracy,
                "AUC_score": AUC_score,
                "precision": precision,
                "recall": recall,
             }
    experiment.log_metrics(metrics, step=1)
    
    #ROC curve
    roc_plot = display_roc(class_probabilities, Y_valid, BASELINE_FOLDER_PATH, titlename, modelname, name_placeholder, experiment)
    
    # Goal rate v/s Shot probability percentile
    goal_rate_plot, cum_goal_rate_plot = calc_percentile_goal_rate_data_display(class_probabilities, Y_valid, BASELINE_FOLDER_PATH, titlename, modelname, name_placeholder, experiment)

    #Calibration Display   
    cali_plot = display_calibration(dummy_clf, X_valid, Y_valid, BASELINE_FOLDER_PATH, titlename, modelname, name_placeholder, experiment)
    print(f"-----Experiment using Random Classifier COMPLETED!-------")

    return roc_plot, goal_rate_plot, cum_goal_rate_plot, cali_plot, AUC_score


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
    print("Final features: ", X_train.columns)
    
    # Available Models
    models = {'log_reg': log_reg_model,
             'random_classifier': random_uniform_classifier,
             }

    # Train Classifier
    roc_plot, goal_rate_plot, cum_goal_rate_plot, cali_plot, auc = models[model_code](X_train,
                       X_valid,
                       Y_train,
                       Y_valid,
                       name_placeholder=name_placeholder,
                       titlename=titlename,
                       features_list=features_for_model,
                       BASELINE_FOLDER_PATH=BASELINE_FOLDER_PATH,
                       BASELINE_MODEL_DIR=BASELINE_MODEL_DIR,
                       experiment=experiment,
                      )

    # plotting graphs
    # get x y data from plt objects
    # Reference  : https://stackoverflow.com/questions/20130768/retrieve-xy-data-from-matplotlib-figure
    
    roc__dist_x, roc__dist_y = roc_plot[0].get_data()
    grate_dist_x, grate_dist_y = goal_rate_plot[0].get_data()
    cugrate_dist_x, cugrate_dist_y = cum_goal_rate_plot[0].get_data()
    cali_dist_x, cali_dist_y = cali_plot.line_.get_data()

    #roc curve
    plt.figure(figsize=(12,8))
    
    plt.subplot(2,2,1)
    plt.plot(roc__dist_x,roc__dist_y,'b',label=f"{name_placeholder}, AUC={auc}")
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve")
    plt.grid(True)
    plt.legend(loc="best")

    #goal rate curve
    plt.subplot(2,2,2)
    plt.plot(grate_dist_x,grate_dist_y,'b', label=f"{name_placeholder}")
    plt.xlim([105,-5])
    plt.ylim([-5,105])
    plt.grid(True)
    plt.ylabel("Goals/(Shots+Goals) (%)")
    plt.xlabel("Shot probability model percentile")
    plt.title(f"Goal Rates")
    plt.legend(loc="best")

    #cumulative goal rate curve
    plt.subplot(2,2,3)
    plt.plot(cugrate_dist_x,cugrate_dist_y,label=f"{name_placeholder}")
    plt.xlim([105,-5])
    plt.ylim([-5,105])
    plt.grid(True)
    plt.ylabel("Proportion (%)")
    plt.xlabel("Shot probability model percentile")
    plt.title(f"Cumulative % of goals")
    plt.legend(loc="best")

    #calibration curve
    plt.subplot(2,2,4)
    plt.plot(cali_dist_x,cali_dist_y, 'b-s', label=f"{name_placeholder}")
    plt.plot([0, 1], [0, 1],'k--', label="Perfectly Calibrated")
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel("Fraction of positives (Positive class: 1)")
    plt.xlabel("Mean predicted probability (Positive class: 1)")
    plt.title(f"Reliability curve ")
    plt.grid(True)
    plt.legend(loc="best")

    plt.tight_layout()
    plt.suptitle(f"CURVES using {model_code}")
    plt.savefig(os.path.join(BASELINE_FOLDER_PATH, f"{name_placeholder}_per-experiment-graph.png"))
    experiment.log_figure(figure_name=f"{model_code}_allgraphs_{name_placeholder}")
    plt.close()
    print("Saving all analysis graph!")
    
    


def run_experiments():

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


    #Using Logistic regression, run the following models
    #Exp 1
    features_for_model = ['shot_dist']
    experiment_setup(features_for_model=features_for_model,
                     model_code='log_reg',
                     name_placeholder="shot_dist_only",
                     titlename="Shot_Distances only",
                     BASELINE_FOLDER_PATH=BASELINE_FOLDER_PATH,
                     BASELINE_MODEL_DIR=BASELINE_MODEL_DIR,
                    )

    # Exp 2
    features_for_model = ['angle_net']
    experiment_setup(features_for_model=features_for_model,
                     model_code='log_reg',
                     name_placeholder="angle_only",
                     titlename="Shot_Angle only",
                     BASELINE_FOLDER_PATH=BASELINE_FOLDER_PATH,
                     BASELINE_MODEL_DIR=BASELINE_MODEL_DIR,
                    )


    # Exp 3
    features_for_model = ['shot_dist', 'angle_net']
    experiment_setup(features_for_model=features_for_model,
                     model_code='log_reg',
                     name_placeholder="shot_dist_and_angle",
                     titlename="Shot Distance & Angle",
                     BASELINE_FOLDER_PATH=BASELINE_FOLDER_PATH,
                     BASELINE_MODEL_DIR=BASELINE_MODEL_DIR,
                    )

    #Exp 4
    #use dummy classifer
    features_for_model = []
    experiment_setup(features_for_model=features_for_model,
                     model_code='random_classifier',
                     name_placeholder="random_classifier",
                     titlename="no features",
                     BASELINE_FOLDER_PATH=BASELINE_FOLDER_PATH,
                     BASELINE_MODEL_DIR=BASELINE_MODEL_DIR,
                    )


if __name__=="__main__":
    run_experiments()
