import matplotlib.pyplot as plt

from comet_ml import Experiment
from joblib import dump, load

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, RocCurveDisplay, auc, roc_curve
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score




from dataloader import *
import argparse
import numpy as np
import os




def calc_percentile_goal_rate_data_display(class_probabilities, Y_valid, BASELINE_FOLDER_PATH, titlename, modelname, name_placeholder):

    #code from Task 6
    probas_and_label = pd.DataFrame(Y_valid.copy())
    probas_and_label['goal_proba']=class_probabilities[:,1]
    # print(probas_and_label.columns)
    probas_and_label = probas_and_label.sort_values(by='goal_proba', ascending=False)
    # goal+shot = count(shot[no goal] or goal) 
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
    # print("probas shape", probas_and_label.shape)
    # print(probas_and_label[['rank', '#goal/#goal+#shot']].head())
    
    plot_x_values = probas_and_label['rank'].iloc[int(len(probas_and_label)*0.01):]
    plot_y_values = probas_and_label['#goal/#goal+#shot'].iloc[int(len(probas_and_label)*0.01):]
    # print(f"plot values\nx: {plot_x_values}\ny: {plot_y_values}")
    
    plt.plot(plot_x_values,plot_y_values,label=name_placeholder)
    plt.xlim([105,-5])
    plt.ylim([-5,105])
    plt.grid(True)
    plt.ylabel("Goals/(Shots+Goals) (%)")
    plt.xlabel("Shot probability model percentile")
    plt.title(f"Goal Rate for {titlename} with {modelname}")
    plt.legend()
    plt.savefig(os.path.join(BASELINE_FOLDER_PATH, f"{modelname}_goalratepercentile_{name_placeholder}.png"))
    plt.close()
    print("Saving goal rate vs shot percentile curve!")

    plt.plot(probas_and_label['rank'],probas_and_label['cum_percent'],label=name_placeholder)
    plt.xlim([105,-5])
    plt.ylim([-5,105])
    plt.grid(True)
    plt.ylabel("Proportion (%)")
    plt.xlabel("Shot probability model percentile")
    plt.title(f"Cumulative % of goals for {titlename} with {modelname}")
    plt.legend()
    plt.savefig(os.path.join(BASELINE_FOLDER_PATH, f"{modelname}_cumulativegoalrate_{name_placeholder}.png"))
    plt.close()
    print("Saving cumulative goal rate vs shot percentile curve!")




def display_calibration(class_probabilities, Y_valid, BASELINE_FOLDER_PATH, titlename, modelname, name_placeholder):

    # print(np.bincount(Y_valid.to_numpy()), class_probabilities[:,1])
    # prob_true, prob_pred = calibration_curve(Y_valid.to_numpy(), class_probabilities[:,1])
    # print(prob_true, "----\n", prob_pred)
    disp = CalibrationDisplay.from_predictions(Y_valid.to_numpy(), class_probabilities[:,1])
    plt.title(f"Reliability curve for {titlename} with {modelname}")
    plt.grid(True)
    plt.show()
    plt.savefig(os.path.join(BASELINE_FOLDER_PATH, f"{modelname}_calibration_{name_placeholder}.png"))
    plt.close()
    print("Saving Calibration curve!")


def display_roc(Y_pred, Y_valid, BASELINE_FOLDER_PATH, titlename, modelname, name_placeholder):

    

    RocCurveDisplay.from_predictions(Y_valid.to_numpy(),
                                     Y_pred,
                                    name=f"{name_placeholder}")
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve for {modelname} with {titlename}")
    plt.grid(True)
    plt.show()
    plt.savefig(os.path.join(BASELINE_FOLDER_PATH, f"{modelname}_ROC_{name_placeholder}.png"))
    print("Saving ROC curve!")
    plt.close()



def log_reg_model(X_train, X_valid, Y_train, Y_valid, features_list, BASELINE_FOLDER_PATH, BASELINE_MODEL_DIR, experiment, name_placeholder="train_cols", titlename="train_cols"):

    modelname="Log_Reg_L1"
    experiment.add_tag(f"{modelname}_{name_placeholder}")
    
    
    clf = LogisticRegression(penalty= 'l1', solver= 'saga')
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

    Y_pred = clf.predict(X_valid)
    precision_dt = precision_score(Y_valid, Y_pred)

    recall_dt = recall_score(Y_valid, clf.predict(X_valid))



    #AUC score
    fpr, tpr, thresholds = roc_curve(Y_valid.to_numpy(), class_probabilities[:,1])
    AUC_score = auc(fpr, tpr)
    roc_dt = roc_auc_score(Y_valid.to_numpy(), Y_pred)

    print(f"AUC score: {AUC_score}")

    #add metrics to experiment
    metrics ={
                "class_probabilities": class_probabilities,
                "validation_accuracy": valid_accuracy,
                "AUC_score": AUC_score,
                "precision_dt": precision_dt,
                "recall_dt": recall_dt,
                "roc_dt": roc_dt
             }
    experiment.log_metrics(metrics, step=1)

    
    #display figures
    #ROC curve
    Y_pred = clf.decision_function(X_valid)
    display_roc(Y_pred, Y_valid, BASELINE_FOLDER_PATH, titlename, modelname, name_placeholder)
    
    # Goal rate v/s Shot probability percentile
    calc_percentile_goal_rate_data_display(class_probabilities, Y_valid, BASELINE_FOLDER_PATH, titlename, modelname, name_placeholder)

    #Calibration Display   
    display_calibration(class_probabilities, Y_valid, BASELINE_FOLDER_PATH, titlename, modelname, name_placeholder)
    print(f"-----Experiment with {features_list} using Logistic Regression COMPLETED!-------")


def random_uniform_classifier(X_train, X_valid, Y_train, Y_valid, features_list, BASELINE_FOLDER_PATH, BASELINE_MODEL_DIR, experiment, name_placeholder="train_cols", titlename="train_cols"):

    modelname="Random_classifer"
    experiment.add_tag(modelname)
    
    #set seed
    np.random.seed(42)

    class_probabilities_1 = np.random.uniform(low=0, high=1, size=len(Y_valid)).reshape(-1,1)
    class_probabilities_0 = 1.0 - class_probabilities_1
    # print(class_probabilities_0.shape)
    class_probabilities = np.hstack((class_probabilities_0, class_probabilities_1))
    # print(class_probabilities.shape)

    Y_pred = np.round(class_probabilities_1)
    
    #compute accuracy on Validation set
    valid_accuracy = accuracy_score(Y_pred, Y_valid)
    print(f"Validation Accuracy: ", valid_accuracy)

    #probability of each class
    # class_probabilities = clf.predict_proba(X_valid)
    print(f"Class prob: {class_probabilities[0]}" )

    #AUC score
    fpr, tpr, thresholds = roc_curve(Y_valid.to_numpy(), class_probabilities[:,1])
    AUC_score = auc(fpr, tpr)
    print(f"AUC score: {AUC_score}")

    #add metrics to experiment
    metrics ={
                "class_probabilities": class_probabilities,
                "validation_accuracy": valid_accuracy,
                "AUC_score": AUC_score,
             }
    experiment.log_metrics(metrics, step=1)
    
    #ROC curve
    display_roc(Y_pred, Y_valid, BASELINE_FOLDER_PATH, titlename, modelname, name_placeholder)
    
    # Goal rate v/s Shot probability percentile
    calc_percentile_goal_rate_data_display(class_probabilities, Y_valid, BASELINE_FOLDER_PATH, titlename, modelname, name_placeholder)

    #Calibration Display   
    display_calibration(class_probabilities, Y_valid, BASELINE_FOLDER_PATH, titlename, modelname, name_placeholder)
    print(f"-----Experiment using Random Classifier COMPLETED!-------")


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
    models = {'log_reg_l1': log_reg_model,
             'random_classifier': random_uniform_classifier,
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
                      )

    


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


    # Using Logistic regression, run the following models
    # Exp 1
    features_for_model = ['shot_dist']
    experiment_setup(features_for_model=features_for_model,
                     model_code='log_reg_l1',
                     name_placeholder="allfeatures",
                     titlename="ALLFEATURES",
                     BASELINE_FOLDER_PATH=BASELINE_FOLDER_PATH,
                     BASELINE_MODEL_DIR=BASELINE_MODEL_DIR,
                    )

    # Exp 2
    features_for_model = ['angle_net']
    experiment_setup(features_for_model=features_for_model,
                     model_code='log_reg_l1',
                     name_placeholder="allfeatures",
                     titlename="ALLFEATURES",
                     BASELINE_FOLDER_PATH=BASELINE_FOLDER_PATH,
                     BASELINE_MODEL_DIR=BASELINE_MODEL_DIR,
                    )

    # Exp 3
    features_for_model = ['shot_dist', 'angle_net']
    experiment_setup(features_for_model=features_for_model,
                     model_code='log_reg_l1',
                     name_placeholder="allfeatures",
                     titlename="ALLFEATURES",
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