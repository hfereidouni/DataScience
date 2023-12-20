"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import os
from pathlib import Path
import logging
from flask import Flask, jsonify, request, abort
import sklearn
import pandas as pd
import joblib
from comet_ml import API

LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")
MODEL_DIR = "./models/"
app = Flask(__name__)

#helper class
class CometClient(object):
    def __init__(self, api_key, work_place,valid_models,newest_model_version):
        self.api_key = api_key
        self.work_place = work_place
        self.valid_models = valid_models
        self.newest_model_version = newest_model_version
        self.default_model = self.valid_models[0]
        
    def model_exist(self,model_name):
        if(model_name not in list(map(str.lower,self.valid_models))):
            return False
        return True

    def download_model(self,model_name,version,output_folder):
        api = API(self.api_key)
        model = api.get_model(workspace=self.work_place, model_name=model_name)
        model.download(version=version,output_folder=output_folder,expand=True)

API_KEY = "v5q8O8LftZtvOcoXlVM8Ku8fH" #os.environ.get("COMET_API_KEY") if get from env var
WORK_PLACE = "hfereidouni"
COMET_CLIENT = None # will be initialized at @app.before_first_request
LOADED_MODEL = None # will be initialized at @app.before_first_request

# helper functions
def load_model(model_path):
    global LOADED_MODEL
    model = joblib.load(model_path)
    LOADED_MODEL = model

def model_file_exist(model_path):
    if os.path.isfile(model_path):
        return True
    return False

@app.route("/check", methods=["GET"])
def check():
    if LOADED_MODEL != None:
        return jsonify(LOADED_MODEL.feature_names_in_.tolist())
    else:
        return jsonify("No model is loaded yet"),400


@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    #setup basic logging configuration
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)

    #initialization: CometClient and Loading default model
    global COMET_CLIENT

    COMET_CLIENT = CometClient(api_key=API_KEY, work_place = WORK_PLACE,
                           valid_models=["Log_Reg_shot_dist_only",
                                        "Log_Reg_shot_dist_and_angle"],
                            newest_model_version={"Log_Reg_shot_dist_only":"1.17.0",
                                                     "Log_Reg_shot_dist_and_angle":"1.9.0"})

    default_model = COMET_CLIENT.default_model
    default_path = MODEL_DIR+default_model+'.joblib'
    print(f"def model: {default_model}")
    print(f"def path: {default_path} {os.path.exists(default_path)} ")
    versions = COMET_CLIENT.newest_model_version
    
    # download default model
    if not model_file_exist(default_path):
        try:
            app.logger.info(f'Default model not found locally, download started...')
            COMET_CLIENT.download_model(default_model,version=versions[default_model],output_folder=MODEL_DIR)
            app.logger.info(f'Default model downloaded, {default_model} version {versions[default_model]}')
        #download failed
        except Exception:
            exception = f"Default model failed downloading, the loaded model is None for now"
            abort(400,exception)
    load_model(default_path)
    app.logger.info(f'Default model is loaded as {default_model}')

@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    
    try:
        text = Path('flask.log').read_text()
    except OSError:
        exception = "Log read failed"
        app.logger.exception(exception)
        return jsonify(exception),400

    response = text.splitlines()
    app.logger.info("Logs read with success.")
    return jsonify(response)


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    input jason looks like:

        request = {
            "model_name":"name",
           "version":"version"
           }
    
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    #read from request
    model_name = json["model_name"]
    version = json["version"]

    model_path = MODEL_DIR+model_name+'.joblib'
    #check if the model exists in comet library
    if not COMET_CLIENT.model_exist(model_name):
        exception = f"{model_name} is not a valide model"
        return jsonify(exception), 400
    else:
        #check to see if the model you are querying for is already downloaded
        if model_file_exist(model_path):
        #if yes, load that model and write to the log about the model change.  
            load_model(model_path)
            response = f"Loaded Model has changed to {model_name}"
            app.logger.info(response)
        else:
        #download model
            try:
                COMET_CLIENT.download_model(model_name,version=version,output_folder=MODEL_DIR)
                #saving successfully
                if model_file_exist(model_path):
                    response = f"{model_name} downloaded and loaded as the current model."
                    app.logger.info(response)
                    load_model(model_path)

                #saving failed
                else:
                    exception = f"{model_name} failed saving to local, the loaded model remains."
                    return jsonify(exception),400
            #download fail
            except Exception:
                exception = f"{model_name} failed downloading, the loaded model remains."
                return jsonify(exception),400
    
    return jsonify(response)  # response must be json serializable!


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    # Make sure the features are filtered before sending request
    X_filtered = pd.read_json(json)
    
    try:
        probas = LOADED_MODEL.predict_proba(X_filtered)
        probas_df =pd.DataFrame(probas[:,1],columns=['chance of goal'])

        app.logger.info("Predicted successfully.")
        return jsonify(probas_df.to_json())
    
    except Exception:
        return jsonify("Prediction failed."),400

# Use gunicorn/waitress
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=os.environ.get("SERVING_PORT", 8080))
