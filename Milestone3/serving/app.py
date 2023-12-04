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


# import ift6758


LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")
MODEL_DIR = "./models/"
API_KEY = "v5q8O8LftZtvOcoXlVM8Ku8fH"
WORK_SPACE = "hfereidouni"
VALID_MODELS = [
    "Log_reg_shot_dist_only",
    "Log_reg_shot_dist_and_angle",
]
DEFAULT_MODEL = VALID_MODELS[0]
LOADED_MODEL = None

app = Flask(__name__)

def download_model(api_key,work_space,model_name,version,output_folder):
    api = API(api_key)
    model = api.get_model(workspace=work_space, model_name=model_name)
    model.download(version=version,output_folder=output_folder,expand=True)

def load_model(model_path):
    model = joblib.load(model_path)
    LOADED_MODEL = model

@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    # TODO: setup basic logging configuration
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)

    # TODO: any other initialization before the first request (e.g. load default model)
    pass
    # load_model(DEFAULT_MODEL)


@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    
    # # TODO: read the log file specified and return the data
    # raise NotImplementedError("TODO: implement this endpoint")

    # response = "hihi"
    # return jsonify(response)  # response must be json serializable!
    try:
        f = open('flask.log','r')
    except OSError:
        app.logger.exception("Log file not found")

    response = f.read()
    app.logger.info("logs read.")
    return jsonify(response)




@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }
    
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    model_name = json["model_name"]
    version = json["version"]

    model_path = MODEL_DIR+model_name+'.joblib'
    # TODO: check to see if the model you are querying for is already downloaded
    if os.path.isfile(model_path):
    # TODO: if yes, load that model and write to the log about the model change.  
    # eg: app.logger.info(<LOG STRING>)
        app.logger.info(f"Loaded Model has changed from {LOADED_MODEL} to {model_name}")
        load_model(model_path)
    else:
        response = "No"
    # TODO: if no, try downloading the model: if it succeeds, load that model and write to the log
    # about the model change. If it fails, write to the log about the failure and keep the 
    # currently loaded model

    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here

    # raise NotImplementedError("TODO: implement this endpoint")

    # response = MODEL_DIR

    app.logger.info(response)
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

    # # TODO:
    # raise NotImplementedError("TODO: implement this enpdoint")
    
    # response = None

    # app.logger.info(response)
    # return jsonify(response)  # response must be json serializable!
    return jsonify(json)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=os.environ.get("SERVING_PORT", 8080))