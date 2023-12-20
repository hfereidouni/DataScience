import json
import requests
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class ServingClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 5000, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        if features is None:
            features = ["distance"]
        self.features = features

        # any other potential initialization

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """

        # Convert DataFrame to JSON
        input_json = X.to_json(orient='records')

        try:
            # Make a POST request to the prediction endpoint
            response = requests.post(f"{self.base_url}/predict", json=json.loads(input_json))
    
            if response.status_code == 200:
                # Parse the JSON response and convert it back to DataFrame
                result_json = response.json()
                result_df = pd.read_json(result_json)
    
                return result_df
            else:
                logger.error(f"Prediction failed with status code: {response.status_code}")
                return pd.DataFrame()
        except requests.RequestException as e:
            logger.error(f"Network error occured: {e}")
            return pd.DataFrame()

    def logs(self) -> dict:
        """Get server logs"""

        # Make a GET request to the logs endpoint
        response = requests.get(f"{self.base_url}/logs")

        if response.status_code == 200:
            # Parse the JSON response
            logs_data = response.json()
            return logs_data
        else:
            logger.error(f"Failed to fetch logs with status code: {response.status_code}")
            return {}

    def download_registry_model(self, workspace: str, model: str, version: str) -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it. 

        See more here:

            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model
        
        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """

        # Prepare the request payload
        request_payload = {
            "workspace": workspace,
            "model_name": model,
            "version": version
        }

        try:
            # Make a POST request to the download_registry_model endpoint
            response = requests.post(f"{self.base_url}/download_registry_model", json=request_payload)
    
            if response.status_code == 200:
                # Parse the JSON response
                result_data = response.json()
                return result_data
            else:
                logger.error(f"Failed to download model with status code: {response.status_code}")
                return {}
        except requests.RequestException as e:
            logger.error(f"Network error occured : {e}")
            return {}
        # ecept Valuerror as e:
            
if __name__=="__main__":

    sc = ServingClient(ip="0.0.0.0", port=8000)
