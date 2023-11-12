## Introduction to Comet.ml

Comet.ml is a powerful tool for tracking, managing, and optimizing your machine learning experiments. It provides a centralized platform for storing all experiment metadata, including configurations, hyperparameters, dataset hashes, and saved models. Comet.ml is especially beneficial for team projects, as it facilitates collaboration and ensures reproducibility of results.


## Setting Up Comet.ml

### Step 1: Create a Comet.ml Account
**For All Team Members**: Sign up for a Comet.ml account using your academic email via Comet.ml signup.
**Claim Your Free Academic Tier**: After signing up, claim your free academic tier benefits.

### Step 2: Establish a Shared Workspace
**Designate a Workspace Host**: One team member should create a workspace in Comet.ml.
**Add Team Members**: The workspace host should add all team members and the TA account ift6758-2023 to the workspace. This is done in the workspace settings.

### Step 3: Install Comet.ml SDK
Install the `Comet.ml` Python SDK in your environment:

`pip install comet_ml`

## Integrating Comet.ml with Scikit-Learn

### Step 1: Import Comet.ml
Import `comet_ml`at the top of your Python script, before other libraries like sklearn.

```python
from comet_ml import Experiment
```

### Step 2: Configure Environment Variables
**Set API Key**: Store your Comet.ml API key in an environment variable named COMET_API_KEY.
**Access API Key**: Use the os package in Python to access the API key.

```python
import os
api_key = os.environ.get('COMET_API_KEY')
```

### Step 3: Create and Configure an Experiment
Initialize a new experiment with your API key, project name, and workspace:

```python
exp = Experiment(
    api_key=api_key,
    project_name='your_project_name',
    workspace='your_workspace_name'
)
```

### Step 4: Tracking Experiments
**Log Metrics**: Use exp.log_metrics() to log metrics like accuracy, loss, etc.

```python
exp.log_metrics({"accuracy": accuracy_score, "loss": loss_value})
```

**Log Parameters**: Log model parameters and hyperparameters.

```python
exp.log_parameters({"C": 1.0, "kernel": "linear"})
```

### Step 5: Saving and Registering Models
**Save Model Locally**: Save your scikit-learn model using joblib or pickle.

```python
from joblib import dump
dump(your_model, 'model.joblib')
```
**Log Model in Comet.ml**: Use exp.log_model() to log the model in Comet.ml.

```python
exp.log_model("model_name", "model.joblib")
```

### Step 6: Review and Analyze
**Accessing Dashboard**: Review your experiments on the Comet.ml dashboard.
**Collaboration**: Share experiment links with your team for collaborative analysis.


## Collaborative Use of Shared Workspace and API

Our team has set up a shared workspace in Comet.ml under the username `hfereidouni` to facilitate collaborative experiment tracking. All team members are encouraged to use this workspace for logging their experiments. To ensure secure and consistent access, we are utilizing a shared API key. This approach allows every team member to connect to the same workspace, enabling us to monitor, compare, and discuss our experiments in real-time. It’s important to remember not to hardcode the API key in your scripts or notebooks. Instead, store it as an environment variable on your machine and access it programmatically. This practice keeps our workspace secure and prevents unauthorized access. By using this centralized setup, we can maintain a cohesive and organized overview of our project's progress, making it easier to identify successful experiments and collaborate more effectively.


## Conclusion

By following these steps, our team can effectively use Comet.ml to track machine learning experiments using scikit-learn. This setup ensures that all experiments are logged and reproducible, facilitating a more organized and collaborative project environment. Remember to frequently consult Comet.ml's documentation for specific features and advanced usage.
