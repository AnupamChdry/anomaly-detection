import os

import mlflow
from dotenv import load_dotenv

load_dotenv()


def setup_mlflow(experiment_name: str):
    """Configure MLflow to use DagsHub as the tracking server."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise EnvironmentError("MLFLOW_TRACKING_URI not set. Check your .env file.")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)