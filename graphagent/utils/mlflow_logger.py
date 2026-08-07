"""
MLflow logging utilities.
"""

import mlflow
from torchgen import model


class MLFlowLogger:

    def __init__(self, experiment_name="GraphAgent"):

        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name):

        mlflow.start_run(run_name=run_name)

    def log_params(self, params):

        mlflow.log_params(params)

    def log_metrics(self, metrics, step=None):

        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model, artifact_path):

        mlflow.pytorch.log_model(
            pytorch_model=model,
            name=artifact_path,
            serialization_format="pickle",
    )

    def end_run(self):

        mlflow.end_run()