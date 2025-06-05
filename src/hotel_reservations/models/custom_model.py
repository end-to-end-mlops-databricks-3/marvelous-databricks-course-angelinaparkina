"""Custom model class.

infer_signature (from mlflow.models) → Captures input-output schema for model tracking.

num_features → List of numerical feature names.
cat_features → List of categorical feature names.
target → The column to predict.
parameters → Hyperparameters for LightGBM.
catalog_name, schema_name → Database schema names for Databricks tables.
"""

from typing import Literal

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from loguru import logger
from mlflow import MlflowClient
from mlflow.data.dataset_source import DatasetSource
from mlflow.models import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from hotel_reservations.config import ProjectConfig, Tags


class HotelReservationsModelWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper class for ML models to be used in MLflow.

    This class wraps a model predicting hotel reservations cancellations.
    """

    def __init__(self, params: dict) -> None:
        """Initializes the model for the class.

        :param: params: The hyperparameters to the model.
        """
        self.model = LGBMClassifier
        self.parameters = params

    def prepare_features(self, cat_features: list[str]) -> None:
        """Prepare features for model training.

        This method sets up a preprocessing pipeline including one-hot encoding for categorical
        features and LightGBM classification model.
        """
        logger.info("🔄 Defining preprocessing pipeline...")
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
            ],
            remainder="passthrough",
        )

        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                ("classification", self.model(**self.parameters)),
            ]
        )
        logger.info("✅ Preprocessing pipeline defined.")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the model using the prepared pipeline."""
        logger.info("🚀 Starting training...")
        self.pipeline.fit(X_train, y_train)

    def predict(
        self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame
    ) -> np.ndarray:
        """Make predictions using the trained model.
        :param context: The MLflow context (unused in this implementation).
        :param model_input: Input data for making predictions.
        :return: A pd.DataFrame containing the predictions.
        """
        logger.info(f"model_input:{model_input}")
        preds = self.pipeline.predict(model_input)
        logger.info(f"predictions: {preds}")
        return preds


class CustomModel:
    """Custom model class for hotel reservations classification.

    This class encapsulates the entire workflow of the model, including data loading, processing, and logging.
    """

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession, code_paths: list[str]) -> None:
        """Initilize the model with project configuration.

        :param config: Project configuration object
        :param tags: tags object
        :param spark: SparkSession object
        """
        self.config = config
        self.spark = spark

        # Extract model features from config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target

        # Extract model hyperparameters
        self.parameters = self.config.parameters

        # Extract catalog and schema
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name

        # Extract model-related information
        self.experiment_name = self.config.experiment_name_custom
        self.model_name = (
            f"{self.catalog_name}.{self.schema_name}.hotel_reservations_model_custom"
        )
        self.tags = tags.dict()
        self.code_paths = code_paths

    def load_data(self) -> None:
        """Load training and testing data from Delta tables.

        Splits data into features (X_train, X_test) and target (y_train, y_test).
        """
        logger.info("🔄 Loading data from Databricks tables...")
        self.train_set_spark = self.spark.table(
            f"{self.catalog_name}.{self.schema_name}.train_set"
        )
        self.train_set = self.train_set_spark.toPandas()
        self.test_set = self.spark.table(
            f"{self.catalog_name}.{self.schema_name}.test_set"
        ).toPandas()
        self.data_version = (
            "0"  # describe history -> retrieve. what if it's not the right version?
        )

        self.X_train = self.train_set[self.num_features + self.cat_features]
        self.y_train = self.train_set[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features]
        self.y_test = self.test_set[self.target]
        logger.info("✅ Data successfully loaded.")

    def log_model(self) -> None:
        """Log the model using MLflow."""
        mlflow.set_experiment(self.experiment_name)
        additional_pip_deps = ["pyspark==3.5.0"]
        for package in self.code_paths:
            whl_name = package.split("/")[-1]
            additional_pip_deps.append(f"./code/{whl_name}")

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id

            model = HotelReservationsModelWrapper(
                params=self.parameters
            )
            model.prepare_features(self.cat_features)
            model.train(self.X_train, self.y_train)
            y_pred = model.predict(context = None, model_input = self.X_test)

            # Evaluate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)

            logger.info(f"📊 Accuracy: {accuracy}")
            logger.info(f"📊 F1-score: {f1}")
            logger.info(f"📊 Precision: {precision}")
            logger.info(f"📊 Recall: {recall}")

            # Log parameters and metrics
            mlflow.log_param("model_type", "LightGBM with preprocessing")
            mlflow.log_params(self.parameters)
            mlflow.log_metrics(
                {
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "precision": precision,
                    "recall": recall,
                }
            )

            # Log the model
            signature = infer_signature(model_input=self.X_train, model_output=y_pred)
            dataset = mlflow.data.from_spark(
                self.train_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.train_set",
                version=self.data_version,
            )
            mlflow.log_input(dataset, context="training")

            conda_env = _mlflow_conda_env(additional_pip_deps=additional_pip_deps)
            mlflow.pyfunc.log_model(
                python_model=model,
                artifact_path="pyfunc-lightgbm-hotel-model",
                code_paths=self.code_paths,
                conda_env=conda_env,
                signature=signature,
                input_example=self.X_train.iloc[0:1],
            )

    def register_model(self) -> None:
        """Register model in Unity Catalog."""
        logger.info("🔄 Registering the model in UC...")
        registered_model = mlflow.register_model(
            f"runs:/{self.run_id}/pyfunc-lightgbm-hotel-model",
            name=self.model_name,
            tags=self.tags,
        )
        logger.info(f"✅ Model registered as version {registered_model.version}.")

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=self.model_name, alias="latest-model", version=latest_version
        )

    def retrieve_current_run_dataset(self) -> DatasetSource:
        """Retrieve Mlflow run dataset.

        :return: Loaded dataset source
        """
        run = mlflow.get_run(self.run_id)
        dataset_info = run.inputs.dataset_inputs[0].dataset
        dataset_source = mlflow.data.get_source(dataset_info)
        logger.info("✅ Dataset source loaded.")
        return dataset_source.load()

    def retrieve_current_run_metadata(self) -> tuple[dict, dict]:
        """Retrieve Mlflow run metadata.

        :return: Tuple containing metrics and parameters dictionairies
        """
        run = mlflow.get_run(self.run_id)
        metrics = run.data.to_dictionary()["metrics"]
        params = run.data.to_dictionary()["params"]
        logger.info("✅ Run metadata loaded.")
        return metrics, params

    def load_latest_model_and_predict(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Load the latest model from Mlflow (alias=latest-model) and make predictions.

        Alias latest is not allowed -> we use latest-model instead as alternative.

        :param input_data: Pandas DataFrame containing input_features for prediction.
        :return: Pandas Data Frame with predictions.
        """
        logger.info("🔄 Loading model from MLflow alias 'production'...")

        model_uri = f"models:/{self.model_name}@latest-model"
        model = mlflow.pyfunc.load_model(model_uri)

        logger.info("✅ Model successfully loaded.")

        # Make predictions
        predictions = model.predict(input_data)

        # Return predictions as a DataFrame
        return predictions