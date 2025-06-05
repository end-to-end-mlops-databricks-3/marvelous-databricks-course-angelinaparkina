"""Basic model implementation.

infer_signature (from mlflow.models) -> Captures input-output schema for model tracking.

num_features -> List of numerical feature names.
cat_features -> List of categorical feature names.
target -> Name of the dependent variable.
parameters -> List of hyperparameter values for LightGBM.
catalog_name, schema_name -> Database schema names for Databricks tables.
"""

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from loguru import logger
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class BasicModel:
    """A basic model class for prediction of hotel reservation cancellations using LightGBM.

    This class handles data loading, feature preparation, model training, and MLflow logging.
    """

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
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
        self.experiment_name = self.config.experiment_name_basic
        self.model_name = (
            f"{self.catalog_name}.{self.schema_name}.hotel_reservations_model_basic"
        )
        self.tags = tags.dict()

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
        self.y_test = self.test - set[self.target]
        logger.info("✅ Data successfully loaded.")

    def prepare_features(self) -> None:
        """Encode categorical features and define a preprocessing pipeline.

        Creates a ColumnTransformer for one-hot encoding categorical features while passing through numerical features. Construct a pipeline combining preprocessing and LightGBM classification model.
        """
        logger.info("🔄 Defining preprocessing pipeline...")
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features)
            ],
            remainder="passthrough",
        )

        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                ("classification", LGBMClassifier(**self.parameters)),
            ]
        )
        logger.info("✅ Preprocessing pipeline defined.")

    def train(self) -> None:
        """Train the model."""
        logger.info("🚀 Starting training...")
        self.pipeline.fit(self.X_train, self.y_train)

    def log_model(self) -> None:
        """Log the model using MLflow."""
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id

            y_pred = self.pipeline.predict(self.X_test)

            # Evaluate metrics
            accuracy_score = accuracy_score(self.y_test, y_pred)
            f1_score = f1_score(self.y_test, y_pred)
            precision_score = precision_score(self.y_test, y_pred)
            recall_score = recall_score(self.y_test, y_pred)

            logger.info(f"📊 Accuracy: {accuracy_score}")
            logger.info(f"📊 F1-score: {f1_score}")
            logger.info(f"📊 Precision: {precision_score}")
            logger.info(f"📊 Recall: {recall_score}")

            # Log parameters and metrics
            mlflow.log_param("model_type", "LightGBM with preprocessing")
            mlflow.log_params(self.parameters)
            mlflow.log_metrics(
                {
                    "accuracy": accuracy_score,
                    "f1_score": f1_score,
                    "precision": precision_score,
                    "recall": recall_score,
                }
            )

            # Log the model
            signature = infer_signature(model_input=self.X_train, model_output=y_pred)
            dataset = mlflow.data.from_spark(
                self.train_set_spark,
                table_name=f"{self.catalog_name}.{self.catalog_schema}.train_set",
                version=self.data_version,
            )
            mlflow.log_input(dataset, context="training")
            mlflow.sklearn.log_model(
                sk_model=self.pipeline,
                artifact_path="lightgbm-pipeline-model",
                signature=signature,
            )

    def register_model(self) -> None:
        """Register model in Unity Catalog."""
        logger.info("🔄 Registering the model in UC...")
        registered_model = mlflow.register_model(
            f"runs:/{self.run_id}/lightgbm-pipeline-model",
            name=self.model_name,
            tags=self.tags,
        )
        logger.info(f"✅ Model registered as version {registered_model.version}.")

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(name=self.model_name, alias="latest-model", version=latest_version)

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

        model_uti = f"models:/{self.model_name}@latest-model"
        model = mlflow.sklearn.load_model(model_uri)

        logger.info("✅ Model successfully loaded.")

        # Make predictions
        predictions = model.predict(input_data)

        # Return predictions as a DataFrame
        return predictions