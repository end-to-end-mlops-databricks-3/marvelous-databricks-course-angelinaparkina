"""Feature Lookup model implementation."""

import mlflow
import numpy as np
import pandas as pd
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from lightgbm import LGBMClassifier
from loguru import logger
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame, SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from hotel_reservations.config import ProjectConfig, Tags


class HotelReservationsModelWrapperFE(mlflow.pyfunc.PythonModel):
    """Wrapper class for ML models to be used in MLflow.

    This class wraps a model predicting hotel reservations cancellations.
    """

    def __init__(self, model: object) -> None:
        """Initialize the model for the class.

        :param model: The underlying machine learning model.
        """
        self.model = model

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model.

        :param context: The MLflow context (unused in this implementation).
        :param model_input: Input data for making predictions.
        :return: A pd.DataFrame containing the predictions.
        """
        logger.info(f"model_input:{model_input}")
        preds = self.model.predict(model_input)
        logger.info(f"predictions: {preds}")
        return preds


class FeatureLookUpModel:
    """A class to manage FeatureLookUpModel."""

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize the model with project configuration."""
        self.config = config
        self.spark = spark
        self.workspace = WorkspaceClient()
        self.fe = feature_engineering.FeatureEngineeringClient()

        # Extract setting from config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name

        # Define table names and function name
        self.feature_table_name = f"{self.catalog_name}.{self.schema_name}.hotel_features"
        self.function_name = f"{self.catalog_name}.{self.schema_name}.calculate_last_minute_trip"

        # MLflow configuration
        self.experiment_name = self.config.experiment_name_fe
        self.tags = tags.dict()

    def create_feature_table(self) -> None:
        """Create or update the hotel_features table and populate it.

        This table stores features related to a hotel stay.
        """
        self.spark.sql(
            f"""
        CREATE OR REPLACE TABLE {self.feature_table_name}
        (Booking_ID STRING NOT NULL, avg_price_per_room FLOAT, room_type_reserved STRING, repeated_guest INT);
        """
        )
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} ADD CONSTRAINT hotel_pk PRIMARY KEY(Booking_ID);")
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT Booking_ID, avg_price_per_room, room_type_reserved, repeated_guest FROM {self.catalog_name}.{self.schema_name}.train_set"
        )
        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT Booking_ID, avg_price_per_room, room_type_reserved, repeated_guest FROM {self.catalog_name}.{self.schema_name}.test_set"
        )
        logger.info("✅ Feature table created and populated.")

    def define_feature_function(self) -> None:
        """Define a function to calculate whether a trip was a last minute trip.

        This function checks the amount of days left before the trip.
        """
        self.spark.sql(
            f"""
        CREATE OR REPLACE FUNCTION {self.function_name}(lead_time BIGINT)
        RETURNS INT
        LANGUAGE PYTHON AS
        $$
        return int(lead_time < 3)
        $$
        """
        )
        logger.info("✅ Feature function defined.")

    def load_data(self) -> None:
        """Load training and testing data from Delta tables.

        Drops specified columns.
        """
        self.train_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set").drop(
            "avg_price_per_room", "room_type_reserved", "repeated_guest"
        )
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set").toPandas()

        logger.info("✅ Data successfully loaded.")

    def feature_engineering(self) -> None:
        """Perform feature engineering by linking data with feature tables.

        Creates a training set using FeatureLookup and FeatureFunction.
        """
        self.training_set = self.fe.create_training_set(
            df=self.train_set,
            label=self.target,
            feature_lookups=[
                FeatureLookup(
                    table_name=self.feature_table_name,
                    feature_names=[
                        "avg_price_per_room",
                        "room_type_reserved",
                        "repeated_guest",
                    ],
                    lookup_key="Booking_ID",
                ),
                FeatureFunction(
                    udf_name=self.function_name,
                    output_name="last_minute_booking",
                    input_bindings={"lead_time": "lead_time"},
                ),
            ],
            exclude_columns=["update_timestamp_utc"],
        )

        self.training_df = self.training_set.load_df().toPandas()

        self.test_set["last_minute_booking"] = (self.test_set["lead_time"] < 3).astype(int)

        self.X_train = self.training_df[self.num_features + self.cat_features + ["last_minute_booking"]]
        self.y_train = self.training_df[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features + ["last_minute_booking"]]
        self.y_test = self.test_set[self.target]

        logger.info("✅ Feature engineering completed.")

    def train(self) -> None:
        """Train the model and log results to Mlflow.

        Uses a pipeline with preprocessing and LightGBM Classifier.
        """
        logger.info("🚀 Starting training...")

        preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features)],
            remainder="passthrough",
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classification", LGBMClassifier(**self.parameters)),
            ]
        )

        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_test)

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
            signature = infer_signature(self.X_train, y_pred)

            self.fe.log_model(
                model=HotelReservationsModelWrapperFE(pipeline),
                flavor=mlflow.pyfunc,
                artifact_path="pyfunc-lightgbm-hotel-model-fe",
                training_set=self.training_set,
                signature=signature,
            )

    def register_model(self) -> str:
        """Register the trained model to Mlflow registry.

        Registers the model and sets alias to 'latest-model'.
        """
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/pyfunc-lightgbm-hotel-model-fe",
            name=f"{self.catalog_name}.{self.schema_name}.hotel_reservations_model_fe",
            tags=self.tags,
        )

        # Fetch the latest version dynamically
        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.catalog_name}.{self.schema_name}.hotel_reservations_model_fe",
            alias="latest-model",
            version=latest_version,
        )

        return latest_version

    def load_latest_model_and_predict(self, X: DataFrame) -> DataFrame:
        """Load the trained model from Mlflow using Feature Engineering Client and make predictions.

        Loads the model with the alias 'latest-model' and scores the batch.
        :param X: DataFrame containing input features.
        :return DataFrame containing predictions.
        """
        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.hotel_reservations_model_fe@latest-model"

        predictions = self.fe.score_batch(model_uri=model_uri, df=X)
        return predictions
