"""Train and register a custom model."""

# COMMAND ----------|^
import argparse

import mlflow
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from hotel_reservations.config import ProjectConfig, Tags
from hotel_reservations.models.custom_model import CustomModel
from marvelous.common import create_parser
from hotel_reservations import __version__ as hotel_reservations_v

# COMMAND ----------|^
# Configure tracking uri
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------|^
args = create_parser()
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--root_path",
#     action="store",
#     default=None,
#     type=str,
#     required=True,
# )
# COMMAND ----------|^
# parser.add_argument(
#     "--env",
#     action="store",
#     default="dev",
#     type=str,
#     required=True,
# )
# COMMAND ----------|^
# parser.add_argument(
#     "--git_sha",
#     action="store",
#     default="abcd",
#     type=str,
#     required=True,
# )

# parser.add_argument(
#     "--job_run_id",
#     action="store",
#     default=None,
#     type=str,
#     required=True,
# )
# COMMAND ----------|^
# parser.add_argument(
#     "--branch",
#     action="store",
#     default=None,
#     type=str,
#     required=True,
# )

# COMMAND ----------|^
# args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"
# COMMAND ----------|^
config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
tags_dict = {"git_sha": args.git_sha, "branch": args.branch, "job_run_id": args.job_run_id}
tags = Tags(**tags_dict)

# COMMAND ----------|^
# Initialize model
custom_model = CustomModel(config=config, tags=tags, spark=spark, code_paths=[f"../dist/hotel_reservations-{hotel_reservations_v}-py3-none-any.whl"])
logger.info("Model initialized.")

# COMMAND ----------|^
# Load data and prepare features
custom_model.load_data()
logger.info("Loaded data.")

# COMMAND ----------|^
# Train + log the model (runs everything including MLflow logging)
custom_model.log_model()
logger.info("Model training completed.")


# COMMAND ----------|^
# Evaluate model
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(100).toPandas()

model_improved = custom_model.model_improved(test_set = test_set)
logger.info("Model evaluation completed, model improved: ", model_improved)

is_test = args.is_test

# when runniing test, always register and deploy
if is_test == 1:
    model_improved = True

if model_improved:
    # Register the model
    latest_version = custom_model.register_model()
    logger.info("Registered model with version: ", latest_version)
    dbutils.jobs.taskValues.set(key = "model_version", value = latest_version)
    dbutils.jobs.taskValues.set(key = "model_updated", value = 1)

else:
    dbutils.jobs.taskValues.set(key = "model_updated", value = 0)