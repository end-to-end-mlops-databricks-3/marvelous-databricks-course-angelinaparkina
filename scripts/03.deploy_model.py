"""Deploy the custom model."""

# COMMAND ----------|^
import os
import time
from typing import Dict, List

import requests
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient

from hotel_reservations.config import ProjectConfig
from hotel_reservations.serving.model_serving import ModelServing
from marvelous.common import create_parser

# COMMAND ----------|^
args = create_parser()

root_path = args.root_path
is_test = args.is_test

# COMMAND ----------|^
# spark session

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# COMMAND ----------|^
# get environment variables
os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------|^
# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name
endpoint_name = "angelina-hotel-reservations-model-serving"

# COMMAND ----------|^
# Initialize feature store manager
model_serving = ModelServing(
    model_name=f"{catalog_name}.{schema_name}.hotel_reservations_model_custom", endpoint_name=endpoint_name
)

# COMMAND ----------|^
# Deploy the model serving endpoint
model_serving.deploy_or_update_serving_endpoint()
logger.info("Started deployment/update of the serving endpoint.")

# COMMAND ----------|^
# Delete endpoint if test
if is_test==1:
    workspace = WorkspaceClient()
    workspace.serving_endpoints.delete(name=endpoint_name)
    logger.info("Deleting serving endpoint.")