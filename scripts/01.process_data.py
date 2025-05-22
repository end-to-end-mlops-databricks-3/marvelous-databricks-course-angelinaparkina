"""Script for first week of the course."""
# COMMAND ----------|^
# %pip install -e ..

# COMMAND ----------|^
# %pip install git+https://github.com/end-to-end-mlops-databricks-3/marvelous

# COMMAND ----------|^
# %restart_python

# COMMAND ----------|^
# from pathlib import Path
# import sys
# sys.path.append(str(Path.cwd().parent / 'src'))

# COMMAND ----------|^
import yaml
from loguru import logger
from pyspark.sql import SparkSession

from marvelous.logging import setup_logging
from marvelous.timer import Timer

from hotel_reservations import DataProcessor, DataReader
from hotel_reservations.config import ProjectConfig

# COMMAND ----------|^

config_path = "../project_config.yml"

config = ProjectConfig.from_yaml(config_path=config_path, env="dev")

# temp_log_file_path = "/tmp/logs/marvelous-1.log"

log_file_path = (
    f"/Volumes/{config.catalog_name}/{config.schema_name}/logs/marvelous-1.log"
)

setup_logging(log_file=log_file_path)

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------|^

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------|^

data_path = (
    f"/Volumes/{config.catalog_name}/{config.schema_name}/files/Hotel Reservations.csv"
)

# COMMAND ----------|^

# Read in the data
with Timer() as reader_timer:  # do i need to use timer with every class i use
    data_reader = DataReader(data_path)
    df = data_reader.read_csv()

logger.info(f"Data reading: {reader_timer}")

# COMMAND ----------|^

# Pre-process the data

with Timer() as preprocess_timer:
    data_processor = DataProcessor(df, config, spark)
    data_processor.preprocess()

logger.info(f"Data preprocessing: {preprocess_timer}")

# COMMAND ----------|^

# Split the data
X_train, X_test = data_processor.split_data()
logger.info(f"Training set shape: {X_train.shape}")
logger.info(f"Testing set shape: {X_test.shape}")

# COMMAND ----------|^

# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test)

# COMMAND ----------|^

logger.remove()

# with open(temp_log_file_path, "r") as log_file:
#     logs = log_file.readlines()

# dbutils.fs.put(log_file_path, "".join(logs), overwrite=True)
