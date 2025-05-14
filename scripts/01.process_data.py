# COMMAND ----
import pandas as pd
import yaml
from pyspark.sql import SparkSession
from loguru import logger
from src.hotel_reservations.config import ProjectConfig
from src.hotel_reservations.data_processor import DataProcessor
from marvelous.logging import setup_logging
from marvelous.timer import Timer

config_path = f"../project_config.yml"

config = ProjectConfig.from_yaml(config_path=config_path,env="dev")

setup_logging(log_file=f"/Volumes/{config.catalog_name}/{config.schema_name}/logs/marvelous-1.log")
logger.info("Configuration loaded:")
logger.info(yaml.dump(config,default_flow_style=False))

spark = SparkSession.builder.getOrCreate()

#Read in the data
df = pd.read_csv(f"/Volumes/{config.catalog_name}/{config.schema_name}/files/Hotel Reservations.csv",header=True)

#Pre-process the data

with Timer() as preprocess_timer:
    data_processor = DataProcessor(df,config,spark)
    data_processor.preprocess()

logger.info(f"Data preprocessing: {preprocess_timer}")

#Split the data
X_train,X_test = data_processor.split_data()
logger.info("Training set shape: %s",X_train.shape)
logger.info("Testing set shape: %s",X_test.shape)

#Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train,X_test)