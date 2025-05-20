#create a class with data pre-processing steps
"""Data preprocessing module."""

import pandas as pd
import datetime
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from sklearn.model_selection import train_test_split
from src.hotel_reservations.config import ProjectConfig

class DataProcessor:

    """A class for preprocessing and managing DataFrame operations.
    This class handles data preprocessing, splitting, and saving to Databricks tables.

    """
    def __init__(self,pandas_df: pd.DataFrame,config:ProjectConfig,spark: SparkSession) -> None:
        self.df = pandas_df
        self.config = config
        self.spark = spark
    
    def preprocess(self) -> None:
        """ Preprocess the df stored in self.df
        This method handles missing values, converts data types, and performs feature engineering.
        
        """
        #handle duplicates
        self.df = self.df.drop_duplicates()
        
        #handle missing values
        self.df[self.config.target].dropna() #a test can be made to check whether this works

        #convert numerical features to numerical type
        num_features = self.config.num_features
        self.df[num_features] = self.df[num_features].apply(pd.to_numeric,errors='coerce')

        #convert categorical features to categorical type
        cat_features = self.config.cat_features
        self.df[cat_features] = self.df[cat_features].astype('category')


        #extract target and relevant features
        target = self.config.target
        relevant_cols = cat_features + num_features + [target] + ['Booking_ID']
        self.df = self.df[relevant_cols]
        self.df["Booking_ID"] = self.df['Booking_ID'].astype(str)

        #change target data type to binary
        self.df[target] = self.df[target].map({'Not_Canceled': 0, 'Canceled': 1}).astype(int)

    def split_data(self,test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame,pd.DataFrame]:
        """Split the DataFrame (self.df) into training and test sets.

        :param test_size: The proportion of the dataset to include in the test split.
        :param random_state: Controls the shuffling applied to the data before applying the split.
        :return: A tuple containing the training and test DataFrames.
        """
        train_set,test_set = train_test_split(self.df,test_size=test_size,random_state=random_state)

        return train_set,test_set
    
    def save_to_catalog(self,train_set: pd.DataFrame,test_set: pd.DataFrame) -> None:
         """Save the train and test sets into Databricks tables.

        :param train_set: The training DataFrame to be saved.
        :param test_set: The test DataFrame to be saved.
        """
         train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn("update_timestamp_utc",f.to_utc_timestamp(f.current_timestamp(),"UTC"))

         test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn("update_timestamp_utc",f.to_utc_timestamp(f.current_timestamp(),"UTC"))

         train_set_with_timestamp.write.mode("overwrite").saveAsTable(f"{self.config.catalog_name}.{self.config.schema_name}.train_set")
         test_set_with_timestamp.write.mode("overwrite").saveAsTable(f"{self.config.catalog_name}.{self.config.schema_name}.test_set")

    def enable_change_data_feed(self) -> None:
        """Enable Change Data Feed for train and test set tables.

        This method alters the tables to enable Change Data Feed functionality.
        """
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )