"""Unit tests for DataProcessor."""

import pandas as pd
import pytest
from conftest import CATALOG_DIR
from delta.tables import DeltaTable
from pyspark.sql import SparkSession

from hotel_reservations.config import ProjectConfig
from hotel_reservations.data_processor import DataProcessor

#create tests to check whether the read in was successful - that the df is not empty
#create test that check whether the amount of features we specify in config is the same as number of columns in the df
#test to check for data types

def test_data_ingestion(sample_data: pd.DataFrame) -> None:
    """Test the data ingestion process by checking the shape of the sample data.

    Asserts that the sample data has at least one row and one column.

    :param sample_data: The sample data to be tested
    """
    assert sample_data.shape[0] > 0
    assert sample_data.shape[1] > 0

def test_dataprocessor_init(
    sample_data: pd.DataFrame,
    config: ProjectConfig,
    spark_session: SparkSession,
) -> None:
    """Test the initialization of DataProcessor.

    :param sample_data: Sample DataFrame for testing
    :param config: Configuration object for the project
    :param spark: SparkSession object
    
    """
    processor = DataProcessor(pandas_df=sample_data, config=config, spark=spark_session)
    assert isinstance(processor.df, pd.DataFrame)
    assert processor.df.equals(sample_data)

    assert isinstance(processor.config, ProjectConfig)
    assert isinstance(processor.spark, SparkSession)


def test_na_handling_target(sample_data: pd.DataFrame,config: ProjectConfig,spark_session: SparkSession) -> None:
    """Test missing value handling in the DataProcessor.
    This test focuses on testing if the target column has no missing values.

    :param sample_data: Input DataFrame containing sample data
    :param config: Configuration object for the project
    :param spark: SparkSession object

    """
    processor = DataProcessor(pandas_df=sample_data, config=config, spark=spark_session)
    processor.preprocess()

    assert processor.df[config.target].isnull().sum() == 0

def test_column_selection(sample_data: pd.DataFrame,config: ProjectConfig, spark_session: SparkSession) -> None:
    """Test the column selection in the Data Processor.
    This test focuses on checking if the amount of columns we want to include based on the config matches what we actually selected.

    :param sample_data: Input DataFrame containing sample data
    :param config: Configuration object for the project
    :param spark: SparkSession object

    """
    processor = DataProcessor(pandas_df=sample_data, config=config, spark=spark_session)
    processor.preprocess()

    selected_cols = config.cat_features + config.num_features + [config.target] + ['Booking_ID']
    assert len(processor.df.columns)==len(selected_cols)

def test_column_transformations(sample_data: pd.DataFrame,config: ProjectConfig, spark_session: SparkSession) -> None:
    """Test the columns were correctly processed by Data Processor.
    Checking if different column transformations were correctly applied.

    :param sample_data: Input DataFrame containing sample data
    :param config: Configuration object for the project
    :param spark: SparkSession object
    """
    processor = DataProcessor(pandas_df=sample_data, config=config, spark=spark_session)
    processor.preprocess()

    assert processor.df['Lead_time'].dtype == 'int64'
    assert processor.df['type_of_meal_plan'].dtype == 'category'
    assert 'arrival_year' not in processor.df.columns

def test_split_data_default_params(
    sample_data: pd.DataFrame, config: ProjectConfig, spark_session: SparkSession
) -> None:
    """Test the default parameters of the split_data method in DataProcessor.

    This function tests if the split_data method correctly splits the input DataFrame
    into train and test sets using default parameters.

    :param sample_data: Input DataFrame to be split
    :param config: Configuration object for the project
    :param spark: SparkSession object
    """
    processor = DataProcessor(pandas_df=sample_data, config=config, spark=spark_session)
    processor.preprocess()
    train, test = processor.split_data()

    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert len(train) + len(test) == len(processor.df)
    assert set(train.columns) == set(test.columns) == set(processor.df.columns)

    # # The following lines are just to mimick the behavior of delta tables in UC
    # # Just one time execution in order for all other tests to work
    train.to_csv((CATALOG_DIR / "train_set.csv").as_posix(), index=False)  # noqa
    test.to_csv((CATALOG_DIR / "test_set.csv").as_posix(), index=False)  # noqa

def test_data_save(sample_data: pd.DataFrame,config: ProjectConfig, spark_session: SparkSession) -> None:
    """Test that the data is saved to UC.

    :param sample_data: Input DataFrame containing sample data
    :param config: Configuration object for the project
    :param spark: SparkSession object
    """
    processor = DataProcessor(pandas_df=sample_data, config=config, spark=spark_session)
    processor.save_to_catalog()

    path = f"{config.catalog}.{config.schema}"
    #not sure how to make this dynamic regardless of table_name, by putting it into the function as parameter?
    assert DeltaTable.isDeltaTable(spark_session, f"{path}.train_set")
    assert DeltaTable.isDeltaTable(spark_session, f"{path}.test_set")
    
    saved_df_train = spark_session.table(f"{path}.train_set")
    assert not saved_df_train.rdd.isEmpty()

    saved_df_test = spark_session.table(f"{path}.test_set")
    assert not saved_df_test.rdd.isEmpty()
