"""
Pytest configuration and shared fixtures for tests.
"""

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark_session():
    """Create a SparkSession for testing."""
    spark = (
        SparkSession.builder
        .appName("pyspark_schema_evaluation_tests")
        .master("local[2]")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "false")
        .getOrCreate()
    )
    yield spark
    spark.stop()


@pytest.fixture(scope="function")
def spark(spark_session):
    """Provide a SparkSession for each test."""
    return spark_session
