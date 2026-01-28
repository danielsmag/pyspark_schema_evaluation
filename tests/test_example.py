
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

from pyspark_schema_evaluation import SchemaEvolution, LayerPolicy


def test_schema_evolution_import():
    """Test that SchemaEvolution can be imported."""
    assert SchemaEvolution is not None


def test_layer_policy_enum():
    """Test that LayerPolicy enum values are accessible."""
    assert LayerPolicy.BRONZE == "bronze"
    assert LayerPolicy.SILVER == "silver"
    assert LayerPolicy.GOLD == "gold"


def test_basic_schema_evolution(spark):
    """Example test for basic schema evolution functionality."""
    # Create a simple schema
    schema = StructType([
        StructField("id", IntegerType(), nullable=False),
        StructField("name", StringType(), nullable=True),
    ])
    
    # Create a simple DataFrame
    data = [(1, "Alice"), (2, "Bob")]
    df = spark.createDataFrame(data, schema)
    
    # Initialize SchemaEvolution
    evolution = SchemaEvolution()
    
    # Test that we can compare schemas
    result = evolution.compare_schema(schema, schema)
    assert result is True
