## pyspark_schema_evaluation

**pyspark_schema_evaluation** is a small library that helps you **validate, evolve, and track PySpark DataFrame schemas** in a medallion-style (Bronze/Silver/Gold) data platform.

It provides:
- **Schema evolution engine** (`SchemaEvolution`) with rich validation and evolution APIs
- **Medallion-aware policies** via `LayerPolicy` and `LayerPolicyRegistry`
- **Schema registry abstraction** (`ISchemaRegistry`, `InMemorySchemaRegistry`)
- **Typed models** for drift, breaking-change detection, and metadata
- **Decorator-based API** to validate/evolve DataFrames around your business functions

The library is designed to be:
- **Safe**: prefers explicit handling of conflicts and drift
- **Typed**: ships with `py.typed` and type annotations
- **PySpark‑native**: built on top of `pyspark.sql` types and DataFrames

---

## Installation

Install from PyPI (once published):

```bash
pip install pyspark-schema-evaluation
```

Requirements:
- Python **3.11+**
- `pyspark>=3.0.0,<5.0.0`
- `pydantic>=2.0.0,<3.0.0`
- `pydantic-settings>=2.0.0,<3.0.0`

---

## Quick start

### Basic usage with `SchemaEvolution`

```python
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql import SparkSession

from pyspark_schema_evaluation import (
    SchemaEvolution,
    LayerPolicy,
    CompatibilityMode,
    ConflictMode,
)

spark = SparkSession.builder.master("local[2]").appName("example").getOrCreate()

incoming_schema = StructType(
    [
        StructField("id", IntegerType(), nullable=False),
        StructField("name", StringType(), nullable=True),
    ]
)

target_schema = incoming_schema  # usually loaded from code, files, or registry

df = spark.createDataFrame([(1, "Alice"), (2, "Bob")], schema=incoming_schema)

evolution = SchemaEvolution()

# Validate and evolve a DataFrame to match the target schema
result = evolution.evolve_df_to_target_schema_with_report(
    df=df,
    target_schema=target_schema,
    compatibility_mode=CompatibilityMode.BACKWARDS,
    extra_allowed=False,
    on_conflict=ConflictMode.STRING,
)

# Access the evolved DataFrame and a rich validation report
evolved_df = result.df
result.log()  # logs summary and warnings (if logging is enabled)
```

### Using the in‑memory schema registry

```python
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

from pyspark_schema_evaluation import (
    SchemaEvolution,
    InMemorySchemaRegistry,
    LayerPolicy,
)

registry = InMemorySchemaRegistry()
evolution = SchemaEvolution(registry=registry, default_layer=LayerPolicy.SILVER)

customer_schema = StructType(
    [
        StructField("id", IntegerType(), nullable=False),
        StructField("name", StringType(), nullable=True),
    ]
)

# Register a schema version
versioned = registry.register_schema(
    table_name="silver.customers",
    schema=customer_schema,
    layer=LayerPolicy.SILVER,
    description="Initial customers schema",
)

# Later, retrieve the latest schema and validate new data against it
latest = registry.get_latest_schema("silver.customers")
assert latest is not None
```

---

## Decorator-based API

The `decorators` module lets you **validate or evolve DataFrame schemas around your functions** using simple decorators.

```python
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql import DataFrame

from pyspark_schema_evaluation.decorators.schema_decorators import (
    validate_schema,
    evolve_df_to_target_schema,
    evolve_for_layer,
)
from pyspark_schema_evaluation import LayerPolicy


def customer_schema() -> StructType:
    return StructType(
        [
            StructField("id", IntegerType(), nullable=False),
            StructField("name", StringType(), nullable=True),
        ]
    )


@validate_schema(
    expected_schema=customer_schema,
    extra_columns_conflict="drop",  # or ConflictMode.DROP
    validate_on="input",
)
def process_customers(df: DataFrame) -> DataFrame:
    # df is now validated (and possibly adjusted) to match the expected schema
    return df


@evolve_for_layer(
    target_schema=customer_schema,
    layer=LayerPolicy.SILVER,
)
def write_customers(df: DataFrame) -> DataFrame:
    # df is evolved according to the layer policy before this function runs
    return df
```

Key decorators:
- `validate_schema` – validate (and optionally adjust) schemas on **input or output**
- `evolve_df_to_target_schema` – evolve to a given target schema
- `evolve_for_layer` – evolve by **medallion layer policy** (Bronze/Silver/Gold/custom)

---

## Public API surface

From the package root you can import the main objects:

```python
from pyspark_schema_evaluation import (
    # Core
    SchemaEvolution,
    ISchemaEvolution,

    # Registry
    ISchemaRegistry,
    InMemorySchemaRegistry,
    LayerPolicyRegistry,

    # Enums
    LayerPolicy,
    CompatibilityMode,
    ConflictMode,
    DriftSeverity,
    DriftType,
    BreakingChangeType,

    # Models
    VersionedSchema,
    SchemaValidationResult,
    SchemaDiff,
    LayerPolicyConfig,
    SchemaDriftAlert,
    BreakingChange,
    BreakingChangeReport,
    ColumnMetadata,
    SchemaMetadata,
    DriftDetectionResult,

    # Exceptions
    SchemaEvolutionError,

    # Result helpers
    Result,
    Ok,
    Err,
)
```

You can also access the decorators via:

```python
from pyspark_schema_evaluation.decorators.schema_decorators import (
    validate_schema,
    evolve_df_to_target_schema,
    evolve_for_layer,
)
```

---

## Testing

This project uses **pytest**.

Install dev dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Run tests with coverage:

```bash
pytest --cov=pyspark_schema_evaluation
```

---

## Development

Recommended steps:

1. Create and activate a virtual environment.
2. Install the project in editable mode with dev extras:

   ```bash
   pip install -e ".[dev]"
   ```

3. Run linters and type checks (if configured in your environment), for example:

   ```bash
   ruff check pyspark_schema_evaluation tests
   mypy pyspark_schema_evaluation
   ```

4. Run the test suite:

   ```bash
   pytest
   ```

---

## License

This project is licensed under the terms of the **MIT License**.  
See the `LICENSE` file for details.