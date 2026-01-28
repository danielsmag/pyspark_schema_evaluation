from __future__ import annotations

from logging import Logger
from typing import Any

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import DataType, StructField, StructType

from pyspark_schema_evaluation.utils._spark._safe import safe_cast_col
from pyspark_schema_evaluation.core._logging import get_logger, safe_log
from pyspark_schema_evaluation.models.models import SchemaDiff

__all__: list[str] = ["_compare_schema", "_diff_schemas", "_add_column_with_default"]

logger: Logger = get_logger()

def _compare_schema(
    actual_schema: StructType,
    expected_schema: StructType,
) -> bool:
    actual_fields: set = {
        (f.name, f.dataType.simpleString()) for f in actual_schema.fields
    }
    expected_fields: set = {
        (f.name, f.dataType.simpleString()) for f in expected_schema.fields
    }

    missing: set = expected_fields - actual_fields
    extra: set = actual_fields - expected_fields

    if missing or extra:
        safe_log(logger, level="error", message=f"Missing fields:{missing}")
        safe_log(logger, level="error", message=f"Extra fields {extra}")
        return False
    return True

def _diff_schemas(
    actual_schema: StructType,
    expected_schema: StructType,
) -> SchemaDiff:
    """
    Get detailed diff between two schemas.

    Args:
        actual_schema: Current/actual schema
        expected_schema: Target/expected schema

    Returns:
        SchemaDiff with detailed comparison
    """
    actual: dict[str, StructField] = {f.name: f for f in actual_schema.fields}
    expected: dict[str, StructField] = {f.name: f for f in expected_schema.fields}

    missing: list[str] = [n for n in expected if n not in actual]
    extra: list[str] = [n for n in actual if n not in expected]

    type_mismatches: dict[str, tuple[str, str]] = {
        name: (actual[name].dataType.simpleString(), expected[name].dataType.simpleString())
        for name in actual.keys() & expected.keys()
        if actual[name].dataType != expected[name].dataType
    }

    nullable_changes: dict[str, tuple[bool, bool]] = {
        name: (actual[name].nullable, expected[name].nullable)
        for name in actual.keys() & expected.keys()
        if actual[name].nullable != expected[name].nullable
    }

    is_compatible: bool = not missing and not type_mismatches

    return SchemaDiff(
        missing_columns=missing,
        extra_columns=extra,
        type_mismatches=type_mismatches,
        nullable_changes=nullable_changes,
        is_compatible=is_compatible,
    )
    
def _add_column_with_default(
        df: DataFrame,
        col_name: str,
        col_type: DataType,
        default_values: dict[str, Any],
) -> DataFrame:
    """Add a column with optional custom default value."""
    default: Any | None = default_values.get(col_name)
    if default is not None:
        return df.withColumn(col_name, F.lit(default).cast(col_type))
    return safe_cast_col(df, col_name, col_type)