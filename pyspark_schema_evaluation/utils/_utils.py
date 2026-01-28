from __future__ import annotations

import threading
from _thread import LockType
from abc import ABCMeta
from typing import Any, Final, Literal

from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import StructType, StructField

from pyspark_schema_evaluation.models.models import (
        SchemaValidationResult,
)
from pyspark_schema_evaluation.core._enums import ConflictMode
from pyspark_schema_evaluation.utils._spark._safe import safe_drop


__all__: list[str] = [
    "TYPE_MAPPING",
    "_handle_extra_columns_with_tracking",
    "_build_compatibility_report",
]


TYPE_MAPPING: Final[dict[str, str]] = {
    "STRING": "STRING",
    "DOUBLE": "DOUBLE",
    "FLOAT": "FLOAT",
    "INT": "INT",
    "INTEGER": "INT",
    "LONG": "BIGINT",
    "BOOLEAN": "BOOLEAN",
    "TIMESTAMP": "TIMESTAMP",
    "DATE": "DATE",
}

def _handle_extra_columns_with_tracking(        
        df: DataFrame,
        expected_schema: StructType,
        extra_columns_conflict: Literal[ConflictMode.IGNORE, ConflictMode.DROP],
        result: SchemaValidationResult,
) -> tuple[DataFrame, SchemaValidationResult]:
        """Handle extra columns with change tracking."""
        expected_names: set[str] = set[str](expected_schema.fieldNames())

        for f in df.schema.fields:
            if f.name not in expected_names:
                if extra_columns_conflict == ConflictMode.DROP:
                    df = safe_drop(df, f.name)
                    result.columns_dropped.append(f.name)

        return df, result


class SingletonMeta(type):
    _instances: dict[Any, Any] = {}

    _lock: LockType = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]
    


class SingletonABCMeta(ABCMeta, SingletonMeta):
    pass


def _build_compatibility_report(table_name: str, diff: Any) -> str:
    """Build a human-readable compatibility report."""
    if diff is None:
        return "No diff available"
    
    lines: list[str] = [f"Schema Compatibility Report for '{table_name}':"]
    
    if diff.missing_columns:
        lines.append(f"  ❌ Missing columns (breaking): {diff.missing_columns}")
    if diff.extra_columns:
        lines.append(f"  ➕ Extra columns (additive): {diff.extra_columns}")
    if diff.type_mismatches:
        type_changes = [f"{k}: {v[0]} → {v[1]}" for k, v in diff.type_mismatches.items()]
        lines.append(f"  🔄 Type changes: {type_changes}")
    if diff.nullable_changes:
        null_changes = [f"{k}: {v[0]} → {v[1]}" for k, v in diff.nullable_changes.items()]
        lines.append(f"  ⚡ Nullable changes: {null_changes}")
    
    lines.append(f"  Compatible: {'✅ Yes' if diff.is_compatible else '❌ No'}")
    
    return "\n".join(lines)


def drop_extra_columns(        
        df: DataFrame,
        existing: dict[str, StructField],
        target: dict[str, StructField],
    ) -> DataFrame:
        """Drop columns that exist in DataFrame but not in target schema."""
        for name in set(existing) - set(target):
            df = safe_drop(df, name)
        return df
    