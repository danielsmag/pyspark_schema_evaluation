from __future__ import annotations

from pyspark.sql.column import Column
from pyspark.sql import functions as F
from pyspark.sql.dataframe import DataFrame
from typing import Any, Final, Optional
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    BooleanType,
    TimestampType,
    DateType,
    FloatType,
    StringType,
    DataType,
    VariantType,
    MapType,
    StructType,
    ArrayType,
    ByteType,
    ShortType,
    DecimalType,
)
from sp_db_pipeline.core.logger.logger import ILogger
from sp_db_pipeline.core.logger.loader import safe_get_logger
from logging import Logger





__all__: list[str] = [
    "safe_int",
    "safe_to_json",
    "safe_double",
    "safe_timestamp",
    "safe_boolean",
    "safe_string",
    "safe_json_array",
    "safe_column_rename",
    "safe_drop",
    "safe_to_bool",
    "safe_date",
    "safe_cast_col",
    "safe_check_if_data_exists",
    "safe_if_empty",
    "safe_deduplicate",
    "is_safe_promotion",
    "safe_to_percent",
]


TYPE_PROMOTION_HIERARCHY: Final[dict[type, list[type]]] = {
        ByteType: [ShortType, IntegerType, LongType, FloatType, DoubleType, DecimalType, StringType],
        ShortType: [IntegerType, LongType, FloatType, DoubleType, DecimalType, StringType],
        IntegerType: [LongType, FloatType, DoubleType, DecimalType, StringType],
        LongType: [FloatType, DoubleType, DecimalType, StringType],
        FloatType: [DoubleType, StringType],
        DoubleType: [DecimalType, StringType],
        DateType: [TimestampType, StringType],
        TimestampType: [StringType],
    }

def is_safe_promotion(from_type: DataType, to_type: DataType) -> bool:
    """Check if type promotion is safe (no data loss)"""
    if type(from_type) == type(to_type):  # noqa: E721
        return True
    
    allowed: list[type] = TYPE_PROMOTION_HIERARCHY.get(type(from_type), [])
    return type(to_type) in allowed


def safe_int(col: str | Column) -> Column:
    """
    Safely cast a column to IntegerType.
    Handles: integers, floats (truncates), scientific notation, null-like values.
    """
    c: Column = F.col(col) if isinstance(col, str) else col
    c_trim: Column = F.trim(c)
    # First try casting to double (handles scientific notation and decimals)
    # Then cast to int (truncates decimal part)
    return (
        F.when(
            c_trim.isNull()
            | (c_trim == "")
            | (F.lower(c_trim).isin("nan", "none", "null", "inf", "-inf")),
            None,
        )
        .otherwise(c_trim.cast("double").cast("int"))
    )


def safe_long(col: str | Column) -> Column:
    """
    Safely cast a column to LongType.
    Handles: integers, floats (truncates), scientific notation, null-like values.
    """
    c: Column = F.col(col) if isinstance(col, str) else col
    c_trim: Column = F.trim(c)
    # First try casting to double (handles scientific notation and decimals)
    # Then cast to long (truncates decimal part)
    return (
        F.when(
            c_trim.isNull()
            | (c_trim == "")
            | (F.lower(c_trim).isin("nan", "none", "null", "inf", "-inf")),
            None,
        )
        .otherwise(c_trim.cast("double").cast("long"))
    )


def safe_double(col: str | Column) -> Column:
    c: Column = F.col(col) if isinstance(col, str) else col
    c_clean: Column = F.regexp_replace(F.trim(c), ",", "")
    return F.when(
        c_clean.isNull()
        | (c_clean == "")
        | (F.lower(c_clean).isin("nan", "none", "null", "inf", "-inf")),
        None,
    ).otherwise(c_clean.cast("double"))


def safe_timestamp(col: str | Column, fmt: str | None = None) -> Column:
    c: Column = F.col(col) if isinstance(col, str) else col
    c_trim: Column = F.trim(c)
    null_condition: Column = c_trim.isNull() | (c_trim == "")
    
    if fmt is not None:
        return F.when(null_condition, None).otherwise(
            F.to_timestamp(c, fmt).cast("timestamp")
        )
    
    try_builtin: Column = F.try_to_timestamp(c_trim)
    
    c_no_z: Column = F.regexp_replace(c_trim, r"Z$", "")
    
    formats: list[str] = [
        "yyyy-MM-dd'T'HH:mm:ss.SSSSSSS",
        "yyyy-MM-dd'T'HH:mm:ss.SSSSSS",
        "yyyy-MM-dd'T'HH:mm:ss.SSSSS",
        "yyyy-MM-dd'T'HH:mm:ss.SSSS",
        "yyyy-MM-dd'T'HH:mm:ss.SSS",
        "yyyy-MM-dd'T'HH:mm:ss.SS",
        "yyyy-MM-dd'T'HH:mm:ss.S",
        "yyyy-MM-dd'T'HH:mm:ss",
        "yyyy-MM-dd HH:mm:ss",
        "yyyy-MM-dd",
    ]
    
    timestamp_cols: list[Column] = [try_builtin]
    for fmt_str in formats:
        timestamp_cols.append(F.try_to_timestamp(c_trim, F.lit(fmt_str)))
        timestamp_cols.append(F.try_to_timestamp(c_no_z, F.lit(fmt_str)))
    
    result: Column = F.coalesce(*timestamp_cols)
    
    return F.when(null_condition, None).otherwise(
        result.cast("timestamp")
    )


def safe_boolean(col: str | Column) -> Column:
    c: Column = F.col(col) if isinstance(col, str) else col
    lc: Column = F.lower(F.trim(c))
    return (
        F.when(lc.isin("true", "1", "yes"), F.lit(True))
        .when(lc.isin("false", "0", "no"), F.lit(False))
        .otherwise(None)
    )


def safe_string(col: str | Column) -> Column:
    c: Column = F.col(col) if isinstance(col, str) else col
    return F.when(F.lower(F.trim(c)).isin("nan", "none", "null", ""), None).otherwise(
        F.trim(c).cast("string")
    )


def safe_json_array(col: str | Column, schema) -> Column:
    c: Column = F.col(col) if isinstance(col, str) else col
    return F.from_json(F.to_json(c), schema)


def safe_column_rename(
    df: DataFrame, old_name: str, new_name: str, default_value: Any = None
) -> DataFrame:
    if old_name not in df.columns:
        df = df.withColumn(new_name, F.lit(default_value))
        return df
    df = df.withColumn(new_name, F.col(old_name))
    df = df.drop(old_name)
    return df


def safe_drop(df: DataFrame, col: Optional[str] = None, cols: list[str] = []) -> DataFrame:
    if col not in df.columns and len(cols) == 0:
        return df
    
    final_cols: list[str] = [col] if col is not None else [] + cols
    
    for c in final_cols:
        if c in df.columns:
            df = df.drop(c)
    return df

def safe_to_bool(col: str | Column) -> Column:
    c: Column = F.col(col) if isinstance(col, str) else col
    c_trim: Column = F.lower(F.trim(safe_string(c)))
    return (
        F.when(
            F.array_contains(
                F.array(
                    F.lit("true"), F.lit("t"), F.lit("yes"), F.lit("y"), F.lit("1")
                ),
                c_trim,
            ),
            F.lit(True),
        )
        .when(
            F.array_contains(
                F.array(
                    F.lit("false"), F.lit("f"), F.lit("no"), F.lit("n"), F.lit("0")
                ),
                c_trim,
            ),
            F.lit(False),
        )
        .otherwise(None)
    )


def safe_date(col: str | Column, fmt: str = "yyyy-MM-dd") -> Column:
    c: Column = F.col(col) if isinstance(col, str) else col
    return F.when(F.trim(c).isNull() | (F.trim(c) == ""), None).otherwise(
        F.to_date(c, fmt).cast("date")
    )


def safe_to_variant(col: str | Column) -> Column:
    """
    Convert a column to VARIANT, using its *actual* Spark type.

    - For Map/Struct/Array -> to_json(col) -> parse_json(json_str)
    - For String/numerics/bool/etc. -> cast to string and parse_json(string)
    - Treat 'nan'/'none'/'null'/'' and NULL as NULL::VARIANT
    """
    c: Column = F.col(col) if isinstance(col, str) else col

    # Complex types: map/struct/array -> JSON string
    if isinstance(c.dataType, (MapType, StructType, ArrayType)):
        json_str = F.to_json(c)
    else:
        # For simple types, use their string representation
        json_str = c.cast("string")

    null_like = json_str.isNull() | F.lower(F.trim(json_str)).isin(
        "nan", "none", "null", ""
    )

    # parse_json(...) returns VARIANT
    return F.when(null_like, F.lit(None).cast("variant")).otherwise(
        F.parse_json(json_str)
    )

def safe_to_json(col: str | Column) -> Column:
    """
    Safely convert a column (typically MAP/STRUCT/ARRAY or VARIANT) to a JSON string.

    Behaviour:
    - For NULL / 'nan' / 'none' / 'null' / '' -> returns NULL (not the literal "null").
    - For non-null values, calls `to_json(col)` so complex types become proper JSON.
    - If you call this on a primitive (string/int/...), make sure the column is in the
      shape Spark `to_json` expects (e.g. a struct/map/array or variant). For plain
      primitives, prefer `safe_string` or wrap manually with `F.struct(...)`.
    """
    c: Column = F.col(col) if isinstance(col, str) else col

    # Use string representation *only* to detect "null-like" sentinel values.
    c_str: Column = F.trim(c.cast("string"))
    null_like: Column = c_str.isNull() | F.lower(c_str).isin(
        "nan", "none", "null", ""
    )

    # For complex types (struct/map/array/variant) Spark's to_json gives a JSON string.
    json_col: Column = F.to_json(c)

    return F.when(null_like, F.lit(None).cast("string")).otherwise(json_col)


def safe_cast_col(df: DataFrame, col_to_cast: str, cast_type: DataType) -> DataFrame:
    if col_to_cast not in df.columns:
        df = df.withColumn(col_to_cast, F.lit(None))
        return df
    col_to_cast_type: str = df.schema[col_to_cast].dataType.simpleString().upper()

    if col_to_cast_type != cast_type:
        if cast_type == StringType():
            df = df.withColumn(col_to_cast, safe_string(col_to_cast))
        elif cast_type == DoubleType():
            df = df.withColumn(col_to_cast, safe_double(col_to_cast))
        elif cast_type == FloatType():
            df = df.withColumn(col_to_cast, safe_double(col_to_cast))
        elif cast_type == IntegerType():
            df = df.withColumn(col_to_cast, safe_int(col_to_cast))
        elif cast_type == LongType():
            df = df.withColumn(col_to_cast, safe_long(col_to_cast))
        elif cast_type == BooleanType():
            df = df.withColumn(col_to_cast, safe_boolean(col_to_cast))
        elif cast_type == TimestampType():
            df = df.withColumn(col_to_cast, safe_timestamp(col_to_cast))
        elif cast_type == DateType():
            df = df.withColumn(col_to_cast, safe_date(col_to_cast))
        elif cast_type == VariantType():
            df = df.withColumn(col_to_cast, safe_to_variant(col_to_cast))
    return df

def safe_check_if_data_exists(df: DataFrame, logger: ILogger | Logger = safe_get_logger()) -> bool:
    try:
        is_empty: bool = df.rdd.isEmpty()
    except Exception as e:
        logger.info(f"[_check_if_data_exists] Error checking if data exists: {e}")
        is_empty = df.limit(1).count() == 0
    if is_empty:
        return False
    return True

def safe_if_empty(
    df: DataFrame, 
    logger: ILogger | Logger = safe_get_logger(), 
    cache: bool = False
) -> bool:
    """
    Check if a DataFrame is empty.

    Behavior:
    - If DataFrame is empty: returns True
    - If DataFrame is not empty: returns False
    """
    if cache:
        df.cache()
    try:
        is_empty: bool = df.rdd.isEmpty()     
    except (AttributeError, Exception) as e:  # noqa: F841
        try:
            c: int = df.limit(1).count()
            is_empty = c == 0
        except (AttributeError, Exception) as e2:
            logger.warning(f"[safe_if_empty] rdd.isEmpty() also failed, trying count(): {e2}")
            try:
                is_empty = df.count() == 0
            except Exception as e3:
                logger.error(f"[safe_if_empty] All methods failed to check if DataFrame is empty: {e3}")
                raise ValueError(f"[safe_if_empty] All methods failed to check if DataFrame is empty: {e3}")
    if cache:
        df.unpersist()
    return is_empty

def safe_deduplicate(df: DataFrame,subset:list[str]) -> DataFrame:
    """
    Deduplicate a DataFrame by dropping duplicates on all non-MapType columns.
    MapType columns cannot be used in dropDuplicates, so they are excluded.
    """
    non_map_cols: list[str] = [
        c for c in subset
        if not isinstance(df.schema[c].dataType, MapType)
    ]
    return df.dropDuplicates(subset=non_map_cols)

def safe_to_percent(col: str | Column, clamp: bool = False) -> Column:
    """
    Safely convert a percentage value to a decimal (0-1).
    
    If value > 1, assumes it's a percentage (0-100) and divides by 100.
    If value is already between 0-1, keeps it as is.
    
    Args:
        col: Column containing percentage values
        clamp: If True, clamp result to [0, 1] range
    
    Returns:
        Column with values between 0 and 1
    """
    c: Column = F.col(col) if isinstance(col, str) else col
    c_clean: Column = F.regexp_replace(F.trim(c.cast("string")), ",", "")
    
    value: Column = F.when(
        c_clean.isNull()
        | (c_clean == "")
        | (F.lower(c_clean).isin("nan", "none", "null", "inf", "-inf")),
        None,
    ).otherwise(c_clean.cast("double"))
    
    result: Column = F.when(value.isNull(), None).otherwise(
        F.when(value > 1, value / 100).otherwise(value)
    )
    
    if clamp:
        result = F.when(result.isNull(), None).otherwise(
            F.greatest(F.lit(0.0), F.least(F.lit(1.0), result))
        )
    
    return result