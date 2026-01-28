from __future__ import annotations

import logging
from typing import Any, Callable, Literal, Optional

import pyspark.sql.functions as F
from pyspark.sql.column import Column
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import (
    ArrayType,
    DataType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
    StringType,
    StructField,
    StructType,
)
from pyspark_schema_evaluation.utils._spark._safe import safe_cast_col, safe_drop
from pyspark_schema_evaluation.utils._spark._schema import _compare_schema, _diff_schemas
from pyspark_schema_evaluation.core.exceptions import SchemaEvolutionError
from pyspark_schema_evaluation.core._logging import get_logger, safe_log
from pyspark_schema_evaluation.models.models import (
    SchemaValidationResult,
    SchemaDiff,
    SchemaMetadata,
    DriftDetectionResult,
    BreakingChangeReport,
    BreakingChange,
    SchemaDriftAlert,
    DriftSeverity,
    DriftType,
    BreakingChangeType,
    LayerPolicyConfig,
    ColumnMetadata,
)
from pyspark_schema_evaluation.core._enums import LayerPolicy, CompatibilityMode, ConflictMode
from pyspark_schema_evaluation.i_schema_evolution import ISchemaEvolution
from pyspark_schema_evaluation.registry.i_schema_registry import ISchemaRegistry
from pyspark_schema_evaluation.utils._utils import (
    _handle_extra_columns_with_tracking,
)
from pyspark_schema_evaluation.utils._result import Err, Ok, Result
from pyspark_schema_evaluation.utils._spark._safe import is_safe_promotion
from pyspark_schema_evaluation.utils._utils import _build_compatibility_report
from pyspark_schema_evaluation.utils._spark._schema import _add_column_with_default
from pyspark_schema_evaluation.utils._utils import drop_extra_columns


__all__: list[str] = [
    "ISchemaEvolution",
    "SchemaEvolution"
]


# Type narrowing map: defines which type conversions can cause data loss
# Maps target_type -> list of source_types that would be narrowed
_TYPE_NARROWING_MAP: dict[type, list[type]] = {
    IntegerType: [LongType, FloatType, DoubleType],
    ShortType: [IntegerType, LongType, FloatType, DoubleType],
    FloatType: [DoubleType],
}

class SchemaEvolution(ISchemaEvolution):
    """
    Handles schema evolution operations for DataFrames in a medallion architecture.

    Supports:
    - Merging schemas with conflict resolution
    - Evolving DataFrames to target schemas with compatibility modes
    - Layer-specific policies (Bronze/Silver/Gold)
    - Schema versioning and registry integration
    - Nested schema evolution
    - Transaction-like rollback on failure
    
    Example:
        >>> evolution = SchemaEvolution(
        ...     registry=InMemorySchemaRegistry(),
        ...     default_layer=LayerPolicy.SILVER,
        ... )
        >>> result = evolution.evolve_for_layer(df, target_schema, LayerPolicy.SILVER)
        
    Logging:
        The library is silent by default (pandas pattern). Users can enable logging:
        
        >>> import logging
        >>> logging.basicConfig(level=logging.INFO)
        >>> evolution = SchemaEvolution()  # Now logs will appear
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        registry: Optional[ISchemaRegistry] = None,
        default_layer: LayerPolicy = LayerPolicy.SILVER,
    ) -> None:
        """
        Initialize SchemaEvolution.
        
        Args:
            logger: Optional logger instance. If None, uses module-level logger
                   (which is silent unless user configures logging via logging.basicConfig()).
            registry: Optional schema registry for versioning
            default_layer: Default medallion layer policy
            
        Example:
            >>> # Silent by default (pandas style)
            >>> evolution = SchemaEvolution()
            
            >>> # User can enable logging if they want
            >>> import logging
            >>> logging.basicConfig(level=logging.INFO)
            >>> evolution = SchemaEvolution()  # Now logs will appear
            
            >>> # Or use custom logger
            >>> custom_logger = logging.getLogger("my_app")
            >>> evolution = SchemaEvolution(logger=custom_logger)
        """
        # Use provided logger or fall back to centralized module logger
        self._logger: logging.Logger = logger if logger is not None else get_logger()
        self._registry: ISchemaRegistry | None = registry
        self._default_layer: LayerPolicy = default_layer

    @property
    def registry(self) -> Optional[ISchemaRegistry]:
        return self._registry

    def evolve_for_layer(
        self,
        df: DataFrame,
        target_schema: StructType,
        layer: Optional[LayerPolicy] = None,
        default_values: Optional[dict[str, Any]] = None,
        table_name: Optional[str] = None,
        strict_compatibility: bool = False,
        custom_policy: Optional[str] = None,
    ) -> SchemaValidationResult:
        """
        Evolve DataFrame using layer-specific policies.
        
        This is the recommended entry point for medallion architecture pipelines.
        
        Args:
            df: Source DataFrame
            target_schema: Target schema to evolve towards
            layer: Medallion layer (BRONZE, SILVER, GOLD, CUSTOM). Uses default if not specified.
            default_values: Custom default values for new columns
            table_name: Optional table name for registry operations
            strict_compatibility: If True, raise error on any incompatible changes (not just GOLD)
            custom_policy: Name of custom policy (required if layer=CUSTOM).
                          Register custom policies with LayerPolicyRegistry.register()
            
        Returns:
            SchemaValidationResult with evolved DataFrame and change tracking
            
        Example with custom policy:
            # First register the policy
            LayerPolicyRegistry.register("STAGING", LayerPolicyConfig(...))
            
            # Then use it
            result = evolution.evolve_for_layer(
                df=df,
                target_schema=schema,
                layer=LayerPolicy.CUSTOM,
                custom_policy="STAGING",
            )
        """
        final_layer: LayerPolicy = layer or self._default_layer
        policy: LayerPolicyConfig = LayerPolicyConfig.from_layer(final_layer, custom_policy)
        
        policy_name = custom_policy if custom_policy else final_layer.value
        safe_log(self._logger, 'info', f"Evolving schema for layer={policy_name} with policy={policy}")
        
        # Check compatibility with registry if available
        if self._registry and table_name:
            compatibility_result = self._check_and_log_compatibility(
                table_name=table_name,
                target_schema=target_schema,
                layer=final_layer,
                strict=strict_compatibility,
            )
            if compatibility_result is not None:
                # Return early with error result if compatibility check failed
                return compatibility_result
        
        result: SchemaValidationResult = self.evolve_df_to_target_schema_with_report(
            df=df,
            target_schema=target_schema,
            compatibility_mode=policy.compatibility_mode,
            extra_allowed=policy.extra_columns_allowed,
            default_values=default_values,
        )
        if result.had_changes:
            safe_log(self._logger, 'info', f"[{self}] Schema evolution had changes: {result.summary()}")
        else:
            safe_log(self._logger, 'info', f"[{self}] Schema evolution had no changes")
        
        # Register new schema version if registry available
        if self._registry and table_name and result.had_changes:
            self._registry.register_schema(
                table_name=table_name,
                schema=result.df.schema,
                layer=final_layer,
                description=f"Auto-evolved: {result.summary()}",
            )
        
        return result
    
    def _check_and_log_compatibility(
        self,
        table_name: str,
        target_schema: StructType,
        layer: LayerPolicy,
        strict: bool = False,
    ) -> Optional[SchemaValidationResult]:
        """
        Check schema compatibility with registry and log results.
        
        Args:
            table_name: Registry table name
            target_schema: New target schema
            layer: Medallion layer
            strict: If True, raise error on any incompatibility
            
        Returns:
            None if compatible, SchemaValidationResult with error if not
            
        Raises:
            SchemaEvolutionError: If GOLD layer or strict mode and incompatible
        """
        if self._registry is None:
            return None
        
        is_compatible, diff = self._registry.check_compatibility(table_name, target_schema)
        
        if is_compatible:
            safe_log(self._logger, 'info', f"[{self}] Schema compatibility check PASSED for '{table_name}'")
            return None
        
        # Build detailed compatibility report
        compatibility_report: str = _build_compatibility_report(table_name, diff)
        
        # GOLD layer: Always fail on incompatibility
        if layer == LayerPolicy.GOLD:
            error_msg = f"[{self}] Schema compatibility check FAILED for GOLD layer:\n{compatibility_report}"
            safe_log(self._logger, 'error', error_msg)
            raise SchemaEvolutionError(
                f"Schema incompatible with registry for GOLD layer '{table_name}':\n"
                f"{compatibility_report}"
            )
        
        # Strict mode: Fail on any incompatibility
        if strict:
            error_msg = f"[{self}] Schema compatibility check FAILED (strict mode):\n{compatibility_report}"
            safe_log(self._logger, 'error', error_msg)
            raise SchemaEvolutionError(
                f"Schema incompatible with registry for '{table_name}' (strict mode):\n"
                f"{compatibility_report}"
            )
        
        # SILVER layer: Warn but continue (backwards compatible changes allowed)
        if layer == LayerPolicy.SILVER:
            if diff and diff.missing_columns:
                # Missing columns = breaking change, warn strongly
                safe_log(
                    self._logger, 'warning',
                    f"[{self}] ⚠️  BREAKING CHANGE detected for SILVER layer '{table_name}':\n"
                    f"   Missing columns (will be added with NULL): {diff.missing_columns}\n"
                    f"   This may cause issues with downstream consumers."
                )
            if diff and diff.type_mismatches:
                safe_log(
                    self._logger, 'warning',
                    f"[{self}] ⚠️  TYPE CHANGES detected for '{table_name}':\n"
                    f"   {diff.type_mismatches}\n"
                    f"   Data may be cast, potential data loss."
                )
        
        # BRONZE layer: Info only (permissive)
        if layer == LayerPolicy.BRONZE:
            safe_log(
                self._logger, 'info',
                f"[{self}] Schema changes detected for BRONZE layer '{table_name}' "
                f"(permissive mode, continuing):\n{compatibility_report}"
            )
        
        return None
        
    def check_compatibility(
        self,
        table_name: str,
        new_schema: StructType,
    ) -> tuple[bool, Any, str]:
        """
        Explicitly check schema compatibility without evolving.
        
        Use this before making schema changes to see what would happen.
        
        Args:
            table_name: Registry table name
            new_schema: Proposed new schema
            
        Returns:
            Tuple of (is_compatible, diff, report_string)
            
        Example:
            >>> is_ok, diff, report = evolution.check_compatibility("silver.raw_cur", new_schema)
            >>> if not is_ok:
            ...     print(f"Breaking changes:\\n{report}")
        """
        if self._registry is None:
            return True, None, "No registry configured"
        
        is_compatible, diff = self._registry.check_compatibility(table_name, new_schema)
        report: str = _build_compatibility_report(table_name, diff)
        
        return is_compatible, diff, report

    def evolve_bronze_to_silver(
        self,
        df: DataFrame,
        silver_schema: StructType,
        default_values: Optional[dict[str, Any]] = None,
        table_name: Optional[str] = None,
    ) -> SchemaValidationResult:
        """
        Convenience method for Bronze -> Silver transformation.
        
        Applies BACKWARDS compatibility:
        - Adds missing columns as nullable with defaults
        - Drops extra columns not in silver schema
        - Safe type promotions
        """
        return self.evolve_for_layer(
            df=df,
            target_schema=silver_schema,
            layer=LayerPolicy.SILVER,
            default_values=default_values,
            table_name=table_name,
        )

    def evolve_silver_to_gold(
        self,
        df: DataFrame,
        gold_schema: StructType,
        table_name: Optional[str] = None,
    ) -> SchemaValidationResult:
        """
        Convenience method for Silver -> Gold transformation.
        
        Applies STRICT compatibility:
        - No schema changes allowed
        - Raises error on any mismatch
        """
        return self.evolve_for_layer(
            df=df,
            target_schema=gold_schema,
            layer=LayerPolicy.GOLD,
            table_name=table_name,
        )

    def evolve_with_rollback(
        self,
        df: DataFrame,
        target_schema: StructType,
        compatibility_mode: CompatibilityMode = CompatibilityMode.BACKWARDS,
        extra_allowed: bool = False,
        default_values: Optional[dict[str, Any]] = None,
        validation_fn: Optional[Callable[[DataFrame], bool]] = None,
    ) -> SchemaValidationResult:
        """
        Evolve DataFrame with rollback capability on failure.
        
        Args:
            df: Source DataFrame
            target_schema: Target schema
            compatibility_mode: How to handle evolution
            extra_allowed: Whether to keep extra columns
            default_values: Default values for new columns
            validation_fn: Optional function to validate result. If returns False, rollback.
            
        Returns:
            SchemaValidationResult with evolved DataFrame
            
        Raises:
            SchemaEvolutionError: If evolution fails or validation_fn returns False
        """
        original_schema: StructType = df.schema
        
        try:
            result: SchemaValidationResult = self.evolve_df_to_target_schema_with_report(
                df=df,
                target_schema=target_schema,
                compatibility_mode=compatibility_mode,
                extra_allowed=extra_allowed,
                default_values=default_values,
            )
            
            # Run custom validation if provided
            if validation_fn is not None:
                if not validation_fn(result.df):
                    raise SchemaEvolutionError(
                        "Custom validation failed after schema evolution"
                    )
            
            return result
            
        except Exception as e:
            error_msg = (
                f"Schema evolution failed, original schema preserved. "
                f"Original: {original_schema.simpleString()}, "
                f"Target: {target_schema.simpleString()}, "
                f"Error: {e}"
            )
            safe_log(self._logger, 'error', error_msg)
            # Return original DataFrame unchanged
            return SchemaValidationResult(
                df=df,
                warnings=[f"Evolution failed, returned original: {str(e)}"],
            )

    def evolve_nested_schema(
        self,
        df: DataFrame,
        target_schema: StructType,
        compatibility_mode: CompatibilityMode = CompatibilityMode.BACKWARDS,
    ) -> DataFrame:
        """
        Recursively evolve nested schemas (structs and arrays of structs).
        
        Args:
            df: Source DataFrame
            target_schema: Target schema with nested structures
            compatibility_mode: How to handle evolution
            
        Returns:
            DataFrame with evolved nested schemas
        """
        for f in target_schema.fields:
            if f.name in df.columns:
                df = self._evolve_nested_field_complete(
                    df=df,
                    field=f,
                    compatibility_mode=compatibility_mode,
                )
            elif compatibility_mode in (CompatibilityMode.BACKWARDS, CompatibilityMode.FULL):
                # Add missing nested field with null
                df = df.withColumn(
                    f.name,
                    F.lit(None).cast(f.dataType)
                )
        
        return df

    def _evolve_nested_field_complete(
        self,
        df: DataFrame,
        field: StructField,
        compatibility_mode: CompatibilityMode,
        parent_path: str = "",
    ) -> DataFrame:
        """Recursively handle nested schema evolution for structs and arrays"""
        col_name: str = field.name
        # full_path: str = f"{parent_path}.{col_name}" if parent_path else col_name
        
        if isinstance(field.dataType, StructType):
            df = self._evolve_struct_field(df, col_name, field.dataType, compatibility_mode)
            
        elif isinstance(field.dataType, ArrayType):
            element_type: DataType = field.dataType.elementType
            if isinstance(element_type, StructType):
                df = self._evolve_array_of_structs(df, col_name, element_type, compatibility_mode)
        
        return df

    def _evolve_struct_field(
        self,
        df: DataFrame,
        col_name: str,
        target_struct: StructType,
        compatibility_mode: CompatibilityMode,
    ) -> DataFrame:
        """Evolve a struct column to match target schema."""
        # Get current struct schema
        current_field = df.schema[col_name]
        if not isinstance(current_field.dataType, StructType):
            raise SchemaEvolutionError(
                f"Expected struct type for {col_name}, got {current_field.dataType}"
            )
        
        current_struct: StructType = current_field.dataType
        current_fields: dict[str, StructField] = {f.name: f for f in current_struct.fields}
        target_fields: dict[str, StructField] = {f.name: f for f in target_struct.fields}
        
        struct_exprs: list[Column] = []
        
        for name, target_field in target_fields.items():
            if name in current_fields:
                current_type: DataType = current_fields[name].dataType
                target_type: DataType = target_field.dataType
                
                if current_type != target_type:
                    if is_safe_promotion(current_type, target_type):
                        struct_exprs.append(
                            F.col(f"{col_name}.{name}").cast(target_type).alias(name)
                        )
                    else:
                        safe_log(
                            self._logger, 'warning',
                            f"Unsafe nested cast {col_name}.{name}: "
                            f"{current_type} -> {target_type}"
                        )
                        struct_exprs.append(
                            F.col(f"{col_name}.{name}").cast(target_type).alias(name)
                        )
                else:
                    struct_exprs.append(F.col(f"{col_name}.{name}").alias(name))
            else:
                # Missing field - add with null if backwards compatible
                if compatibility_mode in (CompatibilityMode.BACKWARDS, CompatibilityMode.FULL):
                    struct_exprs.append(
                        F.lit(None).cast(target_field.dataType).alias(name)
                    )
                else:
                    raise SchemaEvolutionError(
                        f"Missing nested field {col_name}.{name} in {compatibility_mode} mode"
                    )
        
        # Rebuild the struct column
        new_struct: Column = F.struct(*struct_exprs)
        df = df.withColumn(col_name, new_struct)
        
        return df

    def _evolve_array_of_structs(
        self,
        df: DataFrame,
        col_name: str,
        target_element_type: StructType,
        compatibility_mode: CompatibilityMode,
    ) -> DataFrame:
        """Evolve an array of structs column using transform."""
        # Get current array element schema
        current_field: StructField = df.schema[col_name]
        if not isinstance(current_field.dataType, ArrayType):
            raise SchemaEvolutionError(
                f"Expected array type for {col_name}, got {current_field.dataType}"
            )
        
        current_element = current_field.dataType.elementType
        if not isinstance(current_element, StructType):
            raise SchemaEvolutionError(
                f"Expected array of structs for {col_name}, got array of {current_element}"
            )
        
        current_fields: dict[str, StructField] = {f.name: f for f in current_element.fields}
        target_fields: dict[str, StructField] = {f.name: f for f in target_element_type.fields}
        
        # Build transform expression for each element
        def build_element_transform() -> Column:
            struct_exprs: list = []
            
            for name, target_field in target_fields.items():
                if name in current_fields:
                    current_type: DataType = current_fields[name].dataType
                    target_type: DataType = target_field.dataType
                    
                    if current_type != target_type:
                        struct_exprs.append(
                            F.element_at(F.col("x"), name).cast(target_type).alias(name)
                        )
                    else:
                        struct_exprs.append(F.col(f"x.{name}").alias(name))
                else:
                    if compatibility_mode in (CompatibilityMode.BACKWARDS, CompatibilityMode.FULL):
                        struct_exprs.append(
                            F.lit(None).cast(target_field.dataType).alias(name)
                        )
                    else:
                        raise SchemaEvolutionError(
                            f"Missing field {name} in array element for {compatibility_mode} mode"
                        )
            
            return F.struct(*struct_exprs)
        
        # Apply transform to array
        df = df.withColumn(
            col_name,
            F.transform(F.col(col_name), lambda x: build_element_transform())
        )
        
        return df


    def get_delta_merge_options(
        self,
        layer: LayerPolicy,
    ) -> dict[str, str]:
        """
        Get recommended Delta Lake merge options for a layer.
        
        Returns Spark conf options for Delta schema evolution.
        """
        match layer:
            case LayerPolicy.BRONZE:
                return {
                    "spark.databricks.delta.schema.autoMerge.enabled": "true",
                    "spark.databricks.delta.properties.defaults.autoOptimize.optimizeWrite": "true",
                }
            case LayerPolicy.SILVER:
                return {
                    "spark.databricks.delta.schema.autoMerge.enabled": "true",
                    "spark.databricks.delta.properties.defaults.autoOptimize.optimizeWrite": "true",
                }
            case LayerPolicy.GOLD:
                return {
                    "spark.databricks.delta.schema.autoMerge.enabled": "false",
                }
            case _:
                return {}

    def prepare_for_delta_write(
        self,
        df: DataFrame,
        target_schema: StructType,
        layer: LayerPolicy,
        use_delta_auto_merge: bool = False,
    ) -> tuple[DataFrame, dict[str, Any]]:
        """
        Prepare DataFrame for Delta Lake write with appropriate options.
        
        Args:
            df: Source DataFrame
            target_schema: Expected table schema
            layer: Medallion layer
            use_delta_auto_merge: If True, rely on Delta's autoMerge instead of manual evolution
            
        Returns:
            Tuple of (evolved DataFrame, write options dict)
        """
        write_options: dict[str, Any] = {}
        
        if use_delta_auto_merge:
            write_options["mergeSchema"] = "true"
            return df, write_options
        
        result: SchemaValidationResult = self.evolve_for_layer(df, target_schema, layer)
        
        if layer == LayerPolicy.GOLD:
            write_options["overwriteSchema"] = "false"
        
        return result.df, write_options

    def merge_schemas(
        self,
        schema_1: StructType,
        schema_2: StructType,
        on_conflict: ConflictMode = ConflictMode.STRING,
    ) -> StructType:
        """
        Merge two StructTypes with conflict resolution.

        Args:
            schema_1: Base schema
            schema_2: Schema to merge in
            on_conflict: How to handle type conflicts between schemas

        Returns:
            Merged StructType
        """
        merged: dict[str, StructField] = {f.name: f for f in schema_1.fields}

        for f in schema_2.fields:
            if f.name not in merged:
                merged[f.name] = f
            else:
                resolved: StructField | None = self._resolve_field_conflict(
                    merged[f.name], f, on_conflict
                )
                if resolved is None:
                    merged.pop(f.name)
                else:
                    merged[f.name] = resolved

        return StructType(list[StructField](merged.values()))

    def evolve_df_to_target_schema(
        self,
        df: DataFrame,
        target_schema: StructType,
        compatibility_mode: CompatibilityMode = CompatibilityMode.BACKWARDS,
        extra_allowed: bool = False,
        default_values: Optional[dict[str, Any]] = None,
    ) -> DataFrame:
        """
        Evolve DataFrame to match target schema under specified compatibility mode.

        Args:
            df: Source DataFrame
            target_schema: Target schema to evolve towards
            compatibility_mode: STRICT, BACKWARDS, FORWARDS, or FULL
            extra_allowed: Whether to keep extra columns not in target
            default_values: Custom default values for new columns (col_name -> value)

        Returns:
            Evolved DataFrame
        """
        match compatibility_mode:
            case CompatibilityMode.STRICT:
                return self._evolve_strict(df, target_schema)
            case CompatibilityMode.FULL:
                return self._evolve_full(df, target_schema, extra_allowed, default_values)
            case CompatibilityMode.BACKWARDS:
                return self._evolve_backwards(df, target_schema, extra_allowed, default_values)
            case CompatibilityMode.FORWARDS:
                return self._evolve_forwards(df, target_schema, extra_allowed)
            case _:
                raise ValueError(f"Unknown compatibility mode: {compatibility_mode}")

    def evolve_df_to_target_schema_with_report(
        self,
        df: DataFrame,
        target_schema: StructType,
        compatibility_mode: CompatibilityMode = CompatibilityMode.BACKWARDS,
        extra_allowed: bool = False,
        default_values: Optional[dict[str, Any]] = None,
    ) -> SchemaValidationResult:
        """
        Evolve DataFrame to match target schema and return detailed report.

        Args:
            df: Source DataFrame
            target_schema: Target schema to evolve towards
            compatibility_mode: STRICT, BACKWARDS, FORWARDS, or FULL
            extra_allowed: Whether to keep extra columns not in target
            default_values: Custom default values for new columns

        Returns:
            SchemaValidationResult with evolved DataFrame and change tracking
        """
        result: SchemaValidationResult = SchemaValidationResult(df=df)

        match compatibility_mode:
            case CompatibilityMode.STRICT:
                df = self._evolve_strict(df, target_schema)
            case CompatibilityMode.FULL:
                df, result = self._evolve_full_with_tracking(
                    df, target_schema, extra_allowed, default_values, result
                )
            case CompatibilityMode.BACKWARDS:
                df, result = self._evolve_backwards_with_tracking(
                    df, target_schema, extra_allowed, default_values, result
                )
            case CompatibilityMode.FORWARDS:
                df, result = self._evolve_forwards_with_tracking(
                    df, target_schema, extra_allowed, result
                )
            case _:
                raise ValueError(f"Unknown compatibility mode: {compatibility_mode}")

        result.df = df
        return result

    def compare_schema(
        self,
        expected_schema: StructType,
        actual_schema: StructType,
    ) -> bool:
        """Compare two schemas for structural equality."""
        return _compare_schema(
            actual_schema=actual_schema,
            expected_schema=expected_schema,
        )

    def validate_schema(
        self,
        df: DataFrame,
        expected_schema: StructType,
        extra_columns_conflict: Literal[ConflictMode.IGNORE, ConflictMode.DROP] = ConflictMode.IGNORE,
        on_conflict: ConflictMode = ConflictMode.STRING,
        columns_to_ignore: Optional[list[str]] = None,
        columns_to_add: Optional[StructType] = None,
        conflict_mode_for_columns_to_add: ConflictMode = ConflictMode.SECOND,
    ) -> DataFrame:
        """
        Validate DataFrame against expected schema and apply transformations.

        Args:
            df: DataFrame to validate
            expected_schema: Schema to validate against
            extra_columns_conflict: How to handle columns not in expected schema
            on_conflict: How to handle type mismatches
            columns_to_ignore: Columns to skip during validation
            columns_to_add: Additional columns to include in final schema
            conflict_mode_for_columns_to_add: Conflict mode when merging additional columns

        Returns:
            Validated and transformed DataFrame
        """
        columns_to_ignore = columns_to_ignore or []

        final_schema: StructType = self._build_final_schema(
            expected_schema, columns_to_add, conflict_mode_for_columns_to_add
        )

        df = self._validate_expected_columns(df, final_schema, columns_to_ignore, on_conflict)
        df = self._handle_extra_columns(df, final_schema, extra_columns_conflict)

        return df

    def validate_schema_with_report(
        self,
        df: DataFrame,
        expected_schema: StructType,
        extra_columns_conflict: Literal[ConflictMode.IGNORE, ConflictMode.DROP] = ConflictMode.IGNORE,
        on_conflict: ConflictMode = ConflictMode.STRING,
        columns_to_ignore: Optional[list[str]] = None,
        columns_to_add: Optional[StructType] = None,
        conflict_mode_for_columns_to_add: ConflictMode = ConflictMode.SECOND,
    ) -> SchemaValidationResult:
        """
        Validate DataFrame and return detailed result with metrics.

        Same as validate_schema but returns SchemaValidationResult with tracking info.
        """
        final_columns_to_ignore: list[str] = columns_to_ignore or []
        result: SchemaValidationResult = SchemaValidationResult(df=df)

        final_schema = self._build_final_schema(
            expected_schema, columns_to_add, conflict_mode_for_columns_to_add
        )

        df, result = self._validate_expected_columns_with_tracking(
            df, final_schema, final_columns_to_ignore, on_conflict, result
        )
        df, result = _handle_extra_columns_with_tracking(
            df, final_schema, extra_columns_conflict, result
        )

        result.df = df
        return result

    def reorder_df_to_schema(
        self,
        df: DataFrame,
        target_schema: StructType,
        drop_extra: bool = False,
        add_default_values: bool = False,
    ) -> DataFrame:
        """
        Reorder DataFrame columns to match target schema order.

        Args:
            df: DataFrame to reorder
            target_schema: Schema defining desired column order
            drop_extra: Whether to drop columns not in target schema
            add_default_values: Whether to add missing columns with null values

        Returns:
            Reordered DataFrame
        """
        select_exprs: list[Column] = self._build_select_expressions(
            df, target_schema, add_default_values
        )

        if not drop_extra:
            extra_cols: list[str] = self._get_extra_columns(df, target_schema)
            select_exprs.extend(F.col(c) for c in extra_cols)

        return df.select(*select_exprs)

    def diff_schemas(
        self,
        actual_schema: StructType,
        expected_schema: StructType,
    ) -> SchemaDiff:
        return _diff_schemas(
            actual_schema=actual_schema,
            expected_schema=expected_schema,
        )

    def preview_evolution(
        self,
        df: DataFrame,
        target_schema: StructType,
        compatibility_mode: CompatibilityMode = CompatibilityMode.BACKWARDS,
    ) -> SchemaDiff:
        """Preview what changes would be made without applying them."""
        return self.diff_schemas(df.schema, target_schema)

    def validate_type_compatibility(
        self,
        actual_schema: StructType,
        target_schema: StructType,
    ) -> dict[str, str]:
        """
        Check all type changes and return warnings for unsafe promotions.

        Returns:
            Dict of column_name -> warning message for unsafe casts
        """
        warnings: dict[str, str] = {}

        actual: dict[str, StructField] = {f.name: f for f in actual_schema.fields}
        target: dict[str, StructField] = {f.name: f for f in target_schema.fields}

        for name in actual.keys() & target.keys():
            actual_type: DataType = actual[name].dataType
            target_type: DataType = target[name].dataType

            if actual_type != target_type and not is_safe_promotion(actual_type, target_type):
                warnings[name] = (
                    f"Unsafe: {actual_type.simpleString()} -> {target_type.simpleString()}"
                )

        return warnings

    def _evolve_strict(self, df: DataFrame, target_schema: StructType) -> DataFrame:
        """Strict mode: no changes allowed, raise on any mismatch"""
        actual: set[str] = set[str](df.schema.fieldNames())
        expected: set[str] = set[str](target_schema.fieldNames())

        missing: set[str] = expected - actual
        extra: set[str] = actual - expected
        conflicts: list[str] = self._find_type_conflicts(df.schema, target_schema)

        if missing or extra or conflicts:
            raise SchemaEvolutionError(
                f"Schema mismatch in strict mode: "
                f"Missing={missing}, Extra={extra}, TypeConflicts={conflicts}"
            )
        return df

    def _evolve_full(
        self,
        df: DataFrame,
        target_schema: StructType,
        extra_allowed: bool,
        default_values: Optional[dict[str, Any]],
    ) -> DataFrame:
        """Full mode: apply both backwards and forwards compatibility"""
        df = self._evolve_backwards(df, target_schema, extra_allowed, default_values)
        df = self._evolve_forwards(df, target_schema, extra_allowed)
        return df

    def _evolve_backwards(
        self,
        df: DataFrame,
        target_schema: StructType,
        extra_allowed: bool,
        default_values: Optional[dict[str, Any]] = None,
    ) -> DataFrame:
        """Backwards mode: old data reading new schema (add missing as nullable)"""
        final_default_values = default_values or {}
        existing: dict[str, StructField] = {f.name: f for f in df.schema.fields}
        target: dict[str, StructField] = {f.name: f for f in target_schema.fields}

        # Add missing columns
        for name, f in target.items():
            if name not in existing:
                df = _add_column_with_default(df, name, f.dataType, final_default_values)

        # Handle extra columns
        if not extra_allowed:
            df = drop_extra_columns(df, existing, target)

        return df

    def _evolve_forwards(
        self,
        df: DataFrame,
        target_schema: StructType,
        extra_allowed: bool,
    ) -> DataFrame:
        """Forwards mode: new data reading old schema (cast types, fail on missing required)."""
        existing: dict[str, StructField] = {f.name: f for f in df.schema.fields}
        target: dict[str, StructField] = {f.name: f for f in target_schema.fields}

        for name, f in target.items():
            if name not in existing:
                raise SchemaEvolutionError(
                    f"Column {name!r} required by target schema but missing "
                    f"in DataFrame for FORWARDS compatibility"
                )

            actual_type: DataType = existing[name].dataType
            expected_type: DataType = f.dataType

            if actual_type != expected_type:
                if not is_safe_promotion(actual_type, expected_type):
                    safe_log(
                        self._logger, 'warning',
                        f"Unsafe type cast for '{name}': "
                        f"{actual_type.simpleString()} -> {expected_type.simpleString()}"
                    )
                df = safe_cast_col(df, name, expected_type)

        if not extra_allowed:
            df = drop_extra_columns(df, existing, target)

        return df

    def _resolve_field_conflict(
        self,
        field1: StructField,
        field2: StructField,
        on_conflict: ConflictMode,
    ) -> Optional[StructField]:
        """Resolve conflict between two fields with the same name"""
        match on_conflict:
            case ConflictMode.STRING:
                return StructField(field2.name, StringType(), True)
            case ConflictMode.FIRST:
                return field1
            case ConflictMode.SECOND:
                return field2
            case ConflictMode.IGNORE:
                return None
            case _:
                raise ValueError(f"Invalid ConflictMode: {on_conflict}")

    def _build_final_schema(
        self,
        expected_schema: StructType,
        columns_to_add: Optional[StructType],
        conflict_mode: ConflictMode,
    ) -> StructType:
        """Build final expected schema including any additional columns"""
        if columns_to_add is None:
            return expected_schema
        return self.merge_schemas(expected_schema, columns_to_add, conflict_mode)

    def _validate_expected_columns(
        self,
        df: DataFrame,
        expected_schema: StructType,
        columns_to_ignore: list[str],
        on_conflict: ConflictMode,
    ) -> DataFrame:
        """Validate and transform columns that should exist in the DataFrame."""
        df_columns: set[str] = set[str](df.schema.fieldNames())

        for f in expected_schema.fields:
            if f.name in columns_to_ignore:
                continue

            if f.name not in df_columns:
                raise SchemaEvolutionError(
                    f"Column {f.name} is missing in the DataFrame"
                )

            df = self._handle_type_mismatch(df, f, on_conflict)

        return df

    def _validate_expected_columns_with_tracking(
        self,
        df: DataFrame,
        expected_schema: StructType,
        columns_to_ignore: list[str],
        on_conflict: ConflictMode,
        result: SchemaValidationResult,
    ) -> tuple[DataFrame, SchemaValidationResult]:
        """Validate columns with change tracking for reporting."""
        df_columns: set[str] = set[str](df.schema.fieldNames())

        for f in expected_schema.fields:
            if f.name in columns_to_ignore:
                continue

            if f.name not in df_columns:
                raise SchemaEvolutionError(
                    f"Column {f.name} is missing in the DataFrame"
                )

            actual_type: DataType = df.schema[f.name].dataType
            if actual_type != f.dataType:
                df = self._handle_type_mismatch(df, f, on_conflict)
                result.columns_cast[f.name] = (
                    actual_type.simpleString(),
                    f.dataType.simpleString(),
                )

        return df, result

    def _handle_type_mismatch(
        self,
        df: DataFrame,
        expected_field: StructField,
        on_conflict: ConflictMode,
    ) -> DataFrame:
        """Handle type mismatch between DataFrame column and expected field."""
        actual_type: DataType = df.schema[expected_field.name].dataType
        expected_type: DataType = expected_field.dataType

        if actual_type == expected_type:
            return df

        is_safe: bool = is_safe_promotion(actual_type, expected_type)

        safe_log(
            self._logger, 'info',
            f"Column '{expected_field.name}' type mismatch: "
            f"{actual_type.simpleString()} -> {expected_type.simpleString()} "
            f"({'safe' if is_safe else 'UNSAFE'})"
        )

        match on_conflict:
            case ConflictMode.INFORCE:
                return safe_cast_col(df, expected_field.name, expected_type)
            case ConflictMode.IGNORE:
                return df
            case ConflictMode.ERROR:
                raise SchemaEvolutionError(
                    f"Column '{expected_field.name}' type mismatch: "
                    f"{actual_type.simpleString()} != {expected_type.simpleString()}"
                )
            case ConflictMode.DROP:
                return safe_drop(df, expected_field.name)
            case ConflictMode.STRING:
                return safe_cast_col(df, expected_field.name, StringType())
            case _:
                raise ValueError(f"Invalid on_conflict mode: {on_conflict}")

    def _handle_extra_columns(
        self,
        df: DataFrame,
        expected_schema: StructType,
        extra_columns_conflict: Literal[ConflictMode.IGNORE, ConflictMode.DROP],
    ) -> DataFrame:
        """Handle columns in DataFrame that are not in expected schema."""
        expected_names: set[str] = set[str](expected_schema.fieldNames())

        for f in df.schema.fields:
            if f.name not in expected_names and extra_columns_conflict == ConflictMode.DROP:
                df = safe_drop(df, f.name)

        return df

    

    def _find_type_conflicts(
        self,
        actual_schema: StructType,
        target_schema: StructType,
    ) -> list[str]:
        """Find columns with type mismatches between schemas."""
        actual_fields: dict[str, StructField] = {f.name: f for f in actual_schema.fields}
        target_fields: dict[str, StructField] = {f.name: f for f in target_schema.fields}

        return [
            name
            for name in actual_fields.keys() & target_fields.keys()
            if actual_fields[name].dataType != target_fields[name].dataType
        ]

    def _build_select_expressions(
        self,
        df: DataFrame,
        target_schema: StructType,
        add_default_values: bool,
    ) -> list[Column]:
        """Build column expressions for reordering."""
        df_columns: set[str] = set[str](df.columns)
        expressions: list[Column] = []

        for f in target_schema.fields:
            if f.name in df_columns:
                expressions.append(F.col(f.name))
            elif add_default_values:
                expressions.append(F.lit(None).cast(f.dataType).alias(f.name))

        return expressions

    def _get_extra_columns(self, df: DataFrame, target_schema: StructType) -> list[str]:
        """Get columns in DataFrame that are not in target schema."""
        target_names: set[str] = {f.name for f in target_schema.fields}
        return [c for c in df.columns if c not in target_names]

    def _evolve_full_with_tracking(
        self,
        df: DataFrame,
        target_schema: StructType,
        extra_allowed: bool,
        default_values: Optional[dict[str, Any]],
        result: SchemaValidationResult,
    ) -> tuple[DataFrame, SchemaValidationResult]:
        """Full mode with change tracking."""
        df, result = self._evolve_backwards_with_tracking(
            df, target_schema, extra_allowed, default_values, result
        )
        df, result = self._evolve_forwards_with_tracking(
            df, target_schema, extra_allowed, result
        )
        return df, result

    def _evolve_backwards_with_tracking(
        self,
        df: DataFrame,
        target_schema: StructType,
        extra_allowed: bool,
        default_values: Optional[dict[str, Any]],
        result: SchemaValidationResult,
    ) -> tuple[DataFrame, SchemaValidationResult]:
        """Backwards mode with change tracking."""
        final_default_values: dict[str, Any] = default_values or {}
        existing: dict[str, StructField] = {f.name: f for f in df.schema.fields}
        target: dict[str, StructField] = {f.name: f for f in target_schema.fields}

        # Add missing columns
        for name, f in target.items():
            if name not in existing:
                df = _add_column_with_default(df, name, f.dataType, final_default_values)
                result.columns_added.append(name)

        # Handle extra columns
        if not extra_allowed:
            extra_cols: set[str] = set[str](existing) - set[str](target)
            for name in extra_cols:
                df = safe_drop(df, name)
                result.columns_dropped.append(name)

        return df, result

    def _evolve_forwards_with_tracking(
        self,
        df: DataFrame,
        target_schema: StructType,
        extra_allowed: bool,
        result: SchemaValidationResult,
    ) -> tuple[DataFrame, SchemaValidationResult]:
        """Forwards mode with change tracking."""
        existing: dict[str, StructField] = {f.name: f for f in df.schema.fields}
        target: dict[str, StructField] = {f.name: f for f in target_schema.fields}

        for name, f in target.items():
            if name not in existing:
                raise SchemaEvolutionError(
                    f"Column {name!r} required by target schema but missing "
                    f"in DataFrame for FORWARDS compatibility"
                )

            actual_type: DataType = existing[name].dataType
            expected_type: DataType = f.dataType

            if actual_type != expected_type:
                is_safe = is_safe_promotion(actual_type, expected_type)

                if not is_safe:
                    warning = (
                        f"Unsafe cast '{name}': "
                        f"{actual_type.simpleString()} -> {expected_type.simpleString()}"
                    )
                    result.warnings.append(warning)
                    safe_log(self._logger, 'warning', warning)

                df = safe_cast_col(df, name, expected_type)
                result.columns_cast[name] = (
                    actual_type.simpleString(),
                    expected_type.simpleString(),
                )

        if not extra_allowed:
            extra_cols: set[str] = set[str](existing) - set[str](target)
            for name in extra_cols:
                df = safe_drop(df, name)
                result.columns_dropped.append(name)

        return df, result
    
    def try_evolve(
        self,
        df: DataFrame,
        target_schema: StructType,
        compatibility_mode: CompatibilityMode = CompatibilityMode.BACKWARDS,
        extra_allowed: bool = False,
    ) -> Result[DataFrame, SchemaEvolutionError]:
        """
        Evolve DataFrame, returning Result instead of raising.
        
        Example:
            result = evolution.try_evolve(df, schema)
            if is_ok(result):
                evolved_df = unwrap(result)
            else:
                handle_error(unwrap_err(result))
        """
        try:
            evolved: DataFrame = self.evolve_df_to_target_schema(
                df, target_schema, compatibility_mode, extra_allowed
            )
            return Ok[DataFrame](evolved)
        except SchemaEvolutionError as e:
            return Err[SchemaEvolutionError](e)
    
    def try_evolve_with_report(
        self,
        df: DataFrame,
        target_schema: StructType,
        compatibility_mode: CompatibilityMode = CompatibilityMode.BACKWARDS,
    ) -> dict[str, Any]:
        """Result-based version of evolve_df_to_target_schema_with_report.

        Returns:
            dict[str, Any]: Dictionary containing the evolution result and the original DataFrame if the evolution failed
                "success": True if evolution was successful, False otherwise
                "result": Result of the evolution
                "original_df": Original DataFrame if the evolution failed
                "error": Error if the evolution failed
        """
        try:
            result: SchemaValidationResult = self.evolve_df_to_target_schema_with_report(
                df, target_schema,  compatibility_mode
            )
            return {
                "success": True,
                "result": result,
                "original_df": df,
                "error": None,
            }
        except SchemaEvolutionError as e:
            return {
                "success": False,
                "original_df": df,
                "error": e,
                "result": None,
            }
    
    def try_validate(
        self,
        df: DataFrame,
        expected_schema: StructType,
    ) -> dict[str, Any]:
        """Result-based schema validation.

        Returns:
            dict[str, Any]: Dictionary containing the validation result and the original DataFrame if the validation failed
                "success": True if validation was successful, False otherwise
                "result": Result of the validation
                "original_df": Original DataFrame if the validation failed
                "error": Error if the validation failed
        """
        try:
            validated: DataFrame = self.validate_schema(df, expected_schema)
            return {
                "success": True,
                "result": validated,
                "original_df": df,
                "error": None,
            }
        except SchemaEvolutionError as e:
            return {
                "success": False,
                "original_df": df,
                "error": e,
                "result": None,
            }
           
    def detect_drift(
        self,
        df: DataFrame,
        table_name: str,
        layer: Optional[LayerPolicy] = None,
        baseline_schema: Optional[StructType] = None,
        on_drift_callback: Optional[Callable[[DriftDetectionResult], None]] = None,
    ) -> DriftDetectionResult:
        """
        Detect schema drift between incoming DataFrame and baseline schema.
        
        Schema drift is detected by comparing the incoming DataFrame's schema 
        against either:
        1. An explicit baseline_schema parameter, OR
        2. The latest registered schema in the registry (if configured)
        
        Each detected difference generates a SchemaDriftAlert with appropriate
        severity based on the impact of the change.
        
        Args:
            df: Incoming DataFrame to check for drift
            table_name: Table identifier (used for registry lookup and alerts)
            layer: Medallion layer (affects severity classification)
            baseline_schema: Explicit baseline schema (overrides registry lookup)
            on_drift_callback: Optional callback function called when drift is detected.
                              Useful for sending alerts to external systems.
        
        Returns:
            DriftDetectionResult containing all detected drift alerts
            
        Example:
            >>> # Basic drift detection
            >>> result = evolution.detect_drift(df, "silver.orders")
            >>> if result.has_critical:
            ...     raise SchemaEvolutionError(f"Critical drift detected: {result.summary()}")
            
            >>> # With callback for alerting
            >>> def send_slack_alert(result: DriftDetectionResult):
            ...     if result.has_critical:
            ...         slack.post(f"🚨 Schema drift: {result.summary()}")
            >>> 
            >>> result = evolution.detect_drift(
            ...     df, "silver.orders", 
            ...     on_drift_callback=send_slack_alert
            ... )
            
        How Severity is Determined:
            - CRITICAL: Column removal, nullable->required, incompatible types
            - WARNING: Type changes (even safe ones), new required columns
            - INFO: New nullable columns added
        """
        final_layer: LayerPolicy = layer or self._default_layer
        
        # Determine baseline schema
        resolved_baseline: Optional[StructType] = baseline_schema
        if resolved_baseline is None and self._registry:
            latest = self._registry.get_latest_schema(table_name)
            if latest:
                resolved_baseline = latest.schema
        
        # If no baseline, no drift can be detected
        if resolved_baseline is None:
            safe_log(
                self._logger, 'info',
                f"[{self}] No baseline schema found for '{table_name}' - "
                "registering current schema as baseline"
            )
            return DriftDetectionResult(
                table_name=table_name,
                baseline_schema=None,
                actual_schema=df.schema,
            )
        
        # Detect drift
        result: DriftDetectionResult = self._analyze_drift(
            actual_schema=df.schema,
            baseline_schema=resolved_baseline,
            table_name=table_name,
            layer=final_layer,
        )
        
        # Log results
        if result.has_drift:
            log_level: Literal['error', 'warning', 'info'] = "error" if result.has_critical else "warning" if result.warnings else "info"
            safe_log(
                self._logger, log_level,
                f"[{self}] Schema drift detected for '{table_name}':\n{result.summary()}"
            )
        else:
            safe_log(self._logger, 'info', f"[{self}] No schema drift detected for '{table_name}'")
        
        # Execute callback if provided
        if on_drift_callback and result.has_drift:
            on_drift_callback(result)
        
        return result
    
    def _analyze_drift(
        self,
        actual_schema: StructType,
        baseline_schema: StructType,
        table_name: str,
        layer: LayerPolicy,
    ) -> DriftDetectionResult:
        """
        Perform detailed drift analysis between two schemas.
        
        Internal method that generates alerts for each type of drift.
        """
        alerts: list[SchemaDriftAlert] = []
        
        actual_fields: dict[str, StructField] = {f.name: f for f in actual_schema.fields}
        baseline_fields: dict[str, StructField] = {f.name: f for f in baseline_schema.fields}
        
        # 1. Detect REMOVED columns (CRITICAL)
        for name in baseline_fields:
            if name not in actual_fields:
                alerts.append(SchemaDriftAlert(
                    severity=DriftSeverity.CRITICAL,
                    drift_type=DriftType.COLUMN_REMOVED,
                    column_name=name,
                    details=f"Column '{name}' was removed from source data",
                    table_name=table_name,
                    layer=layer,
                    old_value=baseline_fields[name].dataType.simpleString(),
                    new_value=None,
                    recommended_action=(
                        "Investigate upstream data source. If intentional, update schema "
                        "definition. If not, restore the column in source."
                    ),
                ))
        
        # 2. Detect ADDED columns
        for name in actual_fields:
            if name not in baseline_fields:
                field = actual_fields[name]
                severity = DriftSeverity.INFO if field.nullable else DriftSeverity.WARNING
                
                alerts.append(SchemaDriftAlert(
                    severity=severity,
                    drift_type=DriftType.COLUMN_ADDED,
                    column_name=name,
                    details=(
                        f"New column '{name}' ({field.dataType.simpleString()}) "
                        f"appeared in source data"
                    ),
                    table_name=table_name,
                    layer=layer,
                    old_value=None,
                    new_value=field.dataType.simpleString(),
                    recommended_action=(
                        "Add column to target schema definition if needed, "
                        "or configure extra_columns_allowed=True"
                    ),
                ))
        
        # 3. Detect TYPE changes
        for name in actual_fields.keys() & baseline_fields.keys():
            actual_type = actual_fields[name].dataType
            baseline_type = baseline_fields[name].dataType
            
            if actual_type != baseline_type:
                # Determine severity based on type change safety
                is_safe = is_safe_promotion(actual_type, baseline_type)
                is_reverse_safe = is_safe_promotion(baseline_type, actual_type)
                
                if not is_safe and not is_reverse_safe:
                    severity = DriftSeverity.CRITICAL
                elif not is_safe:
                    severity = DriftSeverity.WARNING
                else:
                    severity = DriftSeverity.INFO
                
                alerts.append(SchemaDriftAlert(
                    severity=severity,
                    drift_type=DriftType.TYPE_CHANGED,
                    column_name=name,
                    details=(
                        f"Column '{name}' type changed: "
                        f"{baseline_type.simpleString()} → {actual_type.simpleString()}"
                    ),
                    table_name=table_name,
                    layer=layer,
                    old_value=baseline_type.simpleString(),
                    new_value=actual_type.simpleString(),
                    recommended_action=(
                        "Verify type change is intentional. Update schema definition "
                        "or add type casting logic."
                    ),
                ))
        
        # 4. Detect NULLABLE changes
        for name in actual_fields.keys() & baseline_fields.keys():
            actual_nullable = actual_fields[name].nullable
            baseline_nullable = baseline_fields[name].nullable
            
            if actual_nullable != baseline_nullable:
                # nullable -> non-nullable is CRITICAL (data may have nulls)
                # non-nullable -> nullable is INFO (relaxing constraint)
                if baseline_nullable and not actual_nullable:
                    severity = DriftSeverity.CRITICAL
                    details = f"Column '{name}' changed from nullable to required"
                else:
                    severity = DriftSeverity.INFO
                    details = f"Column '{name}' changed from required to nullable"
                
                alerts.append(SchemaDriftAlert(
                    severity=severity,
                    drift_type=DriftType.NULLABLE_CHANGED,
                    column_name=name,
                    details=details,
                    table_name=table_name,
                    layer=layer,
                    old_value=str(baseline_nullable),
                    new_value=str(actual_nullable),
                    recommended_action=(
                            "Update schema definition or handle null values appropriately"
                    ),
                ))
        
        return DriftDetectionResult(
            alerts=alerts,
            table_name=table_name,
            baseline_schema=baseline_schema,
            actual_schema=actual_schema,
        )
    
    def evolve_with_drift_detection(
        self,
        df: DataFrame,
        target_schema: StructType,
        table_name: str,
        layer: Optional[LayerPolicy] = None,
        fail_on_critical_drift: bool = True,
        on_drift_callback: Optional[Callable[[DriftDetectionResult], None]] = None,
    ) -> tuple[SchemaValidationResult, DriftDetectionResult]:
        """
        Detect drift first, then evolve schema if safe.
        
        Combines drift detection with schema evolution for a complete workflow.
        Use this when you want to be notified of upstream changes before evolving.
        
        Args:
            df: Source DataFrame
            target_schema: Target schema to evolve towards
            table_name: Table identifier
            layer: Medallion layer
            fail_on_critical_drift: If True, raise error on CRITICAL drift
            on_drift_callback: Callback for drift alerts
            
        Returns:
            Tuple of (evolution_result, drift_result)
            
        Raises:
            SchemaEvolutionError: If fail_on_critical_drift=True and CRITICAL drift found
            
        Example:
            >>> evolution_result, drift_result = evolution.evolve_with_drift_detection(
            ...     df=raw_df,
            ...     target_schema=silver_schema,
            ...     table_name="silver.orders",
            ...     layer=LayerPolicy.SILVER,
            ...     fail_on_critical_drift=True,
            ... )
            >>> if drift_result.warnings:
            ...     logger.warning(f"Non-critical drift detected: {drift_result.warnings}")
        """
        final_layer = layer or self._default_layer
        
        # Step 1: Detect drift against target schema
        drift_result = self.detect_drift(
            df=df,
            table_name=table_name,
            layer=final_layer,
            baseline_schema=target_schema,
            on_drift_callback=on_drift_callback,
        )
        
        # Step 2: Check for critical drift
        if fail_on_critical_drift and drift_result.has_critical:
            raise SchemaEvolutionError(
                f"Critical schema drift detected for '{table_name}'. "
                f"Cannot proceed with evolution.\n{drift_result.summary()}"
            )
        
        # Step 3: Evolve schema
        evolution_result = self.evolve_for_layer(
            df=df,
            target_schema=target_schema,
            layer=final_layer,
            table_name=table_name,
        )
        
        return evolution_result, drift_result
    
    def detect_breaking_changes(
        self,
        old_schema: StructType,
        new_schema: StructType,
    ) -> BreakingChangeReport:
        """
        Analyze two schemas and identify all breaking changes.
        
        Breaking changes are modifications that would cause failures or data loss
        for downstream consumers (dashboards, ML models, APIs, other tables).
        
        Use this BEFORE applying schema changes to understand the impact.
        
        Args:
            old_schema: Current/existing schema
            new_schema: Proposed new schema
            
        Returns:
            BreakingChangeReport with all identified breaking changes
            
        Example:
            >>> # Before deploying schema changes
            >>> report = evolution.detect_breaking_changes(
            ...     old_schema=current_table_schema,
            ...     new_schema=proposed_schema
            ... )
            >>> 
            >>> if not report.is_safe:
            ...     print(report.summary())
            ...     print("\\nAffected downstream systems:")
            ...     for change in report.breaking_changes:
            ...         print(f"  - {change.column_name}: {change.impact}")
            ...     
            ...     if not user_confirmed("Proceed anyway?"):
            ...         sys.exit(1)
            
        Breaking Change Types:
            - COLUMN_REMOVAL: Column exists in old but not new
            - TYPE_NARROWING: Type change that can lose data (DOUBLE -> INT)
            - NULLABLE_TO_REQUIRED: Column was nullable, now required
            - INCOMPATIBLE_TYPE: Types cannot be safely converted (STRING -> INT)
        """
        breaking_changes: list[BreakingChange] = []
        
        old_fields: dict[str, StructField] = {f.name: f for f in old_schema.fields}
        new_fields: dict[str, StructField] = {f.name: f for f in new_schema.fields}
        
        # 1. Check for column removals
        for name in old_fields:
            if name not in new_fields:
                breaking_changes.append(BreakingChange(
                    change_type=BreakingChangeType.COLUMN_REMOVAL,
                    column_name=name,
                    description=f"Column '{name}' will be removed",
                    old_type=old_fields[name].dataType.simpleString(),
                    new_type=None,
                    impact=(
                        f"Queries selecting '{name}' will fail. "
                        f"Downstream consumers expecting this column will break."
                    ),
                    mitigation=(
                        "1. Notify downstream consumers before removal\n"
                        "2. Consider soft-delete (mark as deprecated) first\n"
                        "3. Add column back as nullable with NULL values if needed"
                    ),
                ))
        
        # 2. Check for type changes
        for name in old_fields.keys() & new_fields.keys():
            old_type = old_fields[name].dataType
            new_type = new_fields[name].dataType
            
            if old_type != new_type:
                breaking_change = self._check_type_breaking_change(
                    name, old_type, new_type
                )
                if breaking_change:
                    breaking_changes.append(breaking_change)
        
        # 3. Check for nullable -> required changes
        for name in old_fields.keys() & new_fields.keys():
            old_nullable = old_fields[name].nullable
            new_nullable = new_fields[name].nullable
            
            if old_nullable and not new_nullable:
                breaking_changes.append(BreakingChange(
                    change_type=BreakingChangeType.NULLABLE_TO_REQUIRED,
                    column_name=name,
                    description=f"Column '{name}' changed from nullable to required",
                    old_type=f"nullable={old_nullable}",
                    new_type=f"nullable={new_nullable}",
                    impact=(
                        f"Existing NULL values in '{name}' will cause constraint violations. "
                        f"INSERTs without this column will fail."
                    ),
                    mitigation=(
                        "1. Backfill NULL values before changing constraint\n"
                        "2. Add DEFAULT value for the column\n"
                        "3. Keep column nullable and add application-level validation"
                    ),
                ))
        
        return BreakingChangeReport(
            breaking_changes=breaking_changes,
            old_schema=old_schema,
            new_schema=new_schema,
        )
    
    def _check_type_breaking_change(
        self,
        column_name: str,
        old_type: DataType,
        new_type: DataType,
    ) -> Optional[BreakingChange]:
        """
        Check if a type change is breaking.
        
        Returns BreakingChange if breaking, None if safe.
        """
        # Check for type narrowing (data loss)
        if self._is_type_narrowing(old_type, new_type):
            return BreakingChange(
                change_type=BreakingChangeType.TYPE_NARROWING,
                column_name=column_name,
                description=(
                    f"Type narrowing: {old_type.simpleString()} → {new_type.simpleString()} "
                    f"(potential data loss)"
                ),
                old_type=old_type.simpleString(),
                new_type=new_type.simpleString(),
                impact=(
                    "Values may be truncated or overflow. "
                    "For example, DOUBLE 3.14159 → INT becomes 3."
                ),
                mitigation=(
                    "1. Validate no data exceeds new type bounds\n"
                    "2. Consider keeping wider type\n"
                    "3. Add explicit rounding/truncation logic"
                ),
            )
        
        # Check for incompatible types
        if not is_safe_promotion(old_type, new_type) and not is_safe_promotion(new_type, old_type):
            return BreakingChange(
                change_type=BreakingChangeType.INCOMPATIBLE_TYPE,
                column_name=column_name,
                description=(
                    f"Incompatible type change: {old_type.simpleString()} → {new_type.simpleString()}"
                ),
                old_type=old_type.simpleString(),
                new_type=new_type.simpleString(),
                impact=(
                    "Data cannot be safely converted. "
                    "Downstream consumers may receive unexpected values or errors."
                ),
                mitigation=(
                    "1. Create a new column with the new type\n"
                    "2. Migrate data with explicit conversion logic\n"
                    "3. Keep both columns during transition period"
                ),
            )
        
        return None
    
    def _is_type_narrowing(self, old_type: DataType, new_type: DataType) -> bool:
        """
        Check if converting from old_type to new_type would narrow the data.
        
        Type narrowing examples:
        - DOUBLE -> FLOAT -> INT (loses precision/range)
        - LONG -> INT (loses range)
        - STRING -> INT (may fail for non-numeric strings)
        """
        # String to anything is potentially narrowing
        if isinstance(old_type, StringType) and not isinstance(new_type, StringType):
            return True
        
        # Check numeric narrowing
        for target_type, source_types in _TYPE_NARROWING_MAP.items():
            if isinstance(new_type, target_type):
                if any(isinstance(old_type, src) for src in source_types):
                    return True
        
        return False
    
    def safe_evolve_with_breaking_check(
        self,
        df: DataFrame,
        target_schema: StructType,
        table_name: str,
        layer: Optional[LayerPolicy] = None,
        allow_breaking: bool = False,
    ) -> tuple[SchemaValidationResult, BreakingChangeReport]:
        """
        Evolve schema only if no breaking changes, or if explicitly allowed.
        
        This is the safest way to evolve schemas in production. It checks for
        breaking changes BEFORE applying any modifications.
        
        Args:
            df: Source DataFrame
            target_schema: Target schema
            table_name: Table identifier
            layer: Medallion layer
            allow_breaking: If True, proceed even with breaking changes
            
        Returns:
            Tuple of (evolution_result, breaking_change_report)
            
        Raises:
            SchemaEvolutionError: If breaking changes found and allow_breaking=False
            
        Example:
            >>> try:
            ...     result, report = evolution.safe_evolve_with_breaking_check(
            ...         df, new_schema, "silver.customers"
            ...     )
            ... except SchemaEvolutionError as e:
            ...     # Handle breaking change - maybe require approval
            ...     if get_approval_from_data_owner():
            ...         result, report = evolution.safe_evolve_with_breaking_check(
            ...             df, new_schema, "silver.customers",
            ...             allow_breaking=True
            ...         )
        """
        final_layer = layer or self._default_layer
        
        # Check for breaking changes
        breaking_report = self.detect_breaking_changes(df.schema, target_schema)
        
        if not breaking_report.is_safe:
            safe_log(
                self._logger, 'warning',
                f"[{self}] Breaking changes detected for '{table_name}':\n"
                f"{breaking_report.summary()}"
            )
            
            if not allow_breaking:
                raise SchemaEvolutionError(
                    f"Cannot evolve schema for '{table_name}' - breaking changes detected.\n"
                    f"{breaking_report.summary()}\n"
                    f"Set allow_breaking=True to proceed anyway."
                )
        
        # Proceed with evolution
        evolution_result = self.evolve_for_layer(
            df=df,
            target_schema=target_schema,
            layer=final_layer,
            table_name=table_name,
        )
        
        return evolution_result, breaking_report
    
    def create_schema_metadata(
        self,
        table_name: str,
        schema: StructType,
        layer: Optional[LayerPolicy] = None,
        auto_detect_pii: bool = True,
        description: str = "",
        data_owner: Optional[str] = None,
    ) -> SchemaMetadata:
        """
        Create SchemaMetadata from a schema with auto-detection of PII columns.
        
        This creates metadata entries for every column, automatically detecting
        potential PII columns based on common naming patterns.
        
        Args:
            table_name: Table identifier
            schema: PySpark schema
            layer: Medallion layer
            auto_detect_pii: Detect PII columns by name patterns
            description: Table description
            data_owner: Team/person responsible
            
        Returns:
            SchemaMetadata with auto-generated column metadata
            
        Example:
            >>> metadata = evolution.create_schema_metadata(
            ...     table_name="silver.customers",
            ...     schema=customer_schema,
            ...     layer=LayerPolicy.SILVER,
            ...     auto_detect_pii=True,
            ...     description="Cleaned customer data",
            ...     data_owner="customer-team"
            ... )
            >>> print(f"PII columns: {metadata.get_pii_columns()}")
            
        PII Detection Patterns:
            - email, phone, ssn, password, credit_card
            - address, birth, salary, income, name
        """
        metadata = SchemaMetadata.from_schema(
            table_name=table_name,
            schema=schema,
            layer=layer,
            auto_detect_pii=auto_detect_pii,
        )
        metadata.description = description
        metadata.data_owner = data_owner
        
        # Log PII detection results
        pii_cols = metadata.get_pii_columns()
        if pii_cols:
            safe_log(
                self._logger, 'info',
                f"[{self}] Auto-detected {len(pii_cols)} potential PII columns "
                f"in '{table_name}': {pii_cols}"
            )
        
        return metadata
    
    def update_column_metadata(
        self,
        schema_metadata: SchemaMetadata,
        column_name: str,
        description: Optional[str] = None,
        pii: Optional[bool] = None,
        sensitivity_level: Optional[str] = None,
        data_owner: Optional[str] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> SchemaMetadata:
        """
        Update metadata for a specific column.
        
        Only provided fields are updated; others remain unchanged.
        
        Args:
            schema_metadata: Existing schema metadata
            column_name: Column to update
            description: New description
            pii: PII flag
            sensitivity_level: Data classification
            data_owner: Column owner
            tags: Column tags
            **kwargs: Additional ColumnMetadata fields
            
        Returns:
            Updated SchemaMetadata
            
        Example:
            >>> metadata = evolution.update_column_metadata(
            ...     schema_metadata=metadata,
            ...     column_name="email",
            ...     description="Customer email address",
            ...     pii=True,
            ...     sensitivity_level="confidential",
            ...     tags=["contact", "gdpr"]
            ... )
        """
        existing = schema_metadata.get_column_metadata(column_name)
        
        if existing is None:
            # Create new metadata
            existing = ColumnMetadata(column_name=column_name)
        
        # Update fields
        if description is not None:
            existing.description = description
        if pii is not None:
            existing.pii = pii
        if sensitivity_level is not None:
            existing.sensitivity_level = sensitivity_level
        if data_owner is not None:
            existing.data_owner = data_owner
        if tags is not None:
            existing.tags = tags
        
        # Handle additional kwargs
        for key, value in kwargs.items():
            if hasattr(existing, key) and value is not None:
                setattr(existing, key, value)
        
        schema_metadata.add_column_metadata(existing)
        return schema_metadata
    
    def validate_metadata_coverage(
        self,
        schema_metadata: SchemaMetadata,
        require_description: bool = False,
        require_pii_flag: bool = True,
    ) -> dict[str, Any]:
        """
        Validate that all columns have required metadata.
        
        Use this to ensure data governance requirements are met before
        deploying a schema to production.
        
        Args:
            schema_metadata: Metadata to validate
            require_description: Require description for all columns
            require_pii_flag: Require explicit PII flag (not default)
            
        Returns:
            Validation result dict with 'valid', 'issues', and 'coverage_percent'
            
        Example:
            >>> result = evolution.validate_metadata_coverage(
            ...     metadata, require_description=True
            ... )
            >>> if not result['valid']:
            ...     print(f"Metadata issues: {result['issues']}")
        """
        issues: list[str] = []
        
        # Check for columns missing metadata
        missing_metadata = schema_metadata.validate_schema_columns()
        if missing_metadata:
            issues.append(f"Columns missing metadata: {missing_metadata}")
        
        # Check individual column requirements
        for name, col_meta in schema_metadata.columns.items():
            if require_description and not col_meta.description:
                issues.append(f"Column '{name}' missing description")
        
        # Calculate coverage
        total_columns = len(schema_metadata.schema.fields) if schema_metadata.schema else 0
        documented_columns = len([c for c in schema_metadata.columns.values() 
                                  if c.description])
        coverage = (documented_columns / total_columns * 100) if total_columns > 0 else 0
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'coverage_percent': round(coverage, 1),
            'total_columns': total_columns,
            'documented_columns': documented_columns,
            'pii_columns': schema_metadata.get_pii_columns(),
        }
        
    def __str__(self) -> str:
        return f"{self.__class__.__name__}"
    
    def __repr__(self) -> str:
        return self.__str__()