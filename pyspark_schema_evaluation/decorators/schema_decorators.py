from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Optional, Literal
from collections.abc import Callable as AbcCallable
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType

from pyspark_schema_evaluation.schema_evolution import SchemaEvolution
from pyspark_schema_evaluation.core._enums import (
    ConflictMode,
    CompatibilityMode,
    LayerPolicy,
)
from pyspark_schema_evaluation.decorators._utils import (
    _resolve_callable,
    _locate_df,
    _rebuild_args_kwargs,
)
from pyspark_schema_evaluation.core._logging import get_logger
from logging import Logger

logger: Logger = get_logger()


__all__ = [
    "validate_schema",
    "evolve_df_to_target_schema",
    "evolve_for_layer",
    "ConflictMode",
    "CompatibilityMode",
    "LayerPolicy",]


def validate_schema(
    *,
    expected_schema: AbcCallable[[Any], StructType] | StructType,
    columns_to_ignore: Optional[list[str]] = None,
    columns_to_add: Optional[StructType] = None,
    conflict_mode_for_columns_to_add: ConflictMode = ConflictMode.SECOND,
    extra_columns_conflict: Literal[ConflictMode.IGNORE, ConflictMode.DROP] = ConflictMode.DROP,
    on_conflict: ConflictMode = ConflictMode.STRING,
    arg_name_for_df: str = "df",
    validate_on: Literal["input", "output"] = "input",
    logger: Logger = logger,
) -> Callable[..., Callable[..., Any]]:
    """
    Validate the schema of the DataFrame using SchemaEvolutionV2 Service.
    
    Parameters:
        expected_schema: The expected schema or a callable that returns the expected schema.
        columns_to_ignore: The columns to ignore.
        columns_to_add: The columns to add.
        conflict_mode_for_columns_to_add: The conflict mode for the columns to add.
        extra_columns_conflict: The conflict mode for extra columns.
        on_conflict: The conflict mode for the schema.
        arg_name_for_df: The name of the argument that contains the DataFrame.
        validate_on: Whether to validate on "input" or "output".
        logger: Logger instance.
    """

    def decorator(func) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            schema_evolution: SchemaEvolution = SchemaEvolution()
            
            self_obj: Any | None = (
                args[0] if args and not isinstance(args[0], DataFrame) else None
            )
            resolved_schema: Any | object = _resolve_callable(
                expected_schema, self_obj, "validate_schema"
            )
            df, df_index, df_in_kwargs = _locate_df(args, kwargs, arg_name_for_df)
            assert isinstance(resolved_schema, StructType), (
                f"Resolved schema is not a StructType: {resolved_schema}"
            )
            if validate_on == "input":
                result_validation = schema_evolution.validate_schema_with_report(
                    df=df,
                    expected_schema=resolved_schema,
                    extra_columns_conflict=extra_columns_conflict,
                    on_conflict=on_conflict,
                    columns_to_ignore=columns_to_ignore,
                    columns_to_add=columns_to_add,
                    conflict_mode_for_columns_to_add=conflict_mode_for_columns_to_add,
                )
                result_validation.log()
                    
                df_valid: DataFrame = result_validation.df
                new_args, new_kwargs = _rebuild_args_kwargs(
                    df_valid=df_valid,
                    args=args,
                    kwargs=kwargs,
                    df_index=df_index,
                    df_in_kwargs=df_in_kwargs,
                    arg_name_for_df=arg_name_for_df,
                )
                return func(*new_args, **new_kwargs)
            
            elif validate_on == "output":
                result: Any = func(*args, **kwargs)
                assert isinstance(result, DataFrame), (
                    f"Result is not a DataFrame: {result}"
                )
                result_validation = schema_evolution.validate_schema_with_report(
                    df=result,
                    expected_schema=resolved_schema,
                    extra_columns_conflict=extra_columns_conflict,
                    on_conflict=on_conflict,
                    columns_to_ignore=columns_to_ignore,
                    columns_to_add=columns_to_add,
                    conflict_mode_for_columns_to_add=conflict_mode_for_columns_to_add,
                )
                result_validation.log()
                
                if result_validation.warnings:
                    logger.warning(f"Unsafe casts detected: {result_validation.warnings}")

                return result_validation.df
            
        return wrapper
    return decorator


def evolve_df_to_target_schema(
    *,
    target_schema: AbcCallable[[Any], StructType] | StructType,
    compatibility_mode: CompatibilityMode = CompatibilityMode.BACKWARDS,
    extra_allowed: bool = False,
    arg_name_for_df: str = "df",
    logger: Logger = logger,
) -> AbcCallable[[AbcCallable[..., Any]], AbcCallable[..., Any]]:
    """
    Evolve the DataFrame to the target schema using SchemaEvolutionV2 Service.
    
    Parameters:
        target_schema: The target schema or a callable that returns the target schema.
        compatibility_mode: The compatibility mode to use.
        extra_allowed: Whether to allow extra columns in the DataFrame.
        arg_name_for_df: The name of the argument that contains the DataFrame.
        logger: Logger instance.
    """

    def decorator(func: AbcCallable[..., Any]) -> AbcCallable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            schema_evolution: SchemaEvolution = SchemaEvolution()

            self_obj: Any | None = (
                args[0] if args and not isinstance(args[0], DataFrame) else None
            )
            resolved_target_schema: Any | object = _resolve_callable(
                target_schema, self_obj, "evolve_df_to_target_schema"
            )
            df, df_index, df_in_kwargs = _locate_df(args, kwargs, arg_name_for_df)

            assert isinstance(resolved_target_schema, StructType), (
                f"Resolved target schema is not a StructType: {resolved_target_schema}"
            )
            if df is None or not isinstance(df, DataFrame):
                raise ValueError(
                    "@evolve_df_to_target_schema: df is required and must be a pyspark.sql.DataFrame. "
                    f"Expected signature: func({arg_name_for_df}, ...), "
                    f"func(self, {arg_name_for_df}, ...), or {arg_name_for_df} passed as keyword."
                )

            result_evolution = schema_evolution.evolve_df_to_target_schema_with_report(
                df=df,
                target_schema=resolved_target_schema,
                compatibility_mode=compatibility_mode,
                extra_allowed=extra_allowed,
            )
            logger.info(
                f"Schema evolution to target schema successful for {func.__name__}:\n"
                f"    {result_evolution.summary()}"
            )  
            
            if result_evolution.warnings:
                logger.warning(f"Unsafe casts detected: {result_evolution.warnings}")

            df_valid: DataFrame = result_evolution.df
            new_args, new_kwargs = _rebuild_args_kwargs(
                df_valid=df_valid,
                args=args,
                kwargs=kwargs,
                df_index=df_index,
                df_in_kwargs=df_in_kwargs,
                arg_name_for_df=arg_name_for_df,
            )
            return func(*new_args, **new_kwargs)

        return wrapper

    return decorator


def evolve_for_layer(
    *,
    target_schema: AbcCallable[[Any], StructType] | StructType,
    layer: LayerPolicy = LayerPolicy.SILVER,
    table_name: Optional[str] = None,
    default_values: Optional[dict[str, Any]] = None,
    strict_compatibility: bool = False,
    custom_policy: Optional[str] = None,
    arg_name_for_df: str = "df",
    logger: Logger = logger,
) -> AbcCallable[[AbcCallable[..., Any]], AbcCallable[..., Any]]:
    """
    Evolve DataFrame using layer-specific policies with SchemaEvolutionV2.
    
    This decorator applies layer-aware schema evolution with compatibility checking
    and optional registry integration.
    
    Parameters:
        target_schema: The target schema or a callable that returns the target schema.
        layer: Medallion layer (BRONZE, SILVER, GOLD, CUSTOM). Determines evolution policy:
            - BRONZE: Permissive, allows extra columns, logs info on changes
            - SILVER: Backwards compatible, warns on breaking changes
            - GOLD: Strict, raises error on any incompatibility
            - CUSTOM: Use a custom policy registered with LayerPolicyRegistry
        table_name: Optional registry table name for schema versioning and compatibility checks.
        default_values: Custom default values for new columns (e.g., {"new_col": 0}).
        strict_compatibility: If True, raise error on any incompatibility regardless of layer.
        custom_policy: Name of custom policy (required if layer=CUSTOM).
                      Register custom policies with LayerPolicyRegistry.register()
        arg_name_for_df: The name of the argument that contains the DataFrame.
        logger: Logger instance.
        
    Example - Built-in policy:
        @evolve_for_layer(
            target_schema=lambda self: self.target_schema,
            layer=LayerPolicy.SILVER,
            table_name="silver.raw_cur",
        )
        def transform(self, df: DataFrame) -> DataFrame:
            ...
            
    Example - Custom policy:
        # First register the policy (e.g., in your app startup)
        from sp_db_pipeline.services.schema._enums import LayerPolicyRegistry
        from sp_db_pipeline.services.schema._models import LayerPolicyConfig
        
        LayerPolicyRegistry.register("STAGING", LayerPolicyConfig(
            compatibility_mode=CompatibilityMode.BACKWARDS,
            extra_columns_allowed=True,
            on_type_conflict=ConflictMode.INFORCE,
            allow_nullable_changes=True,
            require_all_columns=False,
        ))
        
        # Then use it
        @evolve_for_layer(
            target_schema=MY_SCHEMA,
            layer=LayerPolicy.CUSTOM,
            custom_policy="STAGING",
        )
        def process(df: DataFrame) -> DataFrame:
            ...
    """

    def decorator(func: AbcCallable[..., Any]) -> AbcCallable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            schema_evolution: SchemaEvolution = SchemaEvolution()

            self_obj: Any | None = (
                args[0] if args and not isinstance(args[0], DataFrame) else None
            )
            resolved_target_schema: Any | object = _resolve_callable(
                target_schema, self_obj, "evolve_for_layer"
            )
            df, df_index, df_in_kwargs = _locate_df(args, kwargs, arg_name_for_df)

            assert isinstance(resolved_target_schema, StructType), (
                f"Resolved target schema is not a StructType: {resolved_target_schema}"
            )
            if df is None or not isinstance(df, DataFrame):
                raise ValueError(
                    "@evolve_for_layer: df is required and must be a pyspark.sql.DataFrame. "
                    f"Expected signature: func({arg_name_for_df}, ...), "
                    f"func(self, {arg_name_for_df}, ...), or {arg_name_for_df} passed as keyword."
                )

            result_evolution = schema_evolution.evolve_for_layer(
                df=df,
                target_schema=resolved_target_schema,
                layer=layer,
                default_values=default_values,
                table_name=table_name,
                strict_compatibility=strict_compatibility,
                custom_policy=custom_policy,
            )
            
            policy_name = custom_policy if custom_policy else layer.value
            logger.info(
                f"Schema evolution for layer={policy_name} successful for {func.__name__}:\n"
                f"    {result_evolution.summary()}"
            )  
            
            if result_evolution.warnings:
                logger.warning(f"Unsafe casts detected: {result_evolution.warnings}")

            df_valid: DataFrame = result_evolution.df
            new_args, new_kwargs = _rebuild_args_kwargs(
                df_valid=df_valid,
                args=args,
                kwargs=kwargs,
                df_index=df_index,
                df_in_kwargs=df_in_kwargs,
                arg_name_for_df=arg_name_for_df,
            )
            return func(*new_args, **new_kwargs)

        return wrapper

    return decorator
