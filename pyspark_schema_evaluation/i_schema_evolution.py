from __future__ import annotations

from abc import ABC, abstractmethod
from pyspark.sql.dataframe import DataFrame
from typing import Optional, Literal, Any, Callable
from pyspark.sql.types import StructType
from pyspark_schema_evaluation.core._enums import CompatibilityMode, ConflictMode, LayerPolicy
from pyspark_schema_evaluation.models.models import (
    BreakingChangeReport,
    DriftDetectionResult,
    SchemaDiff,
    SchemaMetadata,
    SchemaValidationResult,
)

__all__: list[str] = [
    "ISchemaEvolution",
]

class ISchemaEvolution(ABC):
    """Interface for schema evolution operations."""

    @abstractmethod
    def merge_schemas(
        self,
        schema_1: StructType,
        schema_2: StructType,
        on_conflict: ConflictMode = ConflictMode.STRING,
    ) -> StructType:
        """Merge two schemas with conflict resolution."""
        pass

    @abstractmethod
    def evolve_df_to_target_schema(
        self,
        df: DataFrame,
        target_schema: StructType,
        compatibility_mode: CompatibilityMode = CompatibilityMode.BACKWARDS,
        extra_allowed: bool = False,
    ) -> DataFrame:
        """Evolve DataFrame to match target schema."""
        pass

    @abstractmethod
    def compare_schema(
        self, expected_schema: StructType, actual_schema: StructType
    ) -> bool:
        """Compare two schemas for equality."""
        pass

    @abstractmethod
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
        """Validate and transform DataFrame to match expected schema."""
        pass

    @abstractmethod
    def reorder_df_to_schema(
        self,
        df: DataFrame,
        target_schema: StructType,
        drop_extra: bool = True,
        add_default_values: bool = False,
    ) -> DataFrame:
        """Reorder DataFrame columns to match target schema order."""
        pass

    @abstractmethod
    def diff_schemas(
        self,
        actual_schema: StructType,
        expected_schema: StructType,
    ) -> SchemaDiff:
        """Get detailed diff between two schemas."""
        pass

    @abstractmethod
    def evolve_df_to_target_schema_with_report(
        self,
        df: DataFrame,
        target_schema: StructType,
        compatibility_mode: CompatibilityMode = CompatibilityMode.BACKWARDS,
        extra_allowed: bool = False,
        default_values: Optional[dict[str, Any]] = None,
    ) -> SchemaValidationResult:
        """Evolve DataFrame to match target schema and return detailed report."""
        pass
    
    @abstractmethod
    def evolve_for_layer(
        self,
        df: DataFrame,
        target_schema: StructType,
        layer: LayerPolicy,
    ) -> SchemaValidationResult:
        """Evolve DataFrame to match target schema for a specific layer."""
        pass
    
    @abstractmethod
    def check_compatibility(
        self,
        table_name: str,
        new_schema: StructType,
    ) -> tuple[bool, Any, str]:
        """Check compatibility between two schemas."""
        pass
    
    @abstractmethod
    def try_evolve_with_report(
        self,
        df: DataFrame,
        target_schema: StructType,
        compatibility_mode: CompatibilityMode = CompatibilityMode.BACKWARDS,
    ) -> dict[str, Any]:
        """Try to evolve DataFrame to target schema and return detailed report.

        Returns:
            dict[str, Any]: Dictionary containing the evolution result and the original DataFrame if the evolution failed
                "success": True if evolution was successful, False otherwise
                "result": Result of the evolution
                "original_df": Original DataFrame if the evolution failed
                "error": Error if the evolution failed
        """
        pass
    
    @abstractmethod
    def try_validate(
        self,
        df: DataFrame,
        expected_schema: StructType,
    ) -> dict[str, Any]:
        """Try to validate DataFrame against expected schema and return detailed report.

        Returns:
            dict[str, Any]: Dictionary containing the validation result and the original DataFrame if the validation failed
                "success": True if validation was successful, False otherwise
                "result": Result of the validation
                "original_df": Original DataFrame if the validation failed
                "error": Error if the validation failed
        """
        pass
    
    # =========================================================================
    # SCHEMA DRIFT DETECTION
    # =========================================================================
    
    @abstractmethod
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
        
        Args:
            df: Incoming DataFrame to check for drift
            table_name: Table identifier
            layer: Medallion layer (affects severity classification)
            baseline_schema: Explicit baseline schema (overrides registry lookup)
            on_drift_callback: Callback function when drift is detected
        
        Returns:
            DriftDetectionResult containing all detected drift alerts
        """
        pass
    
    @abstractmethod
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
        
        Args:
            df: Source DataFrame
            target_schema: Target schema
            table_name: Table identifier
            layer: Medallion layer
            fail_on_critical_drift: Raise error on CRITICAL drift
            on_drift_callback: Callback for drift alerts
            
        Returns:
            Tuple of (evolution_result, drift_result)
        """
        pass
    
    # =========================================================================
    # BREAKING CHANGE DETECTION
    # =========================================================================
    
    @abstractmethod
    def detect_breaking_changes(
        self,
        old_schema: StructType,
        new_schema: StructType,
    ) -> BreakingChangeReport:
        """
        Analyze two schemas and identify breaking changes.
        
        Args:
            old_schema: Current/existing schema
            new_schema: Proposed new schema
            
        Returns:
            BreakingChangeReport with all identified breaking changes
        """
        pass
    
    @abstractmethod
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
        
        Args:
            df: Source DataFrame
            target_schema: Target schema
            table_name: Table identifier
            layer: Medallion layer
            allow_breaking: Proceed even with breaking changes
            
        Returns:
            Tuple of (evolution_result, breaking_change_report)
        """
        pass
    
    # =========================================================================
    # SCHEMA METADATA TRACKING
    # =========================================================================
    
    @abstractmethod
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
        Create SchemaMetadata with auto-detection of PII columns.
        
        Args:
            table_name: Table identifier
            schema: PySpark schema
            layer: Medallion layer
            auto_detect_pii: Detect PII columns by name patterns
            description: Table description
            data_owner: Team/person responsible
            
        Returns:
            SchemaMetadata with auto-generated column metadata
        """
        pass
    
    @abstractmethod
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
        
        Args:
            schema_metadata: Existing schema metadata
            column_name: Column to update
            description: New description
            pii: PII flag
            sensitivity_level: Data classification
            data_owner: Column owner
            tags: Column tags
            
        Returns:
            Updated SchemaMetadata
        """
        pass
    
    @abstractmethod
    def validate_metadata_coverage(
        self,
        schema_metadata: SchemaMetadata,
        require_description: bool = False,
        require_pii_flag: bool = True,
    ) -> dict[str, Any]:
        """
        Validate that all columns have required metadata.
        
        Args:
            schema_metadata: Metadata to validate
            require_description: Require description for all columns
            require_pii_flag: Require explicit PII flag
            
        Returns:
            Validation result dict
        """
        pass