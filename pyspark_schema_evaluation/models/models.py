from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any
from pyspark.sql.types import StructType
from pyspark_schema_evaluation.core._enums import (
    LayerPolicy,
    CompatibilityMode,
    ConflictMode,
    DriftSeverity,
    DriftType,
    BreakingChangeType,
)
from pyspark_schema_evaluation.registry.policy_registry import LayerPolicyRegistry
from pyspark.sql.dataframe import DataFrame
import logging
from pyspark_schema_evaluation.core._logging import get_logger, safe_log


__all__: list[str] = [
    "VersionedSchema", 
    "SchemaValidationResult",
    "SchemaDiff",
    "LayerPolicyConfig",
    "SchemaDriftAlert",
    "BreakingChange",
    "BreakingChangeReport",
    "ColumnMetadata",
    "SchemaMetadata",
    "DriftDetectionResult",
]


@dataclass
class VersionedSchema:
    """
    A schema with version tracking for evolution history.
    
    Attributes:
        schema: The PySpark StructType schema
        version: Monotonically increasing version number
        layer: Which medallion layer this schema belongs to
        created_at: When this schema version was created
        description: Optional description of changes in this version
        previous_version: Link to previous version for lineage
    """
    schema: StructType
    version: int
    layer: Optional[LayerPolicy]
    created_at: datetime = field(default_factory=datetime.utcnow)
    description: Optional[str] = None
    previous_version: Optional[int] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "schema_json": self.schema.json(),
            "version": self.version,
            "layer": self.layer.value if self.layer else None,
            "created_at": self.created_at.isoformat(),
            "description": self.description,
            "previous_version": self.previous_version,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VersionedSchema:
        """Deserialize from dictionary."""
        return cls(
            schema=StructType.fromJson(data["schema_json"]),
            version=data["version"],
            layer=LayerPolicy(data["layer"]) if data["layer"] else None,
            created_at=datetime.fromisoformat(data["created_at"]),
            description=data.get("description"),
            previous_version=data.get("previous_version"),
        )





@dataclass
class LayerPolicyConfig:
    """
    Configuration for a layer policy.
    
    Attributes:
        compatibility_mode: How strict schema compatibility should be
        extra_columns_allowed: Whether to allow columns not in target schema
        on_type_conflict: How to handle type mismatches
        allow_nullable_changes: Whether nullable -> non-nullable is allowed
        require_all_columns: Whether all target columns must exist in source
        
    Example - Creating a custom policy:
        from sp_db_pipeline.services.schema._models import LayerPolicyConfig
        from sp_db_pipeline.services.schema._enums import (
            LayerPolicyRegistry, CompatibilityMode, ConflictMode
        )
        
        # Define a custom "STAGING" policy
        staging_config = LayerPolicyConfig(
            compatibility_mode=CompatibilityMode.BACKWARDS,
            extra_columns_allowed=True,
            on_type_conflict=ConflictMode.INFORCE,
            allow_nullable_changes=True,
            require_all_columns=False,
        )
        
        # Register it
        LayerPolicyRegistry.register("STAGING", staging_config)
    """
    compatibility_mode: CompatibilityMode
    extra_columns_allowed: bool
    on_type_conflict: ConflictMode
    allow_nullable_changes: bool
    require_all_columns: bool

    @classmethod
    def from_layer(
        cls, 
        layer: LayerPolicy, 
        custom_policy_name: Optional[str] = None,
    ) -> LayerPolicyConfig:
        """
        Get policy configuration for a layer.
        
        Args:
            layer: The LayerPolicy enum value
            custom_policy_name: Name of custom policy (required if layer=CUSTOM)
            
        Returns:
            LayerPolicyConfig for the specified layer
            
        Raises:
            ValueError: If layer is CUSTOM but no custom_policy_name provided,
                       or if custom policy not found in registry
        """
        match layer:
            case LayerPolicy.BRONZE:
                return cls(
                    compatibility_mode=CompatibilityMode.FULL,
                    extra_columns_allowed=True,
                    on_type_conflict=ConflictMode.STRING,
                    allow_nullable_changes=True,
                    require_all_columns=False,
                )
            case LayerPolicy.SILVER:
                return cls(
                    compatibility_mode=CompatibilityMode.BACKWARDS,
                    extra_columns_allowed=False,
                    on_type_conflict=ConflictMode.INFORCE,
                    allow_nullable_changes=True,
                    require_all_columns=False,
                )
            case LayerPolicy.GOLD:
                return cls(
                    compatibility_mode=CompatibilityMode.STRICT,
                    extra_columns_allowed=False,
                    on_type_conflict=ConflictMode.ERROR,
                    allow_nullable_changes=False,
                    require_all_columns=True,
                )
            case LayerPolicy.CUSTOM:
                if not custom_policy_name:
                    raise ValueError(
                        "custom_policy_name is required when using LayerPolicy.CUSTOM. "
                        "Register a policy with LayerPolicyRegistry.register() first."
                    )
                config = LayerPolicyRegistry.get(custom_policy_name)
                if config is None:
                    available = LayerPolicyRegistry.list_policies()
                    raise ValueError(
                        f"Custom policy '{custom_policy_name}' not found in registry. "
                        f"Available policies: {available or 'None registered'}"
                    )
                return config
            case _:
                raise ValueError(f"Unknown layer: {layer}")

def _log_schema_validation_result(
    result: SchemaValidationResult, 
    extra_info: Optional[dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Log schema validation result using centralized logging.
    
    Args:
        result: SchemaValidationResult to log
        extra_info: Optional additional info dict
        logger: Optional logger instance. If None, uses module-level logger.
    """
    if logger is None:
        logger = get_logger()
    
    extra_info_str: str = f"{extra_info}\n" if extra_info is not None else ""
    if result.had_changes:
        message = f"[SchemaValidationResult] Schema validation had changes:\n{extra_info_str}{result.summary()}\n"
    else:
        message = f"[SchemaValidationResult] Schema validation had no changes\n{extra_info_str}"
    
    safe_log(logger, 'info', message)



@dataclass
class SchemaValidationResult:
    """Result of schema validation with metrics."""
    df: DataFrame
    columns_added: list[str] = field(default_factory=list)
    columns_dropped: list[str] = field(default_factory=list)
    columns_cast: dict[str, tuple[str, str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    
    @property
    def had_changes(self) -> bool:
        return bool(self.columns_added or self.columns_dropped or self.columns_cast)

    def log(self,extra_info: Optional[dict[str, Any]] = None) -> None:
        _log_schema_validation_result(self, extra_info)
                    
    def summary(self) -> str:
        """Human-readable summary of changes."""
        lines: list[Any] = []
        if self.columns_added:
            lines.append(f"Added: {self.columns_added}")
        if self.columns_dropped:
            lines.append(f"Dropped: {self.columns_dropped}")
        if self.columns_cast:
            casts: list[str] = [f"{k}: {v[0]} -> {v[1]}" for k, v in self.columns_cast.items()]
            lines.append(f"Cast: {casts}")
        if self.warnings:
            lines.append(f"Warnings: {self.warnings}")
        return "\n".join(lines) if lines else "No changes"
    
    
@dataclass
class SchemaDiff:
    """Detailed schema comparison result"""
    missing_columns: list[str]
    extra_columns: list[str]
    type_mismatches: dict[str, tuple[str, str]]  
    nullable_changes: dict[str, tuple[bool, bool]]  
    is_compatible: bool


# =============================================================================
# SCHEMA DRIFT DETECTION MODELS
# =============================================================================

@dataclass
class SchemaDriftAlert:
    """
    Represents a single schema drift event with severity and context.
    
    Schema drift occurs when the incoming data schema differs from the expected
    schema. This can happen due to:
    - Upstream data source changes
    - ETL pipeline modifications
    - Data provider API updates
    
    Attributes:
        severity: Impact level (INFO, WARNING, CRITICAL)
        drift_type: Category of drift (column added, removed, type changed, etc.)
        column_name: Affected column (None for table-level alerts)
        details: Human-readable description of the drift
        table_name: Affected table/dataset name
        layer: Medallion layer where drift was detected
        timestamp: When the drift was detected
        old_value: Previous value (type, nullable, etc.) if applicable
        new_value: New value if applicable
        recommended_action: Suggested remediation step
        
    Example:
        >>> alert = SchemaDriftAlert(
        ...     severity=DriftSeverity.WARNING,
        ...     drift_type=DriftType.TYPE_CHANGED,
        ...     column_name="user_age",
        ...     details="Column type changed from STRING to INT",
        ...     table_name="silver.users",
        ...     layer=LayerPolicy.SILVER,
        ...     old_value="string",
        ...     new_value="int",
        ...     recommended_action="Verify upstream data source and update schema definition"
        ... )
    """
    severity: DriftSeverity
    drift_type: DriftType
    column_name: Optional[str]
    details: str
    table_name: str
    layer: Optional[LayerPolicy] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    recommended_action: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging/storage."""
        return {
            "severity": self.severity.value,
            "drift_type": self.drift_type.value,
            "column_name": self.column_name,
            "details": self.details,
            "table_name": self.table_name,
            "layer": self.layer.value if self.layer else None,
            "timestamp": self.timestamp.isoformat(),
            "old_value": self.old_value,
            "new_value": self.new_value,
            "recommended_action": self.recommended_action,
        }
    
    def __str__(self) -> str:
        col_info = f" [{self.column_name}]" if self.column_name else ""
        return f"[{self.severity.value.upper()}] {self.table_name}{col_info}: {self.details}"


@dataclass
class DriftDetectionResult:
    """
    Complete result of schema drift detection analysis.
    
    Contains all detected drift alerts grouped by severity, plus summary statistics
    for easy decision-making.
    
    Attributes:
        alerts: List of all detected drift alerts
        table_name: Table that was analyzed
        baseline_schema: The expected/registered schema
        actual_schema: The incoming/actual schema
        detection_timestamp: When analysis was performed
        
    Properties:
        has_drift: True if any drift was detected
        has_critical: True if any CRITICAL severity alerts exist
        has_breaking_changes: Alias for has_critical
        
    Example:
        >>> result = drift_detector.detect_drift(df, "silver.orders")
        >>> if result.has_critical:
        ...     raise SchemaEvolutionError(f"Critical drift: {result.critical_alerts}")
        >>> for alert in result.warnings:
        ...     logger.warning(str(alert))
    """
    alerts: list[SchemaDriftAlert] = field(default_factory=list)
    table_name: str = ""
    baseline_schema: Optional[StructType] = None
    actual_schema: Optional[StructType] = None
    detection_timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def has_drift(self) -> bool:
        """Returns True if any drift was detected."""
        return len(self.alerts) > 0
    
    @property
    def has_critical(self) -> bool:
        """Returns True if any critical-severity drift was detected."""
        return any(a.severity == DriftSeverity.CRITICAL for a in self.alerts)
    
    @property
    def has_breaking_changes(self) -> bool:
        """Alias for has_critical - indicates breaking changes exist."""
        return self.has_critical
    
    @property
    def critical_alerts(self) -> list[SchemaDriftAlert]:
        """Get only CRITICAL severity alerts."""
        return [a for a in self.alerts if a.severity == DriftSeverity.CRITICAL]
    
    @property
    def warnings(self) -> list[SchemaDriftAlert]:
        """Get only WARNING severity alerts."""
        return [a for a in self.alerts if a.severity == DriftSeverity.WARNING]
    
    @property
    def info_alerts(self) -> list[SchemaDriftAlert]:
        """Get only INFO severity alerts."""
        return [a for a in self.alerts if a.severity == DriftSeverity.INFO]
    
    def summary(self) -> str:
        """Human-readable summary of drift detection."""
        if not self.has_drift:
            return f"No schema drift detected for '{self.table_name}'"
        
        lines = [
            f"Schema Drift Report for '{self.table_name}':",
            f"  Total alerts: {len(self.alerts)}",
            f"  Critical: {len(self.critical_alerts)}",
            f"  Warnings: {len(self.warnings)}",
            f"  Info: {len(self.info_alerts)}",
        ]
        
        if self.critical_alerts:
            lines.append("\n  ❌ CRITICAL ISSUES:")
            for alert in self.critical_alerts:
                lines.append(f"    - {alert.details}")
        
        if self.warnings:
            lines.append("\n  ⚠️  WARNINGS:")
            for alert in self.warnings:
                lines.append(f"    - {alert.details}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage/API response."""
        return {
            "table_name": self.table_name,
            "has_drift": self.has_drift,
            "has_critical": self.has_critical,
            "alert_count": len(self.alerts),
            "critical_count": len(self.critical_alerts),
            "warning_count": len(self.warnings),
            "info_count": len(self.info_alerts),
            "alerts": [a.to_dict() for a in self.alerts],
            "detection_timestamp": self.detection_timestamp.isoformat(),
        }


# =============================================================================
# BREAKING CHANGE DETECTION MODELS
# =============================================================================

@dataclass
class BreakingChange:
    """
    Represents a single breaking change that would affect downstream consumers.
    
    Breaking changes are schema modifications that could cause:
    - Query failures (SELECT on removed columns)
    - Data loss (narrowing type conversions like DOUBLE -> INT)
    - Runtime errors (NULL in non-nullable column)
    - Application crashes (type mismatches)
    
    Attributes:
        change_type: Category of breaking change
        column_name: Affected column
        description: Detailed explanation
        old_type: Previous data type (if type change)
        new_type: New data type (if type change)
        impact: Expected impact on downstream systems
        mitigation: Suggested fix or workaround
        
    Example:
        >>> change = BreakingChange(
        ...     change_type=BreakingChangeType.COLUMN_REMOVAL,
        ...     column_name="legacy_id",
        ...     description="Column 'legacy_id' will be removed",
        ...     impact="Queries selecting this column will fail",
        ...     mitigation="Update downstream queries to use 'new_id' instead"
        ... )
    """
    change_type: BreakingChangeType
    column_name: str
    description: str
    old_type: Optional[str] = None
    new_type: Optional[str] = None
    impact: Optional[str] = None
    mitigation: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "change_type": self.change_type.value,
            "column_name": self.column_name,
            "description": self.description,
            "old_type": self.old_type,
            "new_type": self.new_type,
            "impact": self.impact,
            "mitigation": self.mitigation,
        }
    
    def __str__(self) -> str:
        return f"[{self.change_type.value}] {self.column_name}: {self.description}"


@dataclass
class BreakingChangeReport:
    """
    Complete report of breaking changes between two schemas.
    
    Use this before applying schema evolution to understand the impact
    on downstream consumers (dashboards, ML models, APIs, etc.)
    
    Attributes:
        breaking_changes: List of all detected breaking changes
        old_schema: Original/current schema
        new_schema: Proposed new schema
        is_safe: True if no breaking changes detected
        
    Example:
        >>> report = evolution.detect_breaking_changes(old_schema, new_schema)
        >>> if not report.is_safe:
        ...     print(f"Found {len(report.breaking_changes)} breaking changes!")
        ...     for change in report.breaking_changes:
        ...         print(f"  - {change}")
        ...     raise ValueError("Cannot proceed with breaking changes")
    """
    breaking_changes: list[BreakingChange] = field(default_factory=list)
    old_schema: Optional[StructType] = None
    new_schema: Optional[StructType] = None
    analyzed_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def is_safe(self) -> bool:
        """Returns True if no breaking changes were detected."""
        return len(self.breaking_changes) == 0
    
    @property
    def has_column_removals(self) -> bool:
        """Check if any columns will be removed."""
        return any(c.change_type == BreakingChangeType.COLUMN_REMOVAL 
                   for c in self.breaking_changes)
    
    @property
    def has_type_changes(self) -> bool:
        """Check if any incompatible type changes exist."""
        return any(c.change_type in (BreakingChangeType.TYPE_NARROWING, 
                                     BreakingChangeType.INCOMPATIBLE_TYPE) 
                   for c in self.breaking_changes)
    
    @property
    def has_nullable_violations(self) -> bool:
        """Check if any nullable -> required changes exist."""
        return any(c.change_type == BreakingChangeType.NULLABLE_TO_REQUIRED 
                   for c in self.breaking_changes)
    
    def summary(self) -> str:
        """Human-readable summary."""
        if self.is_safe:
            return "✅ No breaking changes detected - safe to proceed"
        
        lines = [
            f"⚠️  BREAKING CHANGES DETECTED ({len(self.breaking_changes)} total):",
            "",
        ]
        
        by_type: dict[BreakingChangeType, list[BreakingChange]] = {}
        for change in self.breaking_changes:
            by_type.setdefault(change.change_type, []).append(change)
        
        for change_type, changes in by_type.items():
            lines.append(f"  {change_type.value} ({len(changes)}):")
            for change in changes:
                lines.append(f"    - {change.column_name}: {change.description}")
                if change.mitigation:
                    lines.append(f"      💡 Mitigation: {change.mitigation}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "is_safe": self.is_safe,
            "breaking_change_count": len(self.breaking_changes),
            "has_column_removals": self.has_column_removals,
            "has_type_changes": self.has_type_changes,
            "has_nullable_violations": self.has_nullable_violations,
            "breaking_changes": [c.to_dict() for c in self.breaking_changes],
            "analyzed_at": self.analyzed_at.isoformat(),
        }


# =============================================================================
# SCHEMA METADATA TRACKING MODELS
# =============================================================================

@dataclass
class ColumnMetadata:
    """
    Business and technical metadata for a single column.
    
    Tracks information beyond the technical schema that's important for:
    - Data governance (PII, sensitivity)
    - Documentation (descriptions, examples)
    - Data quality (validation rules)
    - Lineage (source system, transformations)
    
    Attributes:
        column_name: Name of the column
        description: Human-readable description of the column
        data_type: PySpark data type (for reference)
        pii: Whether column contains Personally Identifiable Information
        sensitivity_level: Data classification (public, internal, confidential, restricted)
        data_owner: Team or person responsible for this data
        source_system: Where this data originates
        business_name: Business-friendly name (e.g., "Customer ID" vs "cust_id")
        valid_values: List of allowed values (for enums/categoricals)
        validation_regex: Regex pattern for validation
        min_value: Minimum allowed value (for numerics)
        max_value: Maximum allowed value (for numerics)
        default_value: Default value if not provided
        example_values: Example values for documentation
        tags: Custom tags for categorization
        created_at: When metadata was created
        updated_at: When metadata was last updated
        retention_days: How long to retain this data
        
    Example:
        >>> email_metadata = ColumnMetadata(
        ...     column_name="email",
        ...     description="Customer email address",
        ...     pii=True,
        ...     sensitivity_level="confidential",
        ...     data_owner="customer-data-team",
        ...     validation_regex=r"^[\\w.-]+@[\\w.-]+\\.\\w+$",
        ...     tags=["contact", "pii", "gdpr"]
        ... )
    """
    column_name: str
    description: str = ""
    data_type: Optional[str] = None
    pii: bool = False
    sensitivity_level: str = "internal"  # public, internal, confidential, restricted
    data_owner: Optional[str] = None
    source_system: Optional[str] = None
    business_name: Optional[str] = None
    valid_values: Optional[list[str]] = None
    validation_regex: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    default_value: Optional[Any] = None
    example_values: Optional[list[Any]] = None
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    retention_days: Optional[int] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "column_name": self.column_name,
            "description": self.description,
            "data_type": self.data_type,
            "pii": self.pii,
            "sensitivity_level": self.sensitivity_level,
            "data_owner": self.data_owner,
            "source_system": self.source_system,
            "business_name": self.business_name,
            "valid_values": self.valid_values,
            "validation_regex": self.validation_regex,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "default_value": self.default_value,
            "example_values": self.example_values,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "retention_days": self.retention_days,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ColumnMetadata:
        """Deserialize from dictionary."""
        return cls(
            column_name=data["column_name"],
            description=data.get("description", ""),
            data_type=data.get("data_type"),
            pii=data.get("pii", False),
            sensitivity_level=data.get("sensitivity_level", "internal"),
            data_owner=data.get("data_owner"),
            source_system=data.get("source_system"),
            business_name=data.get("business_name"),
            valid_values=data.get("valid_values"),
            validation_regex=data.get("validation_regex"),
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            default_value=data.get("default_value"),
            example_values=data.get("example_values"),
            tags=data.get("tags", []),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.utcnow(),
            retention_days=data.get("retention_days"),
        )


@dataclass
class SchemaMetadata:
    """
    Complete metadata for a table/dataset schema.
    
    Combines technical schema (StructType) with business metadata for each column,
    plus table-level properties.
    
    Attributes:
        table_name: Fully qualified table name (e.g., "silver.customers")
        schema: PySpark StructType
        columns: Metadata for each column
        description: Table description
        data_owner: Team responsible for this table
        layer: Medallion layer
        source_tables: Upstream dependencies
        refresh_schedule: How often data is updated
        sla_hours: SLA for data freshness
        tags: Custom tags
        created_at: When table was created
        updated_at: When metadata was last updated
        
    Example:
        >>> schema_meta = SchemaMetadata(
        ...     table_name="silver.customers",
        ...     schema=customer_schema,
        ...     description="Cleaned and deduplicated customer data",
        ...     data_owner="customer-data-team",
        ...     layer=LayerPolicy.SILVER,
        ...     source_tables=["bronze.raw_customers"],
        ...     refresh_schedule="hourly",
        ...     sla_hours=2,
        ... )
        >>> schema_meta.add_column_metadata(email_metadata)
    """
    table_name: str
    schema: Optional[StructType] = None
    columns: dict[str, ColumnMetadata] = field(default_factory=dict)
    description: str = ""
    data_owner: Optional[str] = None
    layer: Optional[LayerPolicy] = None
    source_tables: list[str] = field(default_factory=list)
    refresh_schedule: Optional[str] = None  # "hourly", "daily", "realtime", cron expr
    sla_hours: Optional[int] = None
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_column_metadata(self, metadata: ColumnMetadata) -> None:
        """Add or update metadata for a column."""
        self.columns[metadata.column_name] = metadata
        self.updated_at = datetime.utcnow()
    
    def get_column_metadata(self, column_name: str) -> Optional[ColumnMetadata]:
        """Get metadata for a specific column."""
        return self.columns.get(column_name)
    
    def get_pii_columns(self) -> list[str]:
        """Get list of columns marked as PII."""
        return [name for name, meta in self.columns.items() if meta.pii]
    
    def get_columns_by_tag(self, tag: str) -> list[str]:
        """Get columns that have a specific tag."""
        return [name for name, meta in self.columns.items() if tag in meta.tags]
    
    def get_columns_by_sensitivity(self, level: str) -> list[str]:
        """Get columns at a specific sensitivity level."""
        return [name for name, meta in self.columns.items() 
                if meta.sensitivity_level == level]
    
    def validate_schema_columns(self) -> list[str]:
        """
        Check that all schema columns have metadata defined.
        Returns list of columns missing metadata.
        """
        if self.schema is None:
            return []
        schema_columns = set(self.schema.fieldNames())
        metadata_columns = set(self.columns.keys())
        return list(schema_columns - metadata_columns)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "table_name": self.table_name,
            "schema_json": self.schema.json() if self.schema else None,
            "columns": {k: v.to_dict() for k, v in self.columns.items()},
            "description": self.description,
            "data_owner": self.data_owner,
            "layer": self.layer.value if self.layer else None,
            "source_tables": self.source_tables,
            "refresh_schedule": self.refresh_schedule,
            "sla_hours": self.sla_hours,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SchemaMetadata:
        """Deserialize from dictionary."""
        return cls(
            table_name=data["table_name"],
            schema=StructType.fromJson(data["schema_json"]) if data.get("schema_json") else None,
            columns={k: ColumnMetadata.from_dict(v) for k, v in data.get("columns", {}).items()},
            description=data.get("description", ""),
            data_owner=data.get("data_owner"),
            layer=LayerPolicy(data["layer"]) if data.get("layer") else None,
            source_tables=data.get("source_tables", []),
            refresh_schedule=data.get("refresh_schedule"),
            sla_hours=data.get("sla_hours"),
            tags=data.get("tags", []),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.utcnow(),
        )
    
    @classmethod
    def from_schema(
        cls,
        table_name: str,
        schema: StructType,
        layer: Optional[LayerPolicy] = None,
        auto_detect_pii: bool = False,
    ) -> SchemaMetadata:
        """
        Create SchemaMetadata from a StructType with auto-generated column metadata.
        
        Args:
            table_name: Table identifier
            schema: PySpark schema
            layer: Medallion layer
            auto_detect_pii: If True, detect PII columns by name patterns
            
        Example:
            >>> meta = SchemaMetadata.from_schema(
            ...     "silver.users",
            ...     user_schema,
            ...     layer=LayerPolicy.SILVER,
            ...     auto_detect_pii=True
            ... )
        """
        pii_patterns = ["email", "phone", "ssn", "password", "credit_card", 
                        "address", "birth", "salary", "income"]
        
        instance = cls(
            table_name=table_name,
            schema=schema,
            layer=layer,
        )
        
        for f in schema.fields:
            is_pii = False
            if auto_detect_pii:
                field_lower: str = f.name.lower()
                is_pii = any(pattern in field_lower for pattern in pii_patterns)
            
            instance.columns[f.name] = ColumnMetadata(
                column_name=f.name,
                data_type=f.dataType.simpleString(),
                pii=is_pii,
                sensitivity_level="confidential" if is_pii else "internal",
            )
        
        return instance
