from __future__ import annotations

from enum import Enum


class DriftSeverity(str, Enum):
    """
    Severity level for schema drift alerts.
    
    Used to categorize the impact of detected schema changes:
    - INFO: Additive changes that are backwards compatible (new nullable columns)
    - WARNING: Changes that may affect downstream consumers (type promotions)
    - CRITICAL: Breaking changes that will cause failures (column removal, nullable->non-null)
    """
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class DriftType(str, Enum):
    """
    Types of schema drift that can be detected.
    
    Categories:
    - COLUMN_ADDED: New column appeared in source
    - COLUMN_REMOVED: Column was removed from source
    - TYPE_CHANGED: Column data type changed
    - NULLABLE_CHANGED: Column nullability changed
    - COLUMN_RENAMED: Column appears to have been renamed (heuristic)
    - METADATA_CHANGED: Column metadata changed (description, PII flag, etc.)
    """
    COLUMN_ADDED = "column_added"
    COLUMN_REMOVED = "column_removed"
    TYPE_CHANGED = "type_changed"
    NULLABLE_CHANGED = "nullable_changed"
    COLUMN_RENAMED = "column_renamed"
    METADATA_CHANGED = "metadata_changed"


class BreakingChangeType(str, Enum):
    """
    Types of breaking changes that would affect downstream consumers.
    
    Breaking changes are changes that could cause:
    - Query failures (removed columns, incompatible types)
    - Data loss (narrowing type conversions)
    - Constraint violations (nullable -> non-nullable)
    
    Used for pre-evolution impact analysis.
    """
    COLUMN_REMOVAL = "column_removal"
    TYPE_NARROWING = "type_narrowing"  # e.g., DOUBLE -> INT (data loss)
    NULLABLE_TO_REQUIRED = "nullable_to_required"
    INCOMPATIBLE_TYPE = "incompatible_type"  # e.g., STRING -> INT


class LayerPolicy(str, Enum):
    """
    Defines schema evolution policies for each medallion layer.
    
    Built-in policies:
    - BRONZE: Permissive - accept all columns, schema-on-read
    - SILVER: Balanced - enforce schema with safe promotions, add missing as nullable
    - GOLD: Strict - no schema changes allowed, fail on mismatch
    
    Custom policies can be registered using LayerPolicyRegistry.register().
    
    Example:
        # Register a custom policy
        LayerPolicyRegistry.register(
            "PLATINUM",
            LayerPolicyConfig(
                compatibility_mode=CompatibilityMode.STRICT,
                extra_columns_allowed=False,
                on_type_conflict=ConflictMode.ERROR,
                allow_nullable_changes=False,
                require_all_columns=True,
            )
        )
        
        # Use it
        @evolve_for_layer(target_schema=..., layer=LayerPolicy.CUSTOM, custom_policy="PLATINUM")
    """
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    CUSTOM = "custom"


class ConflictMode(str, Enum):
    STRING = "string"
    FIRST = "first"
    SECOND = "second"
    IGNORE = "ignore"
    DROP = "drop"
    ERROR = "error"
    INFORCE = "inforce"


class CompatibilityMode(str, Enum):
    BACKWARDS = "backwards"
    FORWARDS = "forwards"
    FULL = "full"
    STRICT = "strict"