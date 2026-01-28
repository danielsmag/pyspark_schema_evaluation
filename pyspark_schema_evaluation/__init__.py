from typing import List

# Main classes
from pyspark_schema_evaluation.schema_evolution import SchemaEvolution
from pyspark_schema_evaluation.i_schema_evolution import ISchemaEvolution

# Registry interfaces and implementations
from pyspark_schema_evaluation.registry.i_schema_registry import ISchemaRegistry
from pyspark_schema_evaluation.registry.schema_registry import InMemorySchemaRegistry
from pyspark_schema_evaluation.registry.policy_registry import LayerPolicyRegistry

# Enums
from pyspark_schema_evaluation.core._enums import (
    LayerPolicy,
    CompatibilityMode,
    ConflictMode,
    DriftSeverity,
    DriftType,
    BreakingChangeType,
)

# Models
from pyspark_schema_evaluation.models.models import (
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
)

# Exceptions
from pyspark_schema_evaluation.core.exceptions import SchemaEvolutionError

__all__: List[str] = [
    # Version
    "__version__",
    # Main classes
    "SchemaEvolution",
    "ISchemaEvolution",
    # Registry
    "ISchemaRegistry",
    "InMemorySchemaRegistry",
    "LayerPolicyRegistry",
    # Enums
    "LayerPolicy",
    "CompatibilityMode",
    "ConflictMode",
    "DriftSeverity",
    "DriftType",
    "BreakingChangeType",
    # Models
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
    # Exceptions
    "SchemaEvolutionError",
]

__version__ = "0.1.0"
