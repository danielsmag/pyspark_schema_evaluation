from __future__ import annotations

from typing import Optional

from pyspark.sql.types import StructType

from pyspark_schema_evaluation.models.models import SchemaDiff, VersionedSchema
from pyspark_schema_evaluation.core._enums import LayerPolicy
from pyspark_schema_evaluation.registry.i_schema_registry import ISchemaRegistry
from pyspark_schema_evaluation.models.models import SchemaMetadata, ColumnMetadata
from pyspark_schema_evaluation.utils._spark._schema import _diff_schemas
from pyspark_schema_evaluation.utils._utils import SingletonABCMeta

__all__: list[str] = [
    "ISchemaRegistry",
    "InMemorySchemaRegistry",
]


class InMemorySchemaRegistry(ISchemaRegistry, metaclass=SingletonABCMeta):
    """
    In-memory schema registry for testing and development.
    
    Provides:
    - Schema versioning and history
    - Compatibility checking
    - Metadata storage for tables and columns
    
    For production, implement DeltaTableSchemaRegistry or UnityCatalogSchemaRegistry
    that persists data to Delta tables or Unity Catalog.
    
    Example:
        >>> registry = InMemorySchemaRegistry()
        >>> 
        >>> # Register schema
        >>> versioned = registry.register_schema(
        ...     "silver.customers",
        ...     customer_schema,
        ...     LayerPolicy.SILVER,
        ...     "Initial schema"
        ... )
        >>> 
        >>> # Register metadata
        >>> metadata = SchemaMetadata.from_schema("silver.customers", customer_schema)
        >>> registry.register_metadata("silver.customers", metadata)
        >>> 
        >>> # Query PII columns
        >>> pii_cols = registry.get_pii_columns("silver.customers")
    """
    
    def __init__(self) -> None:
        self._schemas: dict[str, list[VersionedSchema]] = {}
        self._metadata: dict[str, SchemaMetadata] = {}
    
    def register_schema(
        self,
        table_name: str,
        schema: StructType,
        layer: LayerPolicy,
        description: Optional[str] = None,
    ) -> VersionedSchema:
        
        if table_name not in self._schemas:
            self._schemas[table_name] = []
        
        history: list[VersionedSchema] = self._schemas[table_name]
        new_version = len(history) + 1
        previous = history[-1].version if history else None
        
        versioned: VersionedSchema = VersionedSchema(
            schema=schema,
            version=new_version,
            layer=layer,
            description=description,
            previous_version=previous,
        )
        
        history.append(versioned)
        return versioned
    
    def get_latest_schema(self, table_name: str) -> Optional[VersionedSchema]:
        history: list[VersionedSchema] = self._schemas.get(table_name, [])
        return history[-1] if history else None
    
    def get_schema_by_version(self, table_name: str, version: int) -> Optional[VersionedSchema]:
        history: list[VersionedSchema] = self._schemas.get(table_name, [])
        for schema in history:
            if schema.version == version:
                return schema
        return None
    
    def get_schema_history(self, table_name: str) -> list[VersionedSchema]:
        return self._schemas.get(table_name, [])
    
    def check_compatibility(
        self,
        table_name: str,
        new_schema: StructType,
    ) -> tuple[bool, Optional[SchemaDiff]]:
        latest: VersionedSchema | None = self.get_latest_schema(table_name)
        if latest is None:
            return True, None
        
        diff: SchemaDiff = _diff_schemas(new_schema, latest.schema)
        return diff.is_compatible, diff

    # =========================================================================
    # METADATA TRACKING IMPLEMENTATION
    # =========================================================================
    
    def register_metadata(
        self,
        table_name: str,
        metadata: SchemaMetadata,
    ) -> None:
        """
        Register or update metadata for a table.
        
        If metadata already exists, it will be replaced.
        """
        self._metadata[table_name] = metadata
    
    def get_metadata(self, table_name: str) -> Optional[SchemaMetadata]:
        """Get metadata for a table."""
        return self._metadata.get(table_name)
    
    def update_column_metadata(
        self,
        table_name: str,
        column_metadata: ColumnMetadata,
    ) -> None:
        """
        Update metadata for a specific column.
        
        Creates table metadata if it doesn't exist.
        """
        if table_name not in self._metadata:
            # Create empty metadata for the table
            self._metadata[table_name] = SchemaMetadata(table_name=table_name)
        
        self._metadata[table_name].add_column_metadata(column_metadata)
    
    def get_pii_columns(self, table_name: str) -> list[str]:
        """Get list of PII columns for a table."""
        metadata = self._metadata.get(table_name)
        if metadata is None:
            return []
        return metadata.get_pii_columns()
    
    def get_tables_by_owner(self, data_owner: str) -> list[str]:
        """Get all tables owned by a specific team/person."""
        return [
            name for name, meta in self._metadata.items()
            if meta.data_owner == data_owner
        ]
    
    def get_all_pii_tables(self) -> dict[str, list[str]]:
        """
        Get all tables that have PII columns.
        
        Returns:
            Dict mapping table_name -> list of PII column names
        """
        result: dict[str, list[str]] = {}
        for table_name, metadata in self._metadata.items():
            pii_cols = metadata.get_pii_columns()
            if pii_cols:
                result[table_name] = pii_cols
        return result
    
    def get_columns_by_sensitivity(
        self, 
        sensitivity_level: str
    ) -> dict[str, list[str]]:
        """
        Get all columns across all tables at a specific sensitivity level.
        
        Args:
            sensitivity_level: "public", "internal", "confidential", or "restricted"
            
        Returns:
            Dict mapping table_name -> list of column names at that sensitivity
        """
        result: dict[str, list[str]] = {}
        for table_name, metadata in self._metadata.items():
            cols = metadata.get_columns_by_sensitivity(sensitivity_level)
            if cols:
                result[table_name] = cols
        return result
    
    def clear(self) -> None:
        """Clear all schemas and metadata. Useful for testing."""
        self._schemas.clear()
        self._metadata.clear()

