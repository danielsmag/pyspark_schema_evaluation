from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from pyspark.sql.types import StructType

from pyspark_schema_evaluation.models.models import SchemaDiff, VersionedSchema
from pyspark_schema_evaluation.core._enums import LayerPolicy
from pyspark_schema_evaluation.models.models import SchemaMetadata, ColumnMetadata


class ISchemaRegistry(ABC):
    """
    Interface for schema registry operations.
    
    The schema registry provides:
    - Schema versioning and history tracking
    - Compatibility checking between schema versions
    - Metadata storage for tables and columns
    """
    
    @abstractmethod
    def register_schema(
        self,
        table_name: str,
        schema: StructType,
        layer: LayerPolicy,
        description: Optional[str] = None,
    ) -> VersionedSchema:
        """Register a new schema version for a table."""
        pass
    
    @abstractmethod
    def get_latest_schema(self, table_name: str) -> Optional[VersionedSchema]:
        """Get the latest schema version for a table."""
        pass
    
    @abstractmethod
    def get_schema_by_version(self, table_name: str, version: int) -> Optional[VersionedSchema]:
        """Get a specific schema version."""
        pass
    
    @abstractmethod
    def get_schema_history(self, table_name: str) -> list[VersionedSchema]:
        """Get all schema versions for a table."""
        pass
    
    @abstractmethod
    def check_compatibility(
        self,
        table_name: str,
        new_schema: StructType,
    ) -> tuple[bool, Optional[SchemaDiff]]:
        """Check if new schema is compatible with current schema."""
        pass
    
    # =========================================================================
    # METADATA TRACKING
    # =========================================================================
    
    @abstractmethod
    def register_metadata(
        self,
        table_name: str,
        metadata: SchemaMetadata,
    ) -> None:
        """
        Register or update metadata for a table.
        
        Args:
            table_name: Table identifier
            metadata: SchemaMetadata to store
        """
        pass
    
    @abstractmethod
    def get_metadata(self, table_name: str) -> Optional[SchemaMetadata]:
        """
        Get metadata for a table.
        
        Args:
            table_name: Table identifier
            
        Returns:
            SchemaMetadata if exists, None otherwise
        """
        pass
    
    @abstractmethod
    def update_column_metadata(
        self,
        table_name: str,
        column_metadata: ColumnMetadata,
    ) -> None:
        """
        Update metadata for a specific column.
        
        Args:
            table_name: Table identifier
            column_metadata: Updated column metadata
        """
        pass
    
    @abstractmethod
    def get_pii_columns(self, table_name: str) -> list[str]:
        """
        Get list of PII columns for a table.
        
        Args:
            table_name: Table identifier
            
        Returns:
            List of column names marked as PII
        """
        pass
    
    @abstractmethod
    def get_tables_by_owner(self, data_owner: str) -> list[str]:
        """
        Get all tables owned by a specific team/person.
        
        Args:
            data_owner: Owner identifier
            
        Returns:
            List of table names
        """
        pass
