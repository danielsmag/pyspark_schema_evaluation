from __future__ import annotations

from pyspark_schema_evaluation.models.models import LayerPolicyConfig
from typing import Optional
from abc import ABC, abstractmethod


__all__: list[str] = [
    "LayerPolicyRegistry",
]

class IPolicyRegistry(ABC):
    @abstractmethod
    def register(self, name: str, config: LayerPolicyConfig) -> None:
        pass

    @abstractmethod
    def update(self, name: str, config: LayerPolicyConfig) -> None:
        pass


class LayerPolicyRegistry:
    """
    Registry for custom layer policies.
    
    Allows users to define their own policies beyond BRONZE/SILVER/GOLD.
    
    Example:
        from pyspark_schema_evaluation.core._enums import (
            LayerPolicyRegistry, CompatibilityMode, ConflictMode
        )
        from pyspark_schema_evaluation.models.models import LayerPolicyConfig
        
        # Register a "STAGING" policy (permissive like bronze but with type enforcement)
        LayerPolicyRegistry.register(
            "STAGING",
            LayerPolicyConfig(
                compatibility_mode=CompatibilityMode.BACKWARDS,
                extra_columns_allowed=True,
                on_type_conflict=ConflictMode.INFORCE,
                allow_nullable_changes=True,
                require_all_columns=False,
            )
        )
        
        # Register a "PLATINUM" policy (stricter than gold)
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
        
        # Get the policy config
        config = LayerPolicyRegistry.get("STAGING")
    """
    
    _registry: dict[str, LayerPolicyConfig] = {}
    
    @classmethod
    def register(cls, name: str, config: LayerPolicyConfig) -> None:
        """
        Register a custom layer policy.
        
        Args:
            name: Unique name for the policy (e.g., "STAGING", "PLATINUM")
            config: LayerPolicyConfig with the policy settings
            
        Raises:
            ValueError: If policy name already exists
        """
        name_upper: str = name.upper()
        if name_upper in cls._registry:
            raise ValueError(
                f"Policy '{name_upper}' already registered. "
                f"Use update() to modify existing policies."
            )
        cls._registry[name_upper] = config
    
    @classmethod
    def update(cls, name: str, config: LayerPolicyConfig) -> None:
        """
        Update an existing custom policy or register a new one.
        
        Args:
            name: Policy name
            config: New LayerPolicyConfig
        """
        cls._registry[name.upper()] = config
    
    @classmethod
    def get(cls, name: str) -> Optional[LayerPolicyConfig]:
        """
        Get a custom policy configuration by name.
        
        Args:
            name: Policy name
            
        Returns:
            LayerPolicyConfig if found, None otherwise
        """
        return cls._registry.get(name.upper())
    
    @classmethod
    def exists(cls, name: str) -> bool:
        """Check if a custom policy exists."""
        return name.upper() in cls._registry
    
    @classmethod
    def list_policies(cls) -> list[str]:
        """List all registered custom policy names."""
        return list(cls._registry.keys())
    
    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        Remove a custom policy.
        
        Args:
            name: Policy name to remove
            
        Returns:
            True if removed, False if not found
        """
        name_upper: str = name.upper()
        if name_upper in cls._registry:
            del cls._registry[name_upper]
            return True
        return False
    
    @classmethod
    def clear(cls) -> None:
        """Remove all custom policies."""
        cls._registry.clear()

