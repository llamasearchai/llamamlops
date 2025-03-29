"""
Model registry for LlamaMlOps.

This module defines the ModelRegistry class for versioning and storing ML models.
"""

import os
import json
import shutil
import yaml
import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import uuid


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""
    name: str
    version: str
    description: str = ""
    
    # Model information
    framework: str = "unknown"
    task_type: str = "unknown"
    tags: List[str] = field(default_factory=list)
    
    # Author information
    author: str = ""
    email: Optional[str] = None
    
    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Training information
    training_dataset: Optional[str] = None
    training_params: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    updated_at: Optional[str] = None
    
    # Storage information
    path: Optional[str] = None
    size_bytes: Optional[int] = None
    
    # Deployment information
    status: str = "registered"  # registered, deployed, archived
    deployment_env: Optional[str] = None
    deployment_url: Optional[str] = None
    
    # Additional custom metadata
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to a dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, metadata_dict: Dict[str, Any]) -> "ModelMetadata":
        """Create metadata from a dictionary."""
        return cls(**metadata_dict)


class ModelRegistry:
    """Registry for versioning and storing ML models."""
    
    def __init__(self, storage_dir: str, project_name: Optional[str] = None):
        """Initialize the model registry.
        
        Args:
            storage_dir: Directory to store models.
            project_name: Optional project name for namespacing.
        """
        self.storage_dir = Path(storage_dir).absolute()
        self.project_name = project_name
        
        # Create the storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Initialize registry index file if it doesn't exist
        self.index_file = self.storage_dir / "registry_index.json"
        if not self.index_file.exists():
            self._write_index({})
    
    def register_model(self, 
                      model_path: str, 
                      name: str, 
                      version: Optional[str] = None, 
                      description: str = "",
                      metadata: Optional[Dict[str, Any]] = None) -> ModelMetadata:
        """Register a model in the registry.
        
        Args:
            model_path: Path to the model file or directory.
            name: Name of the model.
            version: Version of the model. If None, auto-generates a version.
            description: Description of the model.
            metadata: Additional metadata for the model.
            
        Returns:
            ModelMetadata for the registered model.
        """
        # Generate a version if not provided
        if version is None:
            version = self._generate_version()
        
        # Create metadata
        model_metadata = ModelMetadata(
            name=name,
            version=version,
            description=description
        )
        
        # Update with additional metadata if provided
        if metadata:
            for key, value in metadata.items():
                if hasattr(model_metadata, key):
                    setattr(model_metadata, key, value)
                else:
                    model_metadata.custom[key] = value
        
        # Create the model directory
        model_dir = self._get_model_dir(name, version)
        os.makedirs(model_dir, exist_ok=True)
        
        # Copy the model file or directory
        model_path = Path(model_path)
        if model_path.is_file():
            target_path = model_dir / model_path.name
            shutil.copy2(model_path, target_path)
            model_metadata.path = str(target_path.relative_to(self.storage_dir))
            model_metadata.size_bytes = os.path.getsize(model_path)
        elif model_path.is_dir():
            target_path = model_dir / "model"
            if target_path.exists():
                shutil.rmtree(target_path)
            shutil.copytree(model_path, target_path)
            model_metadata.path = str(target_path.relative_to(self.storage_dir))
            model_metadata.size_bytes = sum(
                f.stat().st_size for f in target_path.glob('**/*') if f.is_file()
            )
        else:
            raise ValueError(f"Model path does not exist: {model_path}")
        
        # Save metadata
        metadata_path = model_dir / "metadata.yaml"
        with open(metadata_path, 'w') as f:
            yaml.dump(model_metadata.to_dict(), f, default_flow_style=False)
        
        # Update the registry index
        self._update_index(name, version, model_metadata)
        
        print(f"Model {name} (version {version}) registered successfully.")
        return model_metadata
    
    def get_model(self, name: str, version: Optional[str] = None) -> str:
        """Get the path to a model in the registry.
        
        Args:
            name: Name of the model.
            version: Version of the model. If None, uses the latest version.
            
        Returns:
            Path to the model.
            
        Raises:
            ValueError: If the model doesn't exist.
        """
        if version is None:
            version = self.get_latest_version(name)
            if version is None:
                raise ValueError(f"No versions found for model: {name}")
        
        model_dir = self._get_model_dir(name, version)
        metadata = self.get_model_metadata(name, version)
        
        if metadata.path:
            model_path = self.storage_dir / metadata.path
            if model_path.exists():
                return str(model_path)
        
        # Try to find the model file or directory
        for item in os.listdir(model_dir):
            if item != "metadata.yaml":
                return str(model_dir / item)
        
        raise ValueError(f"Model file not found for {name} (version {version})")
    
    def get_model_metadata(self, name: str, version: Optional[str] = None) -> ModelMetadata:
        """Get metadata for a model.
        
        Args:
            name: Name of the model.
            version: Version of the model. If None, uses the latest version.
            
        Returns:
            ModelMetadata for the model.
            
        Raises:
            ValueError: If the model doesn't exist.
        """
        if version is None:
            version = self.get_latest_version(name)
            if version is None:
                raise ValueError(f"No versions found for model: {name}")
        
        model_dir = self._get_model_dir(name, version)
        metadata_path = model_dir / "metadata.yaml"
        
        if not metadata_path.exists():
            raise ValueError(f"Metadata not found for model {name} (version {version})")
        
        with open(metadata_path, 'r') as f:
            metadata_dict = yaml.safe_load(f)
        
        return ModelMetadata.from_dict(metadata_dict)
    
    def update_model_metadata(self, name: str, version: str, 
                             metadata_updates: Dict[str, Any]) -> ModelMetadata:
        """Update metadata for a model.
        
        Args:
            name: Name of the model.
            version: Version of the model.
            metadata_updates: Dictionary of metadata fields to update.
            
        Returns:
            Updated ModelMetadata.
            
        Raises:
            ValueError: If the model doesn't exist.
        """
        metadata = self.get_model_metadata(name, version)
        
        # Update metadata fields
        for key, value in metadata_updates.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
            else:
                metadata.custom[key] = value
        
        # Update timestamp
        metadata.updated_at = datetime.datetime.now().isoformat()
        
        # Save updated metadata
        model_dir = self._get_model_dir(name, version)
        metadata_path = model_dir / "metadata.yaml"
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata.to_dict(), f, default_flow_style=False)
        
        # Update the registry index
        self._update_index(name, version, metadata)
        
        return metadata
    
    def list_models(self) -> List[str]:
        """List all registered model names.
        
        Returns:
            List of model names.
        """
        index = self._read_index()
        return list(index.keys())
    
    def list_versions(self, name: str) -> List[str]:
        """List all versions for a model.
        
        Args:
            name: Name of the model.
            
        Returns:
            List of versions sorted by creation time (newest first).
            
        Raises:
            ValueError: If the model doesn't exist.
        """
        index = self._read_index()
        if name not in index:
            raise ValueError(f"Model not found: {name}")
        
        versions = list(index[name].keys())
        
        # Sort versions by creation time (newest first)
        versions.sort(key=lambda v: index[name][v].get("created_at", ""), reverse=True)
        
        return versions
    
    def get_latest_version(self, name: str) -> Optional[str]:
        """Get the latest version for a model.
        
        Args:
            name: Name of the model.
            
        Returns:
            Latest version or None if the model doesn't exist.
        """
        try:
            versions = self.list_versions(name)
            return versions[0] if versions else None
        except ValueError:
            return None
    
    def delete_model(self, name: str, version: Optional[str] = None) -> bool:
        """Delete a model from the registry.
        
        Args:
            name: Name of the model.
            version: Version of the model. If None, deletes all versions.
            
        Returns:
            True if the model was deleted, False otherwise.
        """
        index = self._read_index()
        if name not in index:
            return False
        
        if version is None:
            # Delete all versions
            versions = list(index[name].keys())
            for v in versions:
                self._delete_model_version(name, v)
            
            # Remove from index
            del index[name]
            self._write_index(index)
            
            print(f"All versions of model {name} deleted.")
            return True
        else:
            # Delete specific version
            if version not in index[name]:
                return False
            
            self._delete_model_version(name, version)
            
            # Remove from index
            del index[name][version]
            if not index[name]:
                del index[name]
            self._write_index(index)
            
            print(f"Model {name} (version {version}) deleted.")
            return True
    
    def _delete_model_version(self, name: str, version: str) -> None:
        """Delete a specific model version.
        
        Args:
            name: Name of the model.
            version: Version of the model.
        """
        model_dir = self._get_model_dir(name, version)
        if model_dir.exists():
            shutil.rmtree(model_dir)
    
    def _get_model_dir(self, name: str, version: str) -> Path:
        """Get the directory for a model version.
        
        Args:
            name: Name of the model.
            version: Version of the model.
            
        Returns:
            Path to the model directory.
        """
        if self.project_name:
            return self.storage_dir / self.project_name / name / version
        else:
            return self.storage_dir / name / version
    
    def _generate_version(self) -> str:
        """Generate a new version string.
        
        Returns:
            A version string in the format 'YYYYMMDDHHmmss-uuid'.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        uid = str(uuid.uuid4())[:8]
        return f"{timestamp}-{uid}"
    
    def _read_index(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Read the registry index file.
        
        Returns:
            Dictionary mapping model names to versions and metadata.
        """
        if not self.index_file.exists():
            return {}
        
        with open(self.index_file, 'r') as f:
            return json.load(f)
    
    def _write_index(self, index: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """Write the registry index file.
        
        Args:
            index: Dictionary mapping model names to versions and metadata.
        """
        with open(self.index_file, 'w') as f:
            json.dump(index, f, indent=2)
    
    def _update_index(self, name: str, version: str, metadata: ModelMetadata) -> None:
        """Update the registry index with model metadata.
        
        Args:
            name: Name of the model.
            version: Version of the model.
            metadata: ModelMetadata for the model.
        """
        index = self._read_index()
        
        # Create entries if they don't exist
        if name not in index:
            index[name] = {}
        
        # Update version metadata
        index[name][version] = {
            "created_at": metadata.created_at,
            "description": metadata.description,
            "path": metadata.path,
            "status": metadata.status
        }
        
        self._write_index(index)
    
    def __repr__(self) -> str:
        """String representation of the registry."""
        return f"ModelRegistry(storage_dir={self.storage_dir}, project_name={self.project_name})" 