"""
Core project management for LlamaMlOps.

This module defines the Project class for managing ML projects, along with
the ProjectConfig for configuring project settings.
"""

import datetime
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class ProjectConfig:
    """Configuration for an ML project."""

    name: str
    description: str = ""
    version: str = "0.1.0"

    # Project metadata
    author: str = ""
    email: Optional[str] = None
    organization: Optional[str] = None
    repository: Optional[str] = None

    # Project structure
    root_dir: str = "."
    data_dir: str = "data"
    models_dir: str = "models"
    artifacts_dir: str = "artifacts"
    experiments_dir: str = "experiments"

    # Environment configuration
    python_version: str = "3.8"
    dependencies: List[str] = field(default_factory=list)
    env_variables: Dict[str, str] = field(default_factory=dict)

    # Workflow configuration
    stages: List[str] = field(
        default_factory=lambda: ["data", "train", "evaluate", "deploy"]
    )
    default_executor: str = "local"
    executors: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Tracking configuration
    tracking_uri: Optional[str] = None
    experiment_name: Optional[str] = None

    # Additional custom configuration
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary."""
        return asdict(self)

    def to_yaml(self, path: Optional[str] = None) -> Optional[str]:
        """Convert the config to YAML and optionally save to a file.

        Args:
            path: Path to save the YAML file. If None, returns the YAML string.

        Returns:
            YAML string if path is None, otherwise None.
        """
        config_dict = self.to_dict()
        yaml_str = yaml.dump(config_dict, default_flow_style=False)

        if path:
            with open(path, "w") as f:
                f.write(yaml_str)
            return None

        return yaml_str

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ProjectConfig":
        """Create a config from a dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, path: str) -> "ProjectConfig":
        """Load a config from a YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_json(cls, path: str) -> "ProjectConfig":
        """Load a config from a JSON file."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class Project:
    """Main project management class for LlamaMlOps."""

    def __init__(self, config: Union[ProjectConfig, Dict[str, Any], str]):
        """Initialize a project with the given configuration.

        Args:
            config: Project configuration. Can be a ProjectConfig instance,
                a dictionary, or a path to a YAML/JSON file.
        """
        if isinstance(config, str):
            # Load from file
            if config.endswith(".yaml") or config.endswith(".yml"):
                self.config = ProjectConfig.from_yaml(config)
            elif config.endswith(".json"):
                self.config = ProjectConfig.from_json(config)
            else:
                raise ValueError(f"Unsupported config file format: {config}")
        elif isinstance(config, dict):
            self.config = ProjectConfig.from_dict(config)
        elif isinstance(config, ProjectConfig):
            self.config = config
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")

        self.root_dir = Path(self.config.root_dir).absolute()
        self.initialized = False

        # Initialize component managers
        self._registry = None
        self._experiment_tracker = None
        self._deployer = None
        self._pipeline = None

    @property
    def name(self) -> str:
        """Get the project name."""
        return self.config.name

    @property
    def registry(self):
        """Get the model registry for this project."""
        if self._registry is None:
            from llamamlops.core.registry import ModelRegistry

            self._registry = ModelRegistry(
                storage_dir=str(self.root_dir / self.config.models_dir),
                project_name=self.config.name,
            )
        return self._registry

    @property
    def experiment_tracker(self):
        """Get the experiment tracker for this project."""
        if self._experiment_tracker is None:
            from llamamlops.tracking.experiment import ExperimentTracker

            self._experiment_tracker = ExperimentTracker(
                tracking_uri=self.config.tracking_uri,
                default_experiment_name=self.config.experiment_name or self.config.name,
                artifacts_dir=str(self.root_dir / self.config.artifacts_dir),
            )
        return self._experiment_tracker

    @property
    def deployer(self):
        """Get the model deployer for this project."""
        if self._deployer is None:
            from llamamlops.deployment.model import ModelDeployer

            self._deployer = ModelDeployer(
                project_name=self.config.name, registry=self.registry
            )
        return self._deployer

    def initialize(self, force: bool = False) -> None:
        """Initialize the project structure.

        Args:
            force: If True, overwrite existing directories.
        """
        if self.initialized and not force:
            print(f"Project {self.config.name} is already initialized.")
            return

        # Create directories
        dirs = [
            self.config.data_dir,
            self.config.models_dir,
            self.config.artifacts_dir,
            self.config.experiments_dir,
        ]

        for dir_name in dirs:
            dir_path = self.root_dir / dir_name
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"Created directory: {dir_path}")

        # Save project configuration
        config_path = self.root_dir / "llamamlops.yaml"
        self.config.to_yaml(str(config_path))
        print(f"Saved project configuration to: {config_path}")

        # Initialize environment
        self._init_environment()

        self.initialized = True
        print(f"Project {self.config.name} initialized successfully.")

    def _init_environment(self) -> None:
        """Initialize the project environment."""
        from llamamlops.core.environment import Environment

        env = Environment(self.config)
        env.initialize()

    def create_experiment(self, name: Optional[str] = None) -> Any:
        """Create a new experiment.

        Args:
            name: Name of the experiment. If None, uses a timestamped name.

        Returns:
            The created experiment.
        """
        if name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"{self.config.name}_{timestamp}"

        return self.experiment_tracker.create_experiment(name)

    def get_pipeline(self, config_path: Optional[str] = None):
        """Get the pipeline for this project.

        Args:
            config_path: Path to the pipeline configuration. If None, uses default.

        Returns:
            The pipeline.
        """
        if self._pipeline is None:
            from llamamlops.pipelines.pipeline import Pipeline

            if config_path is None:
                config_path = self.root_dir / "pipeline.yaml"
                if not config_path.exists():
                    config_path = None

            self._pipeline = Pipeline.from_config(config_path, project=self)

        return self._pipeline

    @classmethod
    def create(
        cls, name: str, root_dir: str = ".", description: str = "", **kwargs
    ) -> "Project":
        """Create a new project.

        Args:
            name: Project name.
            root_dir: Root directory for the project.
            description: Project description.
            **kwargs: Additional ProjectConfig parameters.

        Returns:
            The created project.
        """
        config = ProjectConfig(
            name=name, description=description, root_dir=root_dir, **kwargs
        )

        project = cls(config)
        project.initialize()

        return project

    @classmethod
    def load(cls, path: str) -> "Project":
        """Load a project from a configuration file.

        Args:
            path: Path to the configuration file.

        Returns:
            The loaded project.
        """
        return cls(path)

    def __repr__(self) -> str:
        """Get string representation of the project."""
        return f"Project(name={self.config.name}, root_dir={self.root_dir})"
