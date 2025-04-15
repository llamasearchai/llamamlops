"""
LlamaMlOps: MLOps framework for managing machine learning lifecycles.

A comprehensive package for orchestrating, deploying, and monitoring machine learning
models in production environments.
"""

__version__ = "0.1.0"

# CLI tools
from llamamlops.cli.commands import deploy, init, monitor, track
from llamamlops.core.environment import Environment, EnvironmentConfig

# Core components
from llamamlops.core.project import Project, ProjectConfig
from llamamlops.core.registry import ModelRegistry
from llamamlops.deployment.container import ContainerBuilder

# Deployment
from llamamlops.deployment.model import DeploymentConfig, ModelDeployer
from llamamlops.deployment.runtime import RuntimeEnvironment
from llamamlops.monitoring.alerts import AlertManager
from llamamlops.monitoring.drift import DriftDetector
from llamamlops.monitoring.logging import LogMonitor

# Monitoring
from llamamlops.monitoring.metrics import MetricsMonitor

# Pipelines
from llamamlops.pipelines.pipeline import Pipeline, PipelineConfig
from llamamlops.pipelines.scheduler import PipelineScheduler
from llamamlops.pipelines.stage import PipelineStage

# Serving
from llamamlops.serving.api import ModelServer, ServerConfig
from llamamlops.serving.inference import InferenceEngine
from llamamlops.serving.scaling import ScalingManager
from llamamlops.tracking.artifacts import ArtifactManager

# Tracking
from llamamlops.tracking.experiment import Experiment, ExperimentTracker
from llamamlops.tracking.metrics import MetricsTracker
from llamamlops.versioning.code import CodeVersion, CodeVersioning

# Versioning
from llamamlops.versioning.data import DataVersion, DataVersioning
from llamamlops.versioning.model import ModelVersion, ModelVersioning

__all__ = [
    # Core
    "Project",
    "ProjectConfig",
    "Environment",
    "EnvironmentConfig",
    "ModelRegistry",
    # Tracking
    "Experiment",
    "ExperimentTracker",
    "MetricsTracker",
    "ArtifactManager",
    # Deployment
    "ModelDeployer",
    "DeploymentConfig",
    "ContainerBuilder",
    "RuntimeEnvironment",
    # Serving
    "ModelServer",
    "ServerConfig",
    "InferenceEngine",
    "ScalingManager",
    # Monitoring
    "MetricsMonitor",
    "AlertManager",
    "LogMonitor",
    "DriftDetector",
    # Versioning
    "DataVersion",
    "DataVersioning",
    "ModelVersion",
    "ModelVersioning",
    "CodeVersion",
    "CodeVersioning",
    # Pipelines
    "Pipeline",
    "PipelineConfig",
    "PipelineStage",
    "PipelineScheduler",
    # CLI tools
    "init",
    "deploy",
    "monitor",
    "track",
]
