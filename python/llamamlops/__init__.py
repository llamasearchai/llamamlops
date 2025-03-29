"""
LlamaMlOps: MLOps framework for managing machine learning lifecycles.

A comprehensive package for orchestrating, deploying, and monitoring machine learning
models in production environments.
"""

__version__ = "0.1.0"

# Core components
from llamamlops.core.project import Project, ProjectConfig
from llamamlops.core.environment import Environment, EnvironmentConfig
from llamamlops.core.registry import ModelRegistry

# Tracking
from llamamlops.tracking.experiment import Experiment, ExperimentTracker
from llamamlops.tracking.metrics import MetricsTracker
from llamamlops.tracking.artifacts import ArtifactManager

# Deployment
from llamamlops.deployment.model import ModelDeployer, DeploymentConfig
from llamamlops.deployment.container import ContainerBuilder
from llamamlops.deployment.runtime import RuntimeEnvironment

# Serving
from llamamlops.serving.api import ModelServer, ServerConfig
from llamamlops.serving.inference import InferenceEngine
from llamamlops.serving.scaling import ScalingManager

# Monitoring
from llamamlops.monitoring.metrics import MetricsMonitor
from llamamlops.monitoring.alerts import AlertManager
from llamamlops.monitoring.logging import LogMonitor
from llamamlops.monitoring.drift import DriftDetector

# Versioning
from llamamlops.versioning.data import DataVersion, DataVersioning
from llamamlops.versioning.model import ModelVersion, ModelVersioning
from llamamlops.versioning.code import CodeVersion, CodeVersioning

# Pipelines
from llamamlops.pipelines.pipeline import Pipeline, PipelineConfig
from llamamlops.pipelines.stage import PipelineStage
from llamamlops.pipelines.scheduler import PipelineScheduler

# CLI tools
from llamamlops.cli.commands import init, deploy, monitor, track

__all__ = [
    # Core
    "Project", "ProjectConfig", "Environment", "EnvironmentConfig",
    "ModelRegistry",
    
    # Tracking
    "Experiment", "ExperimentTracker", "MetricsTracker", "ArtifactManager",
    
    # Deployment
    "ModelDeployer", "DeploymentConfig", "ContainerBuilder",
    "RuntimeEnvironment",
    
    # Serving
    "ModelServer", "ServerConfig", "InferenceEngine", "ScalingManager",
    
    # Monitoring
    "MetricsMonitor", "AlertManager", "LogMonitor", "DriftDetector",
    
    # Versioning
    "DataVersion", "DataVersioning", "ModelVersion", "ModelVersioning",
    "CodeVersion", "CodeVersioning",
    
    # Pipelines
    "Pipeline", "PipelineConfig", "PipelineStage", "PipelineScheduler",
    
    # CLI tools
    "init", "deploy", "monitor", "track"
] 