# LlamaMlOps

[![PyPI version](https://badge.fury.io/py/llamamlops.svg)](https://badge.fury.io/py/llamamlops)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive MLOps framework for managing machine learning lifecycles, from experimentation to production deployment.

## Features

- **Project Management**: Organize ML projects with standardized structure and configuration
- **Environment Management**: Create and manage consistent development environments
- **Experiment Tracking**: Track and compare experiment metrics, parameters, and artifacts
- **Model Registry**: Version and store models with metadata and lineage
- **Deployment**: Deploy models to various platforms (local, cloud, Kubernetes)
- **Monitoring**: Monitor model performance, drift, and system metrics
- **CI/CD Pipelines**: Automate ML workflows with configurable pipelines
- **Cloud Integration**: Seamless integration with AWS, Azure, and GCP

## Installation

```bash
# Basic installation
pip install llamamlops

# With specific features
pip install llamamlops[aws]      # AWS integration
pip install llamamlops[azure]    # Azure integration
pip install llamamlops[gcp]      # GCP integration
pip install llamamlops[mlflow]   # MLflow integration
pip install llamamlops[kubernetes] # Kubernetes integration
pip install llamamlops[tracking] # Experiment tracking with TensorBoard and W&B

# Full installation
pip install llamamlops[all]
```

## Quick Start

### Create a new ML project

```python
from llamamlops import Project

# Create a new project
project = Project.create(
    name="my-ml-project",
    description="My ML project description",
    root_dir="./my-ml-project"
)

# Initialize the project structure
project.initialize()
```

### Track an experiment

```python
from llamamlops import Project
import numpy as np

# Load existing project
project = Project.load("./my-ml-project/llamamlops.yaml")

# Create an experiment
experiment = project.create_experiment("experiment-1")

# Log parameters
experiment.log_params({
    "learning_rate": 0.01,
    "batch_size": 32,
    "epochs": 10
})

# Log metrics (e.g., during training)
for epoch in range(10):
    accuracy = 0.75 + 0.2 * (1 - np.exp(-0.5 * epoch))
    loss = 0.5 * np.exp(-0.5 * epoch)
    experiment.log_metrics({
        "accuracy": accuracy,
        "loss": loss
    }, step=epoch)

# Log artifacts
experiment.log_artifact("model.pkl", "./model.pkl")

# Complete the experiment
experiment.complete()
```

### Register a model

```python
from llamamlops import Project

# Load project
project = Project.load("./my-ml-project/llamamlops.yaml")

# Register model
model_metadata = project.registry.register_model(
    model_path="./model.pkl",
    name="my-model",
    description="My trained model",
    metadata={
        "framework": "scikit-learn",
        "task_type": "classification",
        "metrics": {"accuracy": 0.95},
        "author": "John Doe"
    }
)

print(f"Model {model_metadata.name} (version {model_metadata.version}) registered")
```

### Deploy a model

```python
from llamamlops import Project

# Load project
project = Project.load("./my-ml-project/llamamlops.yaml")

# Deploy the latest version of a model
project.deployer.deploy(
    model_name="my-model",
    env_name="production",
    deployment_type="fastapi",
    replicas=2
)
```

## CLI Commands

LlamaMlOps provides a command-line interface for common operations:

```bash
# Initialize a new project
llamamlops init my-ml-project

# Create a new experiment
llamamlops experiment create experiment-1

# Register a model
llamamlops model register --path ./model.pkl --name my-model

# Deploy a model
llamamlops deploy --model my-model --env production

# Start monitoring
llamamlops monitor --deployment my-model-production
```

## Project Structure

A typical LlamaMlOps project has the following structure:

```
my-ml-project/
├── llamamlops.yaml         # Project configuration
├── data/                   # Data directory
│   ├── raw/                # Raw data
│   ├── processed/          # Processed data
│   └── external/           # External data
├── models/                 # Model registry
│   └── registry_index.json # Model registry index
├── artifacts/              # Experiment artifacts
├── experiments/            # Experiment tracking
├── notebooks/              # Jupyter notebooks
├── src/                    # Source code
│   ├── __init__.py
│   ├── data/               # Data processing scripts
│   ├── features/           # Feature engineering
│   ├── models/             # Model definitions
│   ├── training/           # Training scripts
│   └── serving/            # Serving code
├── tests/                  # Tests
├── deployment/             # Deployment configurations
│   ├── local/
│   ├── dev/
│   └── prod/
├── configs/                # Configuration files
│   ├── data_config.yaml
│   ├── model_config.yaml
│   └── training_config.yaml
├── pipeline.yaml           # Pipeline definition
└── requirements.txt        # Python dependencies
```

## Advanced Features

### Custom Pipelines

```python
from llamamlops.pipelines import Pipeline, PipelineStage

# Define a pipeline
pipeline = Pipeline(
    name="training-pipeline",
    stages=[
        PipelineStage(
            name="data-preparation",
            command=["python", "src/data/prepare_data.py"],
            inputs=["data/raw/dataset.csv"],
            outputs=["data/processed/train.csv", "data/processed/test.csv"]
        ),
        PipelineStage(
            name="training",
            command=["python", "src/models/train.py"],
            inputs=["data/processed/train.csv"],
            outputs=["artifacts/model.pkl"]
        ),
        PipelineStage(
            name="evaluation",
            command=["python", "src/models/evaluate.py"],
            inputs=["artifacts/model.pkl", "data/processed/test.csv"],
            outputs=["artifacts/metrics.json"]
        )
    ]
)

# Run the pipeline
pipeline.run()
```

### Kubernetes Integration

```python
from llamamlops.deployment import DeploymentConfig
from llamamlops import Project

# Load project
project = Project.load("./my-ml-project/llamamlops.yaml")

# Configure Kubernetes deployment
config = DeploymentConfig(
    deployment_type="kubernetes",
    namespace="ml-models",
    resources={
        "cpu": "1000m",
        "memory": "2Gi",
        "gpu": 1
    },
    scaling={
        "min_replicas": 2,
        "max_replicas": 5,
        "target_cpu_utilization": 70
    },
    environment_variables={
        "MODEL_NAME": "my-model",
        "LOG_LEVEL": "INFO"
    }
)

# Deploy
project.deployer.deploy(
    model_name="my-model",
    env_name="production",
    deployment_config=config
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
# Updated in commit 1 - 2025-04-04 17:42:05

# Updated in commit 9 - 2025-04-04 17:42:06

# Updated in commit 17 - 2025-04-04 17:42:06

# Updated in commit 25 - 2025-04-04 17:42:06

# Updated in commit 1 - 2025-04-05 14:42:07

# Updated in commit 9 - 2025-04-05 14:42:07

# Updated in commit 17 - 2025-04-05 14:42:07

# Updated in commit 25 - 2025-04-05 14:42:07

# Updated in commit 1 - 2025-04-05 15:28:16

# Updated in commit 9 - 2025-04-05 15:28:16

# Updated in commit 17 - 2025-04-05 15:28:16

# Updated in commit 25 - 2025-04-05 15:28:16

# Updated in commit 1 - 2025-04-05 16:06:39
