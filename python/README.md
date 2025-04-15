# LlamaMlOps Python Package

The Python implementation of LlamaMlOps, a comprehensive MLOps framework for managing machine learning lifecycles.

## Installation

```bash
# From the current directory
pip install .

# With specific features
pip install .[aws]
pip install .[azure]
pip install .[gcp]
pip install .[all]
```

## Development

For development, install the package in development mode with dev extras:

```bash
pip install -e .[dev]
```

Run tests:

```bash
pytest
```

Run linting:

```bash
flake8 llamamlops
black llamamlops
isort llamamlops
mypy llamamlops
```

## Package Structure

```
llamamlops/
├── core/                 # Core functionality
│   ├── project.py        # Project management
│   ├── environment.py    # Environment management
│   └── registry.py       # Model registry
├── tracking/             # Experiment tracking
├── deployment/           # Model deployment
├── serving/              # Model serving
├── monitoring/           # Model monitoring
├── versioning/           # Data and model versioning
├── pipelines/            # Pipeline management
├── cli/                  # Command-line interface
├── api/                  # API server
├── integrations/         # Cloud and tool integrations
└── utils/                # Utilities
```

## Building and Distribution

Build the package:

```bash
python setup.py sdist bdist_wheel
```

Upload to PyPI:

```bash
twine upload dist/*
```

## Documentation

Generate documentation:

```bash
cd docs
make html
```

View documentation:

```bash
open docs/_build/html/index.html
``` 