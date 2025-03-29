"""
Setup script for the llamamlops package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llamamlops",
    version="0.1.0",
    author="LlamaSearch Team",
    author_email="team@llamasearch.ai",
    description="MLOps framework for managing machine learning lifecycles",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/llamasearch/llamamlops",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pyyaml>=5.1",
        "tqdm>=4.46.0",
        "click>=7.1.2",
        "docker>=5.0.0",
        "psutil>=5.8.0",
        "requests>=2.25.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.5b2",
            "isort>=5.9.1",
            "mypy>=0.812",
            "flake8>=3.9.2",
        ],
        "aws": [
            "boto3>=1.17.0",
            "sagemaker>=2.35.0",
        ],
        "azure": [
            "azure-storage-blob>=12.8.0",
            "azure-ml>=0.0.1",
            "azure-mgmt-containerinstance>=7.0.0",
        ],
        "gcp": [
            "google-cloud-storage>=1.38.0",
            "google-cloud-bigquery>=2.20.0",
            "google-cloud-aiplatform>=1.0.0",
        ],
        "mlflow": [
            "mlflow>=1.15.0",
        ],
        "kubernetes": [
            "kubernetes>=12.0.0",
        ],
        "tracking": [
            "tensorboard>=2.5.0",
            "wandb>=0.12.0",
        ],
        "all": [
            "boto3>=1.17.0",
            "sagemaker>=2.35.0",
            "azure-storage-blob>=12.8.0",
            "azure-ml>=0.0.1",
            "azure-mgmt-containerinstance>=7.0.0",
            "google-cloud-storage>=1.38.0",
            "google-cloud-bigquery>=2.20.0",
            "google-cloud-aiplatform>=1.0.0",
            "mlflow>=1.15.0",
            "kubernetes>=12.0.0",
            "tensorboard>=2.5.0",
            "wandb>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llamamlops=llamamlops.cli.commands:main",
        ],
    },
) 