#!/usr/bin/env python3
"""
LlamaMlOps Project Workflow Example

This example demonstrates a complete MLOps workflow using LlamaMlOps, including:
1. Project creation and initialization
2. Environment management
3. Experiment tracking
4. Model training and evaluation
5. Model registration
6. Deployment
7. Monitoring setup

This is a comprehensive demonstration of LlamaMlOps capabilities.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Add the parent directory to the path to import llamamlops
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llamamlops import Project
from llamamlops.core.environment import Environment
from llamamlops.core.registry import ModelRegistry


def main():
    """Run a complete MLOps workflow example using LlamaMlOps."""
    print("LlamaMlOps Workflow Example")
    print("===========================")

    # --------------------- Project Setup ---------------------
    print("\n1. Project Setup")
    print("----------------")

    # Set project directory
    project_dir = "./diabetes_prediction"
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)

    # Create a new project
    print("\nCreating new project: Diabetes Prediction")
    project = Project.create(
        name="diabetes_prediction",
        description="Predicting diabetes progression using machine learning",
        root_dir=project_dir,
    )

    # Initialize project structure
    print("\nInitializing project structure...")
    project.initialize()

    # --------------------- Environment Setup ---------------------
    print("\n2. Environment Setup")
    print("-------------------")

    # Create development environment
    print("\nCreating development environment...")
    dev_env = Environment(
        name="development",
        description="Development environment for experimenting",
        project=project,
    )

    # Create production environment
    print("\nCreating production environment...")
    prod_env = Environment(
        name="production",
        description="Production environment for deployed models",
        project=project,
    )

    # Configure environments
    print("\nConfiguring environments...")
    dev_env.configure(
        dependencies=[
            "scikit-learn>=1.0.0",
            "pandas>=1.3.0",
            "numpy>=1.20.0",
            "matplotlib>=3.4.0",
        ],
        resources={"cpu": 4, "memory": "8Gi"},
    )

    prod_env.configure(
        dependencies=["scikit-learn>=1.0.0", "pandas>=1.3.0", "numpy>=1.20.0"],
        resources={"cpu": 2, "memory": "4Gi"},
        scaling={"min_replicas": 1, "max_replicas": 3},
    )

    # Add environments to project
    project.add_environment(dev_env)
    project.add_environment(prod_env)

    # --------------------- Data Preparation ---------------------
    print("\n3. Data Preparation")
    print("------------------")

    # Load and prepare data
    print("\nLoading diabetes dataset...")
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target

    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Feature names: {diabetes.feature_names}")

    # Create dataframe for easier manipulation
    df = pd.DataFrame(X, columns=diabetes.feature_names)
    df["target"] = y

    # Save data to project data directory
    data_dir = os.path.join(project_dir, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    raw_data_path = os.path.join(data_dir, "diabetes.csv")
    df.to_csv(raw_data_path, index=False)
    print(f"\nSaved raw data to {raw_data_path}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Save train/test splits
    train_df = pd.DataFrame(X_train, columns=diabetes.feature_names)
    train_df["target"] = y_train
    train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)

    test_df = pd.DataFrame(X_test, columns=diabetes.feature_names)
    test_df["target"] = y_test
    test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False)

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # --------------------- Experiment Tracking ---------------------
    print("\n4. Experiment Tracking")
    print("---------------------")

    # Create experiment for random forest model
    print("\nCreating experiment: random_forest_regression")
    rf_experiment = project.create_experiment("random_forest_regression")

    # Log experiment parameters
    print("\nLogging experiment parameters...")
    rf_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42,
    }
    rf_experiment.log_params(rf_params)

    # Train model
    print("\nTraining Random Forest model...")
    rf_model = RandomForestRegressor(**rf_params)
    rf_model.fit(X_train, y_train)

    # Get predictions
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)

    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    # Log metrics
    print("\nLogging metrics...")
    rf_experiment.log_metrics(
        {
            "train_mse": train_mse,
            "test_mse": test_mse,
            "train_r2": train_r2,
            "test_r2": test_r2,
        }
    )

    print(f"Training MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")

    # Log feature importance
    feature_importance = rf_model.feature_importances_
    importance_df = pd.DataFrame(
        {"feature": diabetes.feature_names, "importance": feature_importance}
    ).sort_values("importance", ascending=False)

    importance_path = os.path.join(project_dir, "artifacts", "feature_importance.csv")
    os.makedirs(os.path.dirname(importance_path), exist_ok=True)
    importance_df.to_csv(importance_path, index=False)

    # Log artifact
    print("\nLogging feature importance artifact...")
    rf_experiment.log_artifact("feature_importance.csv", importance_path)

    # Save model
    model_path = os.path.join(project_dir, "models", "random_forest.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    import joblib

    joblib.dump(rf_model, model_path)

    # Log model artifact
    print("\nLogging model artifact...")
    rf_experiment.log_artifact("model.pkl", model_path)

    # Complete experiment
    rf_experiment.complete()
    print("\nExperiment completed")

    # --------------------- Model Registry ---------------------
    print("\n5. Model Registry")
    print("-----------------")

    # Get model registry
    registry = project.registry

    # Register model
    print("\nRegistering model in registry...")
    model_metadata = registry.register_model(
        model_path=model_path,
        name="diabetes_predictor",
        description="Random Forest Regression model to predict diabetes progression",
        metadata={
            "framework": "scikit-learn",
            "task_type": "regression",
            "metrics": {"mse": test_mse, "r2": test_r2},
            "feature_importance": importance_df.to_dict(orient="records"),
            "data_source": "diabetes_dataset",
            "experiment_id": rf_experiment.id,
        },
    )

    print(f"Model registered with ID: {model_metadata.id}")
    print(f"Model version: {model_metadata.version}")

    # List registered models
    print("\nListing registered models:")
    for model in registry.list_models():
        print(f"  - {model.name} (version {model.version})")

    # --------------------- Deployment ---------------------
    print("\n6. Deployment")
    print("-------------")

    # Deploy model to production
    print("\nDeploying model to production environment...")
    deployment = project.deployer.deploy(
        model_name="diabetes_predictor",
        env_name="production",
        deployment_type="rest-api",
        replicas=1,
    )

    print(f"Model deployed as {deployment.name}")
    print(f"Deployment URL: {deployment.url}")
    print("Deployment status: Active")

    # --------------------- Monitoring ---------------------
    print("\n7. Monitoring")
    print("-------------")

    # Set up monitoring
    print("\nSetting up model monitoring...")
    monitoring_config = {
        "metrics": ["prediction_drift", "feature_drift", "accuracy"],
        "schedule": "hourly",
        "alerts": {"slack_channel": "#model-alerts", "email": ["team@example.com"]},
        "thresholds": {"drift_threshold": 0.1, "accuracy_drop_threshold": 0.05},
    }

    monitoring = project.setup_monitoring(
        deployment_name=deployment.name, config=monitoring_config
    )

    print("Monitoring configured and activated")

    # --------------------- Summary ---------------------
    print("\n8. Summary")
    print("----------")

    print("\nLlamaMlOps workflow completed successfully!")
    print("\nWorkflow summary:")
    print(f"  - Project: {project.name}")
    print(
        f"  - Environments: {', '.join([env.name for env in project.list_environments()])}"
    )
    print(
        f"  - Experiments: {', '.join([exp.name for exp in project.list_experiments()])}"
    )
    print(
        f"  - Registered Models: {', '.join([model.name for model in registry.list_models()])}"
    )
    print(f"  - Deployments: {deployment.name}")
    print("  - Monitoring: Active")

    print("\nProject assets are saved in:", project_dir)
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
