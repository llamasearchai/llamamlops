"""
Environment management for LlamaMlOps.

This module defines the Environment class for managing ML environments and
the EnvironmentConfig for configuring environment settings.
"""

import os
import sys
import subprocess
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

@dataclass
class EnvironmentConfig:
    """Configuration for an ML environment."""
    name: str
    description: str = ""
    
    # Python configuration
    python_version: str = "3.8"
    python_packages: List[str] = field(default_factory=list)
    
    # Environment variables
    env_variables: Dict[str, str] = field(default_factory=dict)
    
    # Docker configuration
    use_docker: bool = False
    base_image: str = "python:3.8-slim"
    docker_file: Optional[str] = None
    
    # Compute resources
    cpu_count: Optional[int] = None
    memory_limit: Optional[str] = None
    gpu_count: Optional[int] = None
    
    # Environment manager
    manager: str = "conda"  # conda, virtualenv, docker
    
    # Additional custom configuration
    params: Dict[str, Any] = field(default_factory=dict)


class Environment:
    """Environment management for ML projects."""
    
    def __init__(self, config: Union[EnvironmentConfig, Dict[str, Any]]):
        """Initialize an environment with the given configuration.
        
        Args:
            config: Environment configuration or project configuration.
        """
        if isinstance(config, dict):
            # Check if this is a ProjectConfig dict
            if "python_version" in config:
                self.config = EnvironmentConfig(
                    name=config.get("name", "default"),
                    python_version=config.get("python_version", "3.8"),
                    python_packages=config.get("dependencies", []),
                    env_variables=config.get("env_variables", {}),
                )
            else:
                self.config = EnvironmentConfig(**config)
        elif hasattr(config, "python_version"):
            # This is a ProjectConfig
            self.config = EnvironmentConfig(
                name=getattr(config, "name", "default"),
                python_version=getattr(config, "python_version", "3.8"),
                python_packages=getattr(config, "dependencies", []),
                env_variables=getattr(config, "env_variables", {}),
            )
        else:
            self.config = config
        
        self.env_path = None
    
    @property
    def name(self) -> str:
        """Get the environment name."""
        return self.config.name
    
    def initialize(self, force: bool = False) -> bool:
        """Initialize the environment.
        
        Args:
            force: If True, recreate the environment if it exists.
            
        Returns:
            True if initialization was successful, False otherwise.
        """
        manager = self.config.manager.lower()
        
        if manager == "conda":
            return self._init_conda(force)
        elif manager == "virtualenv":
            return self._init_virtualenv(force)
        elif manager == "docker":
            return self._init_docker(force)
        else:
            print(f"Unsupported environment manager: {manager}")
            return False
    
    def _init_conda(self, force: bool = False) -> bool:
        """Initialize a conda environment.
        
        Args:
            force: If True, recreate the environment if it exists.
            
        Returns:
            True if initialization was successful, False otherwise.
        """
        try:
            # Check if conda is available
            subprocess.run(["conda", "--version"], check=True, capture_output=True)
            
            env_name = self.config.name
            
            # Check if the environment exists
            result = subprocess.run(
                ["conda", "env", "list", "--json"],
                check=True,
                capture_output=True,
                text=True
            )
            import json
            envs = json.loads(result.stdout)["envs"]
            env_exists = any(env_name in path for path in envs)
            
            if env_exists and force:
                print(f"Removing existing conda environment: {env_name}")
                subprocess.run(["conda", "env", "remove", "-n", env_name, "-y"], check=True)
                env_exists = False
            
            if not env_exists:
                print(f"Creating conda environment: {env_name}")
                cmd = [
                    "conda", "create", "-n", env_name,
                    f"python={self.config.python_version}",
                    "-y"
                ]
                subprocess.run(cmd, check=True)
                
                # Install packages
                if self.config.python_packages:
                    self.install_packages(self.config.python_packages)
            
            # Set environment variables
            # Note: These are only set for the current process and won't persist
            for key, value in self.config.env_variables.items():
                os.environ[key] = value
            
            # Find the environment path
            result = subprocess.run(
                ["conda", "env", "list", "--json"],
                check=True,
                capture_output=True,
                text=True
            )
            envs = json.loads(result.stdout)["envs"]
            for path in envs:
                if env_name in path:
                    self.env_path = path
                    break
            
            print(f"Conda environment {env_name} initialized successfully.")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error initializing conda environment: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error initializing conda environment: {e}")
            return False
    
    def _init_virtualenv(self, force: bool = False) -> bool:
        """Initialize a virtualenv environment.
        
        Args:
            force: If True, recreate the environment if it exists.
            
        Returns:
            True if initialization was successful, False otherwise.
        """
        try:
            # Check if virtualenv is available
            subprocess.run(["virtualenv", "--version"], check=True, capture_output=True)
            
            env_name = self.config.name
            env_path = Path(".venv") / env_name
            self.env_path = str(env_path.absolute())
            
            # Check if the environment exists
            env_exists = env_path.exists()
            
            if env_exists and force:
                print(f"Removing existing virtualenv: {env_path}")
                shutil.rmtree(env_path)
                env_exists = False
            
            if not env_exists:
                print(f"Creating virtualenv: {env_path}")
                cmd = ["virtualenv", str(env_path), f"--python=python{self.config.python_version}"]
                subprocess.run(cmd, check=True)
                
                # Install packages
                if self.config.python_packages:
                    self.install_packages(self.config.python_packages)
            
            # Set environment variables
            # Note: These are only set for the current process and won't persist
            for key, value in self.config.env_variables.items():
                os.environ[key] = value
            
            print(f"Virtualenv {env_name} initialized successfully.")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error initializing virtualenv: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error initializing virtualenv: {e}")
            return False
    
    def _init_docker(self, force: bool = False) -> bool:
        """Initialize a Docker environment.
        
        Args:
            force: If True, rebuild the Docker image if it exists.
            
        Returns:
            True if initialization was successful, False otherwise.
        """
        try:
            # Check if Docker is available
            subprocess.run(["docker", "--version"], check=True, capture_output=True)
            
            image_name = f"llamamlops-{self.config.name.lower()}"
            
            # Check if the Dockerfile exists or create it
            if self.config.docker_file and Path(self.config.docker_file).exists():
                dockerfile_path = self.config.docker_file
            else:
                dockerfile_path = "Dockerfile"
                self._create_dockerfile(dockerfile_path)
            
            # Build the Docker image
            print(f"Building Docker image: {image_name}")
            cmd = ["docker", "build", "-t", image_name, "-f", dockerfile_path, "."]
            if force:
                cmd.append("--no-cache")
            
            subprocess.run(cmd, check=True)
            
            print(f"Docker environment {image_name} initialized successfully.")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error initializing Docker environment: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error initializing Docker environment: {e}")
            return False
    
    def _create_dockerfile(self, path: str) -> None:
        """Create a Dockerfile for the environment.
        
        Args:
            path: Path to save the Dockerfile.
        """
        with open(path, 'w') as f:
            f.write(f"FROM {self.config.base_image}\n\n")
            
            # Set environment variables
            if self.config.env_variables:
                for key, value in self.config.env_variables.items():
                    f.write(f"ENV {key}={value}\n")
                f.write("\n")
            
            # Install packages
            if self.config.python_packages:
                f.write("RUN pip install --no-cache-dir \\\n")
                for i, package in enumerate(self.config.python_packages):
                    if i == len(self.config.python_packages) - 1:
                        f.write(f"    {package}\n\n")
                    else:
                        f.write(f"    {package} \\\n")
            
            # Set working directory
            f.write("WORKDIR /app\n\n")
            
            # Copy project files
            f.write("COPY . /app/\n\n")
            
            # Default command
            f.write('CMD ["python", "-m", "llamamlops.cli"]\n')
    
    def install_packages(self, packages: List[str]) -> bool:
        """Install Python packages in the environment.
        
        Args:
            packages: List of packages to install.
            
        Returns:
            True if installation was successful, False otherwise.
        """
        try:
            manager = self.config.manager.lower()
            
            if manager == "conda":
                cmd = ["conda", "run", "-n", self.config.name, "pip", "install"]
                cmd.extend(packages)
                subprocess.run(cmd, check=True)
            
            elif manager == "virtualenv":
                if sys.platform == "win32":
                    pip_path = Path(self.env_path) / "Scripts" / "pip"
                else:
                    pip_path = Path(self.env_path) / "bin" / "pip"
                
                cmd = [str(pip_path), "install"]
                cmd.extend(packages)
                subprocess.run(cmd, check=True)
            
            elif manager == "docker":
                print("Package installation in Docker is handled during image build.")
                return True
            
            else:
                print(f"Unsupported environment manager: {manager}")
                return False
            
            print(f"Successfully installed packages: {', '.join(packages)}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error installing packages: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error installing packages: {e}")
            return False
    
    def run(self, command: List[str]) -> subprocess.CompletedProcess:
        """Run a command in the environment.
        
        Args:
            command: The command to run.
            
        Returns:
            CompletedProcess instance with return code and output.
            
        Raises:
            subprocess.CalledProcessError: If the command fails.
        """
        manager = self.config.manager.lower()
        
        if manager == "conda":
            cmd = ["conda", "run", "-n", self.config.name]
            cmd.extend(command)
            return subprocess.run(cmd, check=True)
        
        elif manager == "virtualenv":
            if sys.platform == "win32":
                python_path = Path(self.env_path) / "Scripts" / "python"
            else:
                python_path = Path(self.env_path) / "bin" / "python"
            
            cmd = [str(python_path)]
            cmd.extend(command)
            return subprocess.run(cmd, check=True)
        
        elif manager == "docker":
            image_name = f"llamamlops-{self.config.name.lower()}"
            cmd = ["docker", "run", "--rm"]
            
            # Add environment variables
            for key, value in self.config.env_variables.items():
                cmd.extend(["-e", f"{key}={value}"])
            
            # Add volume mount for current directory
            cmd.extend(["-v", f"{os.getcwd()}:/app"])
            
            # Add image name
            cmd.append(image_name)
            
            # Add command
            cmd.extend(command)
            
            return subprocess.run(cmd, check=True)
        
        else:
            raise ValueError(f"Unsupported environment manager: {manager}")
    
    def __repr__(self) -> str:
        """Get string representation of the environment."""
        return f"Environment(name={self.config.name}, manager={self.config.manager})" 