#!/usr/bin/env python3
# Copyright (C) 2024-2025 Javier Blanco-Romero @fj-blanco (UC3M, QURSA project)
#
# benchmark.py - QKD-IPSec Testing Orchestrator
"""
QKD-IPSec Testing Orchestrator

A Python script for managing QKD-IPSec tests, including environment setup,
test execution, and result analysis.
"""

import argparse
import datetime
import json
import os
import re
import socket
import subprocess
import sys
import time
import yaml
import docker
from pathlib import Path
from benchmark_utils.logs import (
    log_versions,
    log_python_packages,
    log_network_configuration,
)

# Default configuration values
DEFAULT_CONFIG_FILE = "config/shared/benchmark_config.yml"
DEFAULT_ALICE_IP = "172.30.0.3"
DEFAULT_BOB_IP = "172.30.0.2"


class QKDTestOrchestrator:
    """Manages QKD-IPSec test execution and environment setup."""

    def __init__(self, config_file=DEFAULT_CONFIG_FILE, cli_args=None):
        """Initialize the test orchestrator with configuration."""
        self.config_file = config_file
        self.config = self._load_config()
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Validate ETSI API version and backend compatibility
        self._validate_etsi_backend_compatibility()

        # Extract network mode and IPs from new config format
        network_config = self.config.get("docker", {}).get("network", {})
        self.network_mode = network_config.get("mode", "single-machine")

        # Extract Alice and Bob configuration
        alice_config = network_config.get("alice", {})
        bob_config = network_config.get("bob", {})

        # Set IP addresses from new config structure
        self.alice_ip = alice_config.get("container_ip", DEFAULT_ALICE_IP)
        self.bob_ip = bob_config.get("container_ip", DEFAULT_BOB_IP)
        self.alice_local_ip = alice_config.get("local_ip", "auto")
        self.bob_local_ip = bob_config.get("local_ip", "auto")
        self.alice_qkd_port = alice_config.get("qkd_server_port", 25575)
        self.bob_qkd_port = bob_config.get("qkd_server_port", 25576)
        self.coordination_port = network_config.get("coordination_port", 8080)

        # Initialize role-specific attributes
        self.role = getattr(cli_args, "role", "both")
        self.remote_ip = getattr(cli_args, "remote_ip", None)
        self.local_ip = getattr(cli_args, "local_ip", None)

        # If CLI args don't specify local_ip, use config values
        if not self.local_ip:
            if self.role == "alice":
                self.local_ip = self.alice_local_ip
            elif self.role == "bob":
                self.local_ip = self.bob_local_ip

        # Get build settings with defaults
        self.build = self.config.get("docker", {}).get("build", {}).get("build", False)
        self.use_cache = (
            self.config.get("docker", {}).get("build", {}).get("use_cache", True)
        )

        # Override config with CLI args if provided
        if cli_args:
            self._apply_cli_overrides(cli_args)

        # Validate distributed mode configuration
        self._validate_distributed_config()

        # Auto-detect local IP if needed
        if self.role in ["alice", "bob"] and (
            not self.local_ip or self.local_ip == "auto"
        ):
            self.local_ip = self._get_local_ip()

        # Select appropriate compose file based on mode and role
        self._select_compose_file()

        # Prepare output directories
        self.dirs = self._setup_directories()

        # Initialize Docker client
        self.docker_client = docker.from_env()

        # Track Pumba containers
        self.pumba_containers = []

    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_file, "r") as f:
                config = yaml.safe_load(f)
                return config
        except Exception as e:
            print(f"Error loading configuration: {e}")
            sys.exit(1)

    def _validate_etsi_backend_compatibility(self):
        """Validate that ETSI API version and QKD backend are compatible."""
        if "docker" not in self.config or "qkd" not in self.config["docker"]:
            print("Warning: QKD configuration missing, using defaults")
            return

        etsi_version = self.config["docker"]["qkd"].get("etsi_api_version")
        backend = self.config["docker"]["qkd"].get("backend")
        initiation_mode = self.config["docker"]["qkd"].get("initiation_mode", "client")

        if not etsi_version or not backend:
            print("Warning: ETSI API version or QKD backend not specified")
            return

        # Define the allowed backends for each ETSI version
        allowed_backends = {
            "004": ["python_client"],
            "014": ["qukaydee", "cerberis-xgr", "simulated"],
        }

        if etsi_version not in allowed_backends:
            print(
                f"Warning: Unknown ETSI API version: {etsi_version}. "
                f"Supported versions are: {', '.join(allowed_backends.keys())}"
            )
            return

        if backend not in allowed_backends[etsi_version]:
            print(
                f"Error: Incompatible configuration - ETSI API version '{etsi_version}' "
                f"must be used with one of these backends: {', '.join(allowed_backends[etsi_version])}"
            )
            print(f"Current backend is '{backend}', automatically adjusting...")

            sys.exit(1)

        if etsi_version == "004" and initiation_mode == "server":
            print(
                "Error: Server-initiated QKD mode is not compatible with ETSI 004 API"
            )
            print(
                "Use ETSI 014 API with server-initiated mode, or use client-initiated mode with ETSI 004"
            )
            sys.exit(1)

    def _validate_distributed_config(self):
        """Validate distributed mode configuration."""
        # Automatically switch to distributed mode when role is specified
        if self.role in ["alice", "bob"]:
            print(
                f"Role '{self.role}' specified - automatically switching to distributed mode"
            )
            self.network_mode = "distributed"

        # If network mode is distributed or role is specified, validate
        if self.network_mode == "distributed" or self.role in ["alice", "bob"]:
            if self.role in ["alice", "bob"] and not self.remote_ip:
                print("Error: --remote-ip is required when running in distributed mode")
                sys.exit(1)

    def _apply_cli_overrides(self, args):
        """Apply command line argument overrides to configuration."""
        etsi_version_changed = False
        backend_changed = False

        if hasattr(args, "build") and args.build:
            print("CLI --build flag provided, overriding YAML build setting")
            self.build = True

        if hasattr(args, "no_cache") and args.no_cache:
            print("CLI --no-cache flag provided, overriding YAML use_cache setting")
            self.use_cache = False

        # ETSI version and backend overrides (if provided)
        if args.etsi_version is not None:
            self.config["docker"]["qkd"]["etsi_api_version"] = args.etsi_version
            etsi_version_changed = True

        if args.qkd_backend is not None:
            self.config["docker"]["qkd"]["backend"] = args.qkd_backend
            backend_changed = True

        if args.qkd_initiation_mode is not None:
            self.config["docker"]["qkd"]["initiation_mode"] = args.qkd_initiation_mode

        # If either changed, but not both, validate compatibility
        if etsi_version_changed or backend_changed:
            self._validate_etsi_backend_compatibility()

        # Test parameters
        if args.iterations is not None:
            self.config["test"]["iterations"] = args.iterations

        if args.no_network_conditions:
            self.config["test"]["network"]["apply"] = False

        if args.latency is not None:
            self.config["test"]["network"]["latency"] = args.latency

        if args.jitter is not None:
            self.config["test"]["network"]["jitter"] = args.jitter

        if args.packet_loss is not None:
            self.config["test"]["network"]["packet_loss"] = args.packet_loss

        if hasattr(args, "duration") and args.duration is not None:
            self.config["test"]["network"]["duration"] = args.duration

        if args.no_analyze:
            self.config["test"]["analyze_results"] = False

        # IP address overrides - update the nested structure
        if args.alice_ip is not None:
            if "docker" not in self.config:
                self.config["docker"] = {}
            if "network" not in self.config["docker"]:
                self.config["docker"]["network"] = {}
            if "alice" not in self.config["docker"]["network"]:
                self.config["docker"]["network"]["alice"] = {}

            self.config["docker"]["network"]["alice"]["container_ip"] = args.alice_ip
            self.alice_ip = args.alice_ip

        if args.bob_ip is not None:
            if "docker" not in self.config:
                self.config["docker"] = {}
            if "network" not in self.config["docker"]:
                self.config["docker"]["network"] = {}
            if "bob" not in self.config["docker"]["network"]:
                self.config["docker"]["network"]["bob"] = {}

            self.config["docker"]["network"]["bob"]["container_ip"] = args.bob_ip
            self.bob_ip = args.bob_ip

    def _setup_directories(self):
        """Set up the directory structure for test outputs and backup configuration files."""
        # Extract config values
        etsi_version = self.config["docker"]["qkd"]["etsi_api_version"]
        qkd_backend = self.config["docker"]["qkd"]["backend"]
        initiation_mode = self.config["docker"]["qkd"].get("initiation_mode", "client")
        iterations = self.config["test"]["iterations"]

        path_components = [etsi_version, qkd_backend, initiation_mode + "_init"]
        # Create directory path based on network conditions
        if self.config["test"]["network"]["apply"]:
            network_config = self.config["test"]["network"]
            network_part = f"lat{network_config['latency']}_jit{network_config['jitter']}_loss{network_config['packet_loss']}"
            path_components.append(network_part)
        else:
            path_components.append("no_network_conditions")

        path_components.extend([f"iter{iterations}", f"time_{self.timestamp}"])

        # Join all components
        relative_dir = "/".join(path_components)

        # Define directory paths
        dirs = {
            "output_dir": f"./results/{relative_dir}",
            "docker_output_dir": f"/output/{relative_dir}",
            "analysis_dir": f"./analysis/{relative_dir}",
        }

        # Create directories
        Path(dirs["output_dir"]).mkdir(parents=True, exist_ok=True)

        # Create test configuration file
        self._create_test_config_json(dirs["output_dir"])

        # Backup configuration files to results directory
        self._backup_config_files(dirs["output_dir"])

        return dirs

    def _select_compose_file(self):
        """Select the appropriate Docker Compose file based on mode and role."""
        compose_files = self.config.get("docker", {}).get("compose_files", {})

        if self.network_mode == "distributed" or self.role in ["alice", "bob"]:
            # Use distributed compose file
            self.compose_file = compose_files.get(
                "distributed", "docker-compose.004.distr.yml"
            )
        else:
            # Use single machine compose file
            self.compose_file = compose_files.get(
                "single_machine", "docker-compose.004.yml"
            )

        print(
            f"Using compose file: {self.compose_file} for mode: {self.network_mode}, role: {self.role}"
        )

    def _backup_config_files(self, output_dir):
        """Backup configuration files to the results directory."""
        try:
            # Copy benchmark config file
            with open(self.config_file, "r") as src_file:
                benchmark_config_content = src_file.read()

            with open(f"{output_dir}/benchmark_config.yml", "w") as dest_file:
                dest_file.write(benchmark_config_content)

            # Copy proposals config file (assuming standard location)
            proposals_config = self.config.get("proposals", {}).get(
                "file", "config/shared/proposals_config.yml"
            )
            if os.path.exists(proposals_config):
                with open(proposals_config, "r") as src_file:
                    proposals_config_content = src_file.read()

                with open(f"{output_dir}/proposals_config.yml", "w") as dest_file:
                    dest_file.write(proposals_config_content)
            else:
                print(
                    f"Warning: Proposals config file {proposals_config} not found, couldn't create backup"
                )

            print(f"Configuration files backed up to {output_dir}")
        except Exception as e:
            print(f"Warning: Could not backup configuration files: {e}")

    def _create_test_config_json(self, output_dir):
        """Create a JSON file with test configuration details."""
        etsi_version = self.config["docker"]["qkd"]["etsi_api_version"]
        qkd_backend = self.config["docker"]["qkd"]["backend"]
        initiation_mode = self.config["docker"]["qkd"]["initiation_mode"]
        iterations = self.config["test"]["iterations"]
        network_config = self.config["test"]["network"]

        config_json = {
            "timestamp": self.timestamp,
            "etsi_api_version": etsi_version,
            "qkd_backend": qkd_backend,
            "initiation_mode": initiation_mode,
            "iterations": iterations,
            "network_conditions": {
                "applied": network_config["apply"],
                "latency_ms": network_config["latency"],
                "jitter_ms": network_config["jitter"],
                "packet_loss_percent": network_config["packet_loss"],
                "duration": network_config["duration"],
            },
            "network_mode": self.network_mode,
            "container_network": {
                "alice": {
                    "container_ip": self.alice_ip,
                    "local_ip": self.alice_local_ip,
                    "qkd_server_port": self.alice_qkd_port,
                },
                "bob": {
                    "container_ip": self.bob_ip,
                    "local_ip": self.bob_local_ip,
                    "qkd_server_port": self.bob_qkd_port,
                },
                "coordination_port": self.coordination_port,
            },
            "runtime_config": {
                "role": self.role,
                "remote_ip": self.remote_ip,
                "local_ip": self.local_ip,
            },
        }

        with open(f"{output_dir}/test_config.json", "w") as f:
            json.dump(config_json, f, indent=2)

    def _get_container_status(self):
        """Check the status of containers based on role."""
        try:
            containers_to_check = []

            if self.role in ["alice", "both"]:
                containers_to_check.append("alice")
            if self.role in ["bob", "both"]:
                containers_to_check.append("bob")

            running_containers = []
            existing_containers = []

            for container_name in containers_to_check:
                try:
                    container = self.docker_client.containers.get(container_name)
                    existing_containers.append(container)
                    if container.status == "running":
                        running_containers.append(container)
                except docker.errors.NotFound:
                    return "not_found", None

            if len(running_containers) == len(containers_to_check):
                return "running", tuple(existing_containers)
            elif len(existing_containers) == len(containers_to_check):
                return "exists", tuple(existing_containers)
            else:
                return "not_found", None

        except Exception:
            return "not_found", None

    def _start_existing_containers(self, containers):
        """Start existing containers that are not running."""
        try:
            print("Containers exist but are not running. Starting containers...")

            for container in containers:
                if container.status != "running":
                    print(f"Starting {container.name}...")
                    container.start()

            print("Waiting for containers to start...")
            time.sleep(5)

            # Verify containers are running now
            all_running = True
            for container in containers:
                container.reload()
                if container.status != "running":
                    print(
                        f"Failed to start {container.name}. Status: {container.status}"
                    )
                    all_running = False

            if all_running:
                print("Containers started successfully")
                return True
            else:
                return False

        except Exception as e:
            print(f"Error starting existing containers: {e}")
            return False

    def _run_docker_command(self, cmd, env=None, check=True, capture_output=False):
        """Run a Docker command with appropriate environment variables."""
        full_env = {**os.environ}
        if env:
            full_env.update(env)

        try:
            result = subprocess.run(
                cmd,
                env=full_env,
                check=check,
                capture_output=capture_output,
                text=capture_output,
            )
            return result
        except subprocess.CalledProcessError as e:
            if not check:
                return e
            raise

    def check_containers(self):
        """Check if Docker containers are running and start them if needed."""
        try:
            # Check if containers exist and are running
            status, containers = self._get_container_status()

            if status == "running":
                print(f"Docker containers are running for role: {self.role}")
                return True

            elif status == "exists":
                return self._start_existing_containers(containers)

            elif status == "not_found":
                return self._start_or_build_containers()

        except Exception as e:
            print(f"Error checking containers: {e}")
            self._show_manual_startup_instructions()
            return False

    def _start_or_build_containers(self):
        """Try to start containers or build them if needed."""
        try:
            # Try docker-compose up to start existing containers
            compose_file = self.compose_file  # Use the selected compose file
            env = self._get_docker_env()

            print("Running docker-compose up to see if containers can be started...")
            up_cmd = ["docker-compose", "-f", compose_file, "up", "-d"]
            self._run_docker_command(up_cmd, env=env)
            print("Containers started. Waiting for them to initialize...")
            time.sleep(5)

            # Verify containers are running
            status, _ = self._get_container_status()
            if status == "running":
                return True

            print("Containers not running after docker-compose up. Trying to build...")
            # Try building and starting containers
            build_success = self.setup_environment(build_only=False, detached=True)
            if not build_success:
                raise Exception("Both container start and build attempts failed")

            print("Waiting for containers to start...")
            time.sleep(1)
            return True
        except Exception as up_error:
            print(f"Starting containers failed: {up_error}")
            print("Now trying to build and start containers...")

            # Try building and starting containers
            build_success = self.setup_environment(build_only=False, detached=True)
            if not build_success:
                raise Exception("Both container start and build attempts failed")

            print("Waiting for containers to start...")
            time.sleep(1)
            return True

    def _show_manual_startup_instructions(self):
        """Show instructions for manually starting containers."""
        print("\nPlease start the Docker containers manually using:")
        print(
            f"  ETSI_API_VERSION={self.config['docker']['qkd']['etsi_api_version']} \\"
        )
        print(f"  QKD_BACKEND={self.config['docker']['qkd']['backend']} \\")
        if self.config["docker"]["qkd"]["account_id"]:
            print(f"  ACCOUNT_ID={self.config['docker']['qkd']['account_id']} \\")
        print(f"  docker-compose -f {self.compose_file} up -d")

        print("\nThen run this script again after containers are running.")

    def _get_docker_env(self):
        """Get environment variables for Docker commands."""
        base_env = {
            "ETSI_API_VERSION": self.config["docker"]["qkd"]["etsi_api_version"],
            "QKD_BACKEND": self.config["docker"]["qkd"]["backend"],
            "QKD_INITIATION_MODE": self.config["docker"]["qkd"].get(
                "initiation_mode", "client"
            ),
            "ACCOUNT_ID": self.config["docker"]["qkd"]["account_id"],
            "STRONGSWAN_VERSION": self.config["docker"]["build"]["strongswan_version"],
            "BUILD_QKD_ETSI": str(
                self.config["docker"]["build"]["build_qkd_etsi"]
            ).lower(),
            "BUILD_QKD_KEM": str(
                self.config["docker"]["build"]["build_qkd_kem"]
            ).lower(),
            "BENCHMARK_ROLE": self.role,
            "COORDINATION_PORT": str(self.coordination_port),
            # Always provide these IPs for consistent networking
            "ALICE_IP": self.alice_ip,
            "BOB_IP": self.bob_ip,
            "ALICE_QKD_IP": "172.30.0.10",
            "BOB_QKD_IP": "172.30.0.11",
            "ALICE_KEYGEN_IP": "172.30.0.12",
            "BOB_KEYGEN_IP": "172.30.0.13",
            "BOB_INTRANET_IP": "172.31.0.2",
        }

        # Configure based on role and network mode
        if self.role == "alice":
            base_env.update(
                {
                    "NETWORK_MODE": (
                        "host" if self.network_mode == "distributed" else "bridge"
                    ),
                    "REMOTE_BOB_IP": self.remote_ip or self.bob_ip,
                    "LOCAL_IP": self.local_ip or "auto",
                    # Use default ports in host mode, different ports in bridge mode
                    "ALICE_IKE_PORT": (
                        "500" if self.network_mode == "distributed" else "501"
                    ),
                    "ALICE_NATT_PORT": (
                        "4500" if self.network_mode == "distributed" else "4501"
                    ),
                    "ALICE_COORD_PORT": str(self.coordination_port),
                    "ALICE_QKD_PORT": str(self.alice_qkd_port),
                    "BOB_KEYGEN_PORT": "5000",
                    # Bob settings for single machine mode
                    "BOB_IKE_PORT": "500",
                    "BOB_NATT_PORT": "4500",
                    "BOB_COORD_PORT": str(
                        self.coordination_port
                        if self.network_mode == "distributed"
                        else self.coordination_port
                    ),
                    "BOB_QKD_PORT": str(self.bob_qkd_port),
                }
            )
        elif self.role == "bob":
            base_env.update(
                {
                    "NETWORK_MODE": (
                        "host" if self.network_mode == "distributed" else "bridge"
                    ),
                    "REMOTE_ALICE_IP": self.remote_ip or self.alice_ip,
                    "LOCAL_IP": self.local_ip or "auto",
                    # Use default ports in host mode
                    "BOB_IKE_PORT": "500",
                    "BOB_NATT_PORT": "4500",
                    "BOB_COORD_PORT": str(self.coordination_port),
                    "BOB_QKD_PORT": str(self.bob_qkd_port),
                    "BOB_KEYGEN_PORT": "5000",
                    # Alice settings for cross-reference
                    "ALICE_IKE_PORT": "501",
                    "ALICE_NATT_PORT": "4501",
                    "ALICE_COORD_PORT": str(self.coordination_port + 1),
                    "ALICE_QKD_PORT": str(self.alice_qkd_port),
                }
            )
        else:  # both (single machine)
            base_env.update(
                {
                    "NETWORK_MODE": "bridge",  # Always use bridge for single machine
                    # Use different ports to avoid conflicts in single machine
                    "ALICE_IKE_PORT": "501",
                    "ALICE_NATT_PORT": "4501",
                    "ALICE_COORD_PORT": str(self.coordination_port + 1),  # 8081
                    "BOB_IKE_PORT": "500",
                    "BOB_NATT_PORT": "4500",
                    "BOB_COORD_PORT": str(self.coordination_port),  # 8080
                    "ALICE_QKD_PORT": str(self.alice_qkd_port),
                    "BOB_QKD_PORT": str(self.bob_qkd_port),
                    "BOB_KEYGEN_PORT": "5000",
                    # Set remote IPs to container IPs for single machine
                    "REMOTE_ALICE_IP": self.alice_ip,
                    "REMOTE_BOB_IP": self.bob_ip,
                }
            )

        return base_env

    def setup_environment(self, build_only=False, detached=True):
        """Build and start Docker containers."""
        compose_file = self.compose_file  # Use the selected compose file

        # Clean up existing containers
        self._cleanup_existing_containers(compose_file)

        # Get environment variables
        env = self._get_docker_env()

        # Display build configuration
        self._display_build_config()

        # Build containers
        if not self._build_containers(compose_file, env):
            return False

        # Start containers if not build-only
        if not build_only:
            return self._start_containers(compose_file, env, detached)
        else:
            print("Build completed. Containers not started (--build-only specified).")
            return True

    def _cleanup_existing_containers(self, compose_file):
        """Clean up existing containers based on role."""
        print("Cleaning up existing containers...")
        try:
            if self.role == "alice":
                # Only stop Alice-related containers
                containers_to_stop = ["alice", "qkd_server_alice", "generate_key_alice"]
                for container in containers_to_stop:
                    try:
                        subprocess.run(
                            ["docker", "stop", container],
                            check=False,
                            capture_output=True,
                        )
                        subprocess.run(
                            ["docker", "rm", container],
                            check=False,
                            capture_output=True,
                        )
                        print(f"Stopped and removed {container}")
                    except:
                        pass
            elif self.role == "bob":
                # Only stop Bob-related containers
                containers_to_stop = ["bob", "qkd_server_bob", "generate_key_bob"]
                for container in containers_to_stop:
                    try:
                        subprocess.run(
                            ["docker", "stop", container],
                            check=False,
                            capture_output=True,
                        )
                        subprocess.run(
                            ["docker", "rm", container],
                            check=False,
                            capture_output=True,
                        )
                        print(f"Stopped and removed {container}")
                    except:
                        pass
            else:
                # For "both", clean up everything
                cleanup_cmd = [
                    "docker-compose",
                    "-f",
                    compose_file,
                    "down",
                    "--volumes",
                    "--remove-orphans",
                ]
                self._run_docker_command(cleanup_cmd, check=False)
        except Exception as e:
            print(f"Warning during cleanup: {e}")

    def _display_build_config(self):
        """Display the current build configuration."""
        etsi_version = self.config["docker"]["qkd"]["etsi_api_version"]
        qkd_backend = self.config["docker"]["qkd"]["backend"]
        initiation_mode = self.config["docker"]["qkd"].get("initiation_mode", "client")
        account_id = self.config["docker"]["qkd"]["account_id"]
        strongswan_version = self.config["docker"]["build"]["strongswan_version"]
        build_qkd_etsi = self.config["docker"]["build"]["build_qkd_etsi"]
        build_qkd_kem = self.config["docker"]["build"]["build_qkd_kem"]

        print("Building with configuration:")
        print(f"  - ETSI API Version: {etsi_version}")
        qkd_service = (
            f"{qkd_backend} (Account: {account_id})" if account_id else qkd_backend
        )
        print(f"  - QKD Backend: {qkd_service}")
        print(f"  - QKD Initiation Mode: {initiation_mode}")
        print(f"  - StrongSwan Version: {strongswan_version}")
        print(f"  - Build QKD ETSI API: {build_qkd_etsi}")
        print(f"  - Build QKD-KEM Provider: {build_qkd_kem}")
        print(f"  - Alice IP: {self.alice_ip}")
        print(f"  - Bob IP: {self.bob_ip}")
        print("")

    def _build_containers(self, compose_file, env):
        """Build Docker containers."""
        print("Building containers...")
        build_cmd = ["docker-compose", "-f", compose_file, "build"]
        if not self.use_cache:
            build_cmd.append("--no-cache")

        try:
            self._run_docker_command(build_cmd, env=env)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Build error: {e}")
            return False

    def _start_containers(self, compose_file, env, detached):
        """Start Docker containers with role-specific profiles."""
        print("Starting containers...")

        # Base command
        up_cmd = ["docker-compose", "-f", compose_file, "up", "-d"]

        # Add profile-specific services based on role
        if self.role == "alice":
            up_cmd.extend(
                ["strongswan-base", "alice", "qkd_server_alice", "generate_key_alice"]
            )
            print("Starting Alice-related containers only...")
        elif self.role == "bob":
            up_cmd.extend(
                ["strongswan-base", "bob", "qkd_server_bob", "generate_key_bob"]
            )
            print("Starting Bob-related containers only...")
        else:  # both
            print("Starting all containers...")
            # For 'both', start all services (default behavior)

        try:
            self._run_docker_command(up_cmd, env=env)
            print(f"Containers started for role: {self.role}")

            # Verify that containers are running
            self._verify_containers_running(compose_file, env)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error starting containers: {e}")
            return False

    def _verify_containers_running(self, compose_file, env):
        """Verify that containers are running based on role."""
        ps_cmd = ["docker-compose", "-f", compose_file, "ps"]
        result = self._run_docker_command(ps_cmd, env=env, capture_output=True)

        # Check for role-specific containers
        if self.role == "alice":
            if "alice" not in result.stdout:
                print("Warning: Alice container may not be running correctly.")
        elif self.role == "bob":
            if "bob" not in result.stdout:
                print("Warning: Bob container may not be running correctly.")
        else:  # both
            if "alice" not in result.stdout or "bob" not in result.stdout:
                print("Warning: Containers may not be running correctly.")

    def _cleanup_pumba_containers(self):
        """Stop all Pumba containers."""
        if not self.pumba_containers:
            return

        print("Stopping Pumba containers...")
        stopped_count = 0

        for container_id in self.pumba_containers:
            try:
                # Stop the container
                subprocess.run(
                    ["docker", "stop", container_id],
                    check=True,
                    capture_output=True,
                    timeout=10,
                )
                stopped_count += 1
                print(f"Stopped Pumba container: {container_id[:12]}")
            except subprocess.CalledProcessError as e:
                print(f"Warning: Could not stop container {container_id[:12]}: {e}")
            except subprocess.TimeoutExpired:
                print(f"Warning: Timeout stopping container {container_id[:12]}")
                # Force kill if stop times out
                try:
                    subprocess.run(
                        ["docker", "kill", container_id],
                        check=True,
                        capture_output=True,
                        timeout=5,
                    )
                    print(f"Force killed container: {container_id[:12]}")
                    stopped_count += 1
                except:
                    print(
                        f"Warning: Could not force kill container {container_id[:12]}"
                    )

        print(f"Stopped {stopped_count}/{len(self.pumba_containers)} Pumba containers")
        self.pumba_containers.clear()

    def _cleanup_all_pumba_containers(self):
        """Clean up any lingering Pumba containers from previous runs."""
        try:
            # Find all running Pumba containers
            result = subprocess.run(
                ["docker", "ps", "-q", "--filter", "ancestor=gaiaadm/pumba"],
                capture_output=True,
                text=True,
                check=True,
            )

            if result.stdout.strip():
                container_ids = result.stdout.strip().split("\n")
                print(
                    f"Found {len(container_ids)} existing Pumba containers to clean up"
                )

                for container_id in container_ids:
                    try:
                        subprocess.run(
                            ["docker", "stop", container_id],
                            check=True,
                            capture_output=True,
                            timeout=10,
                        )
                        print(
                            f"Cleaned up existing Pumba container: {container_id[:12]}"
                        )
                    except:
                        print(
                            f"Warning: Could not clean up container {container_id[:12]}"
                        )

        except subprocess.CalledProcessError:
            # No existing Pumba containers found
            pass

    def _get_local_ip(self) -> str:
        """Auto-detect local IP address."""
        try:
            # Connect to a remote address to determine the local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"

    def _discover_interface(self, container_name, target_ip):
        """Discover which network interface a container uses to reach a target IP."""
        try:
            result = self._execute_container_command(
                container_name, f"ip route get {target_ip}"
            )
            output = result.output.decode().strip()

            match = re.search(r"dev\s+(\S+)", output)
            if match:
                return match.group(1)
            else:
                print(f"Warning: Could not parse interface from route output: {output}")
                return None

        except Exception as e:
            print(f"Error discovering interface for {container_name}: {e}")
            return None

    def apply_network_conditions(self):
        """Apply network conditions using Pumba with unified logic for both single and distributed modes."""
        if not self.config["test"]["network"]["apply"]:
            print("Network conditions disabled. Skipping...")
            return

        # Extract network configuration
        latency = self.config["test"]["network"]["latency"]
        jitter = self.config["test"]["network"]["jitter"]
        packet_loss = self.config["test"]["network"]["packet_loss"]
        duration = self.config["test"]["network"]["duration"]

        print("Applying network conditions for testing...")
        print(f"  - Latency: {latency}ms with {jitter}ms jitter")
        print(f"  - Packet loss: {packet_loss}%")
        print(f"  - Duration: {duration}")
        print(f"  - Network mode: {self.network_mode}")

        # Clear any existing Pumba containers from previous runs
        self._cleanup_pumba_containers()

        # Determine which containers and targets to apply conditions to
        containers_and_targets = []

        if self.network_mode == "distributed":
            # Distributed mode: only apply to local container targeting remote IP
            if self.role == "alice":
                containers_and_targets = [("alice", self.remote_ip)]
                print(
                    f"  - Alice applying conditions targeting Bob at {self.remote_ip}"
                )
            elif self.role == "bob":
                containers_and_targets = [("bob", self.remote_ip)]
                print(
                    f"  - Bob applying conditions targeting Alice at {self.remote_ip}"
                )
            else:
                print(
                    "Warning: Distributed network conditions require --role alice or --role bob"
                )
                return
        else:
            # Single machine mode: apply to both containers targeting each other
            containers_and_targets = [("alice", self.bob_ip), ("bob", self.alice_ip)]
            print(
                f"  - Single machine mode: Alice targeting {self.bob_ip}, Bob targeting {self.alice_ip}"
            )

        # Apply network conditions to each container-target pair
        pumba_base = [
            "docker",
            "run",
            "-d",
            "--rm",
            "-v",
            "/var/run/docker.sock:/var/run/docker.sock",
            "gaiaadm/pumba",
            "netem",
            "--tc-image",
            "ghcr.io/alexei-led/pumba-alpine-nettools:latest",
            "--duration",
            duration,
        ]

        try:
            for container_name, target_ip in containers_and_targets:
                # Discover the correct network interface
                interface = self._discover_interface(container_name, target_ip)
                if not interface:
                    print(
                        f"ERROR: Could not determine network interface for {container_name} to reach {target_ip}!"
                    )
                    raise RuntimeError("Interface discovery failed")

                print(
                    f"  - {container_name} communicates with {target_ip} via interface: {interface}"
                )

                # Apply latency if specified
                if latency > 0:
                    latency_cmd = pumba_base + [
                        "--interface",
                        interface,
                        "--target",
                        target_ip,
                        "delay",
                        "--time",
                        str(latency),
                        "--jitter",
                        str(jitter),
                        "--distribution",
                        "normal",
                        container_name,
                    ]
                    result = subprocess.run(
                        latency_cmd, check=True, capture_output=True, text=True
                    )
                    container_id = result.stdout.strip()
                    if container_id:
                        self.pumba_containers.append(container_id)
                        print(
                            f"Started Pumba container for {container_name} latency: {container_id[:12]}"
                        )

                # Apply packet loss if specified
                if packet_loss > 0:
                    loss_cmd = pumba_base + [
                        "--interface",
                        interface,
                        "--target",
                        target_ip,
                        "loss",
                        "--percent",
                        str(packet_loss),
                        "--correlation",
                        "20",
                        container_name,
                    ]
                    result = subprocess.run(
                        loss_cmd, check=True, capture_output=True, text=True
                    )
                    container_id = result.stdout.strip()
                    if container_id:
                        self.pumba_containers.append(container_id)
                        print(
                            f"Started Pumba container for {container_name} packet loss: {container_id[:12]}"
                        )

            print(f"Total Pumba containers started: {len(self.pumba_containers)}")
            if self.pumba_containers:
                print("Waiting for network conditions to be applied...")
                time.sleep(5)
            else:
                print("No network conditions applied - no Pumba containers started")

        except subprocess.CalledProcessError as e:
            print(f"Error applying network conditions: {e}")
            print(f"Error output: {e.stderr if e.stderr else 'No stderr'}")
            self._cleanup_pumba_containers()
            raise

    def _execute_container_command(
        self, container_name, command, detach=False, stream=False
    ):
        """Execute a command on a container."""
        try:
            container = self.docker_client.containers.get(container_name)
            return container.exec_run(command, detach=detach, stream=stream)
        except docker.errors.NotFound:
            print(f"Error: {container_name} container not found.")
            sys.exit(1)

    def run_tests(self):
        """Run the test suite with proper Pumba cleanup."""
        try:
            # Check if containers are running, start them if needed
            if not self.check_containers():
                print("Cannot proceed without running containers.")
                sys.exit(1)

            # Create docker output directory and set permissions
            self._prepare_output_directory()

            # Apply network conditions if enabled
            if self.config["test"]["network"]["apply"]:
                self.apply_network_conditions()

            # Run the actual tests
            self._run_test_scripts()

        finally:
            # Always clean up Pumba containers when tests finish
            if self.config["test"]["network"]["apply"]:
                print("Cleaning up network conditions...")
                self._cleanup_pumba_containers()

    def _prepare_output_directory(self):
        """Create and prepare the output directory in the appropriate Docker container."""
        print(f"Creating output directory: {self.dirs['docker_output_dir']}")

        # Determine which container to use for directory creation
        if self.role == "alice" or self.role == "both":
            container_name = "alice"
        else:  # bob
            container_name = "bob"

        # Create directory and set permissions in the chosen container
        self._execute_container_command(
            container_name, f"mkdir -p {self.dirs['docker_output_dir']}"
        )
        self._execute_container_command(
            container_name, f"chmod -R 777 {self.dirs['docker_output_dir']}"
        )

    def _run_test_scripts(self):
        """Run the test scripts based on role with proper coordination."""
        iterations = self.config["test"]["iterations"]
        print(f"Running tests with {iterations} iterations...")
        print(f"Results will be stored in {self.dirs['output_dir']}...")

        if self.role == "both":
            # Single machine mode - run both Alice and Bob
            print("Starting Bob's test script in the background...")
            bob_cmd = (
                f"bash -c 'export IS_TLS_SERVER=1 && source /set_env.sh && "
                f"python3 /etc/swanctl/bob_tests.py > {self.dirs['docker_output_dir']}/bob_log.txt 2>&1'"
            )
            self._execute_container_command("bob", bob_cmd, detach=True)

            print("Running Alice's test script...")
            alice_cmd = (
                f"bash -c 'source /set_env.sh && "
                f"python3 /etc/swanctl/alice_tests.py --iterations {iterations} "
                f"--output-dir {self.dirs['docker_output_dir']}'"
            )
            alice_result = self._execute_container_command(
                "alice", alice_cmd, stream=True
            )
            for output in alice_result.output:
                print(output.decode().strip())

        elif self.role == "alice":
            # Distributed mode - Alice only
            print("Running Alice's test script in distributed mode...")
            print(f"Alice will coordinate with Bob at {self.remote_ip}")

            # Alice uses a different coordination port (8081) to avoid conflicts
            alice_coord_port = self.coordination_port + 1
            print(f"Alice using coordination port: {alice_coord_port}")
            print(f"Bob should be using coordination port: {self.coordination_port}")

            alice_cmd = (
                f"bash -c 'source /set_env.sh && "
                f"python3 /etc/swanctl/alice_tests.py --iterations {iterations} "
                f"--output-dir {self.dirs['docker_output_dir']} "
                f"--remote-ip {self.remote_ip} "
                f"--coordination-port {alice_coord_port} "
                f"--distributed'"
            )
            alice_result = self._execute_container_command(
                "alice", alice_cmd, stream=True
            )
            for output in alice_result.output:
                print(output.decode().strip())

        elif self.role == "bob":
            # Distributed mode - Bob coordinates with Alice
            print(f"Running Bob's test script in distributed mode...")
            print(f"Bob will coordinate with Alice at {self.remote_ip}")
            print(f"Bob using coordination port: {self.coordination_port}")

            # Bob needs to signal Alice on Alice's coordination port (8081)
            alice_coord_port = self.coordination_port + 1
            print(f"Bob will signal Alice on her coordination port: {alice_coord_port}")
            print("")
            print("Bob will:")
            print("1. Start coordination server on port 8080")
            print("2. Signal Alice on her port 8081 that he's ready")
            print("3. Wait for Alice to start test server")
            print("4. Connect and run synchronized tests")
            print("")

            # REMOVE OUTPUT REDIRECTION TO SEE BOB'S ACTUAL OUTPUT
            bob_cmd = (
                f"bash -c 'export IS_TLS_SERVER=1 && source /set_env.sh && "
                f"python3 /etc/swanctl/bob_tests.py "
                f"--alice-ip {self.remote_ip} "
                f"--coordination-port {self.coordination_port} "
                f"--alice-coordination-port {alice_coord_port} "
                f"--distributed'"
            )

            print("Starting Bob - he will coordinate with Alice...")
            print(f"DEBUG: Bob command: {bob_cmd}")
            bob_result = self._execute_container_command("bob", bob_cmd, stream=True)

            print("Bob's output:")
            for output in bob_result.output:
                print(output.decode().strip())

        print(f"Tests completed. Results accessible in {self.dirs['output_dir']}/")

    def analyze_results(self):
        """Analyze the test results using the updated analysis script."""
        if not self.config["test"]["analyze_results"]:
            print("Analysis disabled. Skipping...")
            return

        print("Analyzing results...")

        # Create analysis directory
        Path(self.dirs["analysis_dir"]).mkdir(parents=True, exist_ok=True)

        # Prepare file paths
        plugin_timing_file = f"{self.dirs['output_dir']}/plugin_timing_raw.csv"
        plugin_timing_summary_file = (
            f"{self.dirs['output_dir']}/plugin_timing_summary.csv"
        )
        pcap_bytes_file = f"{self.dirs['output_dir']}/pcap_measurements.csv"

        # Check which files exist
        has_plugin_timing = os.path.exists(plugin_timing_file)
        has_plugin_timing_summary = os.path.exists(plugin_timing_summary_file)
        has_pcap_bytes = os.path.exists(pcap_bytes_file)

        if not has_plugin_timing and not has_pcap_bytes:
            print(
                "Warning: No analysis files found (plugin_timing_raw.csv or pcap_measurements.csv)"
            )
            return

        # Build analysis command
        analysis_cmd = ["python3", "analyze_results.py"]  # Updated script name

        # Add plugin timing analysis if file exists
        if has_plugin_timing:
            analysis_cmd.extend(["--plugin-timing", plugin_timing_file])
            print(f"  - Plugin timing analysis: {plugin_timing_file}")

        if has_plugin_timing_summary:
            analysis_cmd.extend(["--plugin-timing-summary", plugin_timing_summary_file])
            print(f"  - Plugin timing summary analysis: {plugin_timing_summary_file}")

        # Add PCAP bytes analysis if file exists
        if has_pcap_bytes:
            analysis_cmd.extend(["--pcap-bytes", pcap_bytes_file])
            print(f"  - PCAP bytes analysis: {pcap_bytes_file}")

        # Add output directory
        analysis_cmd.extend(["--output", self.dirs["analysis_dir"]])

        # Add log scale if enabled
        if self.config.get("test", {}).get("log_scale", False):
            analysis_cmd.append("--log-scale")
            print("  - Using logarithmic scale for plots")

        try:
            print(f"Running analysis command: {' '.join(analysis_cmd)}")
            self._run_docker_command(analysis_cmd)
            print(
                f"✓ Analysis completed! Results available in {self.dirs['analysis_dir']}"
            )

            # List generated files
            if os.path.exists(self.dirs["analysis_dir"]):
                analysis_files = os.listdir(self.dirs["analysis_dir"])
                if analysis_files:
                    print("Generated analysis files:")
                    for file in sorted(analysis_files):
                        print(f"  - {file}")

        except subprocess.CalledProcessError as e:
            print(f"✗ Analysis failed: {e}")
            print("Check that analyze_results.py is in the current directory")
        except Exception as e:
            print(f"✗ Analysis error: {e}")

    def _prompt_container_shutdown(self):
        """Ask user if they want to stop containers based on role."""
        while True:
            response = input("Stop containers? (y/n): ").strip().lower()
            if response in ("y", "yes"):
                print("Stopping containers...")

                if self.role == "alice":
                    # Only stop Alice-related containers
                    containers_to_stop = [
                        "alice",
                        "qkd_server_alice",
                        "generate_key_alice",
                    ]
                    for container in containers_to_stop:
                        try:
                            subprocess.run(
                                ["docker", "stop", container],
                                check=False,
                                capture_output=True,
                            )
                            print(f"Stopped {container}")
                        except:
                            pass
                elif self.role == "bob":
                    # Only stop Bob-related containers
                    containers_to_stop = ["bob", "qkd_server_bob", "generate_key_bob"]
                    for container in containers_to_stop:
                        try:
                            subprocess.run(
                                ["docker", "stop", container],
                                check=False,
                                capture_output=True,
                            )
                            print(f"Stopped {container}")
                        except:
                            pass
                else:
                    # For "both", stop everything
                    compose_file = self.compose_file
                    self._run_docker_command(
                        ["docker-compose", "-f", compose_file, "down"]
                    )

                print("Containers stopped.")
                break
            elif response in ("n", "no"):
                break
            else:
                print("Please enter 'y' or 'n'.")

    def execute_workflow(self):
        """Execute the complete testing workflow with proper cleanup and coordination."""
        try:
            # Clean up any existing Pumba containers first
            self._cleanup_all_pumba_containers()

            if self.build:
                print("Rebuild requested - building containers from scratch")
                if not self.setup_environment(
                    build_only=False,
                    detached=True,
                ):
                    print("Container preparation failed. Exiting.")
                    return 1
            else:
                # Only check containers if not rebuilding
                if not self.check_containers():
                    print("Cannot proceed without running containers.")
                    return 1

            # Special instructions for distributed mode
            if self.network_mode == "distributed" and self.role in ["alice", "bob"]:
                print("\n" + "=" * 60)
                print("DISTRIBUTED MODE COORDINATION")
                print("=" * 60)

                if self.role == "alice":
                    print("You are running Alice. Make sure Bob is ready:")
                    print(
                        f"1. Bob should run: python benchmark.py --role bob --remote-ip {self.local_ip}"
                    )
                    print("2. Bob will start coordination server and signal when ready")
                    print("3. Alice will wait for Bob's signal before starting tests")
                    print("4. Tests will run in synchronized fashion")

                elif self.role == "bob":
                    print("You are running Bob. Coordination sequence:")
                    print("1. Bob (this instance) will start coordination server")
                    print("2. Bob will signal Alice that he's ready")
                    print(
                        f"3. Alice should then run: python benchmark.py --role alice --remote-ip {self.local_ip}"
                    )
                    print("4. Alice will coordinate test execution")
                    print("5. Tests will run in synchronized fashion")

                print("=" * 60)

                # Give user a moment to read
                input("Press Enter to continue...")

            # Log environment versions, Python packages and network parameters for reproducibility
            print("Logging environment versions for reproducibility...")
            log_versions(self, self.dirs["output_dir"])
            log_python_packages(self, self.dirs["output_dir"])
            log_network_configuration(self, self.dirs["output_dir"])

            # Run tests (containers are now guaranteed to be running)
            self.run_tests()

            # Analyze results if enabled
            if self.config["test"]["analyze_results"]:
                self.analyze_results()

            # Ask to stop containers
            self._prompt_container_shutdown()

            return 0

        except KeyboardInterrupt:
            print("\nTest interrupted by user. Cleaning up...")
            self._cleanup_pumba_containers()
            return 1
        except Exception as e:
            print(f"Error executing workflow: {e}")
            self._cleanup_pumba_containers()
            return 1
        finally:
            self._cleanup_pumba_containers()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="QKD-IPSec Testing Orchestrator")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_FILE,
        help=f"Path to configuration file (default: {DEFAULT_CONFIG_FILE})",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Force rebuild of Docker containers from scratch (same as build: true in YAML)",
    )
    parser.add_argument(
        "--detached",
        "-d",
        action="store_true",
        help="Start containers in detached mode when using --build",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Build without using Docker cache when using --build",
    )
    parser.add_argument(
        "--etsi-version",
        type=str,
        choices=["004", "014"],
        help="ETSI QKD API version (004 or 014)",
    )
    parser.add_argument(
        "--qkd-backend",
        type=str,
        choices=["python_client", "qukaydee", "cerberis-xgr", "simulated"],
        help="QKD backend to use",
    )
    parser.add_argument(
        "--role",
        type=str,
        choices=["alice", "bob", "both"],
        default="both",
        help="Role to run on this machine (alice, bob, or both for single-machine)",
    )
    parser.add_argument(
        "--remote-ip",
        type=str,
        help="IP address of the remote machine (required when role is alice or bob)",
    )
    parser.add_argument(
        "--local-ip",
        type=str,
        help="Local IP address to bind to (optional, auto-detected if not specified)",
    )
    parser.add_argument(
        "--coordination-port",
        type=int,
        default=8080,
        help="Port for coordination between Alice and Bob machines (default: 8080)",
    )
    parser.add_argument(
        "--qkd-initiation-mode",
        type=str,
        choices=["client", "server"],
        help="QKD initiation mode (client or server)",
    )
    parser.add_argument(
        "--iterations", "-i", type=int, help="Override number of test iterations"
    )
    parser.add_argument(
        "--no-analyze",
        "-n",
        action="store_true",
        help="Skip running analysis after tests",
    )
    parser.add_argument(
        "--no-network-conditions",
        "-nn",
        action="store_true",
        help="Skip applying network conditions",
    )
    parser.add_argument(
        "--latency", "-l", type=int, help="Override network latency in ms"
    )
    parser.add_argument(
        "--jitter", "-j", type=int, help="Override latency jitter in ms"
    )
    parser.add_argument(
        "--packet-loss", "-p", type=int, help="Override packet loss percentage"
    )
    parser.add_argument(
        "--duration",
        type=str,
        help="Override network conditions duration (e.g., '20m', '1h', '180m')",
    )
    parser.add_argument(
        "--alice-ip",
        type=str,
        help=f"Override Alice's IP address (default: {DEFAULT_ALICE_IP})",
    )
    parser.add_argument(
        "--bob-ip",
        type=str,
        help=f"Override Bob's IP address (default: {DEFAULT_BOB_IP})",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    orchestrator = QKDTestOrchestrator(config_file=args.config, cli_args=args)
    sys.exit(orchestrator.execute_workflow())
