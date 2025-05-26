#!/usr/bin/env python3
# Copyright (C) 2024-2025 Javier Blanco-Romero @fj-blanco (UC3M, QURSA project)
#
# utils/logs.py
"""
Logs utils
"""

import datetime
import json
import platform
import subprocess
import sys
from pathlib import Path


def log_versions(orchestrator, output_dir):
    """Log all relevant version information for scientific reproducibility."""

    versions = {
        "timestamp": datetime.datetime.now().isoformat(),
        "host_system": {
            "os": platform.system(),
            "os_version": platform.version(),
            "platform": platform.platform(),
            "architecture": platform.architecture()[0],
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "container_tools": {},
        "host_tools": {},
    }

    # Get versions from Alice container
    try:
        alice_container = orchestrator.docker_client.containers.get("alice")

        # Key tools in container
        container_tools = {
            "python3 --version": "python_version",
            "tshark --version": "tshark_version",
            "/usr/libexec/ipsec/charon --version": "strongswan_version",
            "openssl version": "openssl_version",
        }

        for cmd, key in container_tools.items():
            try:
                result = alice_container.exec_run(cmd)
                if result.exit_code == 0:
                    versions["container_tools"][key] = (
                        result.output.decode().strip().split("\n")[0]
                    )
            except:
                versions["container_tools"][key] = "not_available"

    except Exception as e:
        versions["container_tools"]["error"] = f"Could not access containers: {str(e)}"

    # Host tools
    host_tools = ["docker --version", "docker-compose --version", "git --version"]
    for cmd in host_tools:
        tool_name = cmd.split()[0] + "_version"
        try:
            result = subprocess.run(
                cmd.split(), capture_output=True, text=True, check=True
            )
            versions["host_tools"][tool_name] = result.stdout.strip().split("\n")[0]
        except:
            versions["host_tools"][tool_name] = "not_available"

    # Write to files
    try:
        output_file = Path(output_dir) / "environment_versions.json"
        with open(output_file, "w") as f:
            json.dump(versions, f, indent=2, sort_keys=True)

        readable_file = Path(output_dir) / "environment_versions.txt"
        with open(readable_file, "w") as f:
            f.write("=== QKD-IPSec Test Environment Versions ===\n")
            f.write(f"Generated: {versions['timestamp']}\n\n")

            for category, items in versions.items():
                if category != "timestamp" and isinstance(items, dict):
                    f.write(f"[{category.replace('_', ' ').title()}]\n")
                    for key, value in items.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")

        print(f"Environment versions logged to: {output_file}")

    except Exception as e:
        print(f"Warning: Could not write version log files: {e}")

    return versions


def log_python_packages(orchestrator, output_dir):
    """Log all installed Python packages and their versions."""
    packages_info = {
        "timestamp": datetime.datetime.now().isoformat(),
        "host_packages": {},
        "container_packages": {},
    }

    # Host Python packages
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            check=True,
        )
        host_packages = json.loads(result.stdout)
        packages_info["host_packages"] = {
            pkg["name"]: pkg["version"] for pkg in host_packages
        }
    except Exception as e:
        packages_info["host_packages"][
            "error"
        ] = f"Could not get host packages: {str(e)}"

    # Container Python packages
    try:
        alice_container = orchestrator.docker_client.containers.get("alice")
        result = alice_container.exec_run("python3 -m pip list --format=json")

        if result.exit_code == 0:
            container_packages = json.loads(result.output.decode())
            packages_info["container_packages"] = {
                pkg["name"]: pkg["version"] for pkg in container_packages
            }
        else:
            packages_info["container_packages"][
                "error"
            ] = "pip list failed in container"

    except Exception as e:
        packages_info["container_packages"][
            "error"
        ] = f"Could not access container: {str(e)}"

    # Write to files
    try:
        # JSON format
        json_file = Path(output_dir) / "python_packages.json"
        with open(json_file, "w") as f:
            json.dump(packages_info, f, indent=2, sort_keys=True)

        # Human readable format
        txt_file = Path(output_dir) / "python_packages.txt"
        with open(txt_file, "w") as f:
            f.write("=== Python Packages Versions ===\n")
            f.write(f"Generated: {packages_info['timestamp']}\n\n")

            f.write("[Host Python Packages]\n")
            if (
                isinstance(packages_info["host_packages"], dict)
                and "error" not in packages_info["host_packages"]
            ):
                for name, version in sorted(packages_info["host_packages"].items()):
                    f.write(f"  {name}: {version}\n")
            else:
                f.write(
                    f"  Error: {packages_info['host_packages'].get('error', 'Unknown error')}\n"
                )
            f.write("\n")

            f.write("[Container Python Packages]\n")
            if (
                isinstance(packages_info["container_packages"], dict)
                and "error" not in packages_info["container_packages"]
            ):
                for name, version in sorted(
                    packages_info["container_packages"].items()
                ):
                    f.write(f"  {name}: {version}\n")
            else:
                f.write(
                    f"  Error: {packages_info['container_packages'].get('error', 'Unknown error')}\n"
                )

        print(f"Python packages logged to: {json_file}")

    except Exception as e:
        print(f"Warning: Could not write Python packages log files: {e}")

    return packages_info
