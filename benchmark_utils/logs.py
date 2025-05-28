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


def log_network_configuration(orchestrator, output_dir):
    """Log network configuration including MTU settings for scientific reproducibility."""

    network_info = {
        "timestamp": datetime.datetime.now().isoformat(),
        "host_network": {},
        "container_network": {},
        "docker_network": {},
        "fragmentation_settings": {},
    }

    print("=== Logging Network Configuration ===")

    # Check host MTU
    try:
        result = subprocess.run(
            ["ip", "link", "show"], capture_output=True, text=True, check=True
        )

        host_interfaces = {}
        for line in result.stdout.split("\n"):
            if "mtu" in line.lower():
                # Parse interface name and MTU
                parts = line.strip().split()
                if len(parts) >= 2:
                    interface = parts[1].rstrip(":")
                    mtu_idx = next(
                        (i for i, part in enumerate(parts) if part == "mtu"), None
                    )
                    if mtu_idx and mtu_idx + 1 < len(parts):
                        mtu = parts[mtu_idx + 1]
                        host_interfaces[interface] = {
                            "mtu": mtu,
                            "full_line": line.strip(),
                        }

        network_info["host_network"]["interfaces"] = host_interfaces
        print(f"Host network interfaces logged: {len(host_interfaces)} found")

    except Exception as e:
        network_info["host_network"]["error"] = str(e)
        print(f"Could not get host MTU: {e}")

    # Check container MTUs and routing
    for container in ["alice", "bob"]:
        try:
            container_obj = orchestrator.docker_client.containers.get(container)

            # Get interface info
            result = container_obj.exec_run("ip link show")
            if result.exit_code == 0:
                container_interfaces = {}
                for line in result.output.decode().split("\n"):
                    if "mtu" in line.lower():
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            interface = parts[1].rstrip(":")
                            mtu_idx = next(
                                (i for i, part in enumerate(parts) if part == "mtu"),
                                None,
                            )
                            if mtu_idx and mtu_idx + 1 < len(parts):
                                mtu = parts[mtu_idx + 1]
                                container_interfaces[interface] = {
                                    "mtu": mtu,
                                    "full_line": line.strip(),
                                }

                network_info["container_network"][container] = {
                    "interfaces": container_interfaces
                }

            # Get routing table
            result = container_obj.exec_run("ip route show")
            if result.exit_code == 0:
                network_info["container_network"][container][
                    "routing"
                ] = result.output.decode().strip()

            # Get default route MTU if available
            result = container_obj.exec_run("ip route get 8.8.8.8")
            if result.exit_code == 0:
                route_output = result.output.decode().strip()
                network_info["container_network"][container][
                    "default_route"
                ] = route_output

        except Exception as e:
            network_info["container_network"][container] = {"error": str(e)}
            print(f"Could not get {container} network info: {e}")

    # Check Docker network MTU
    try:
        networks = orchestrator.docker_client.networks.list()
        docker_networks = {}

        for network in networks:
            # Check if our containers are in this network
            network_containers = (
                [c.name for c in network.containers]
                if hasattr(network, "containers")
                else []
            )
            if "alice" in network_containers or "bob" in network_containers:
                network_attrs = network.attrs
                options = network_attrs.get("Options", {})

                docker_networks[network.name] = {
                    "id": network.id[:12],
                    "driver": network_attrs.get("Driver", "unknown"),
                    "mtu": options.get("com.docker.network.driver.mtu", "default"),
                    "containers": network_containers,
                    "ipam": network_attrs.get("IPAM", {}),
                }

        network_info["docker_network"] = docker_networks
        print(f"Docker networks logged: {len(docker_networks)} relevant networks found")

    except Exception as e:
        network_info["docker_network"]["error"] = str(e)
        print(f"Could not get Docker network info: {e}")

    # Check strongSwan fragmentation settings (minimal)
    try:
        alice_container = orchestrator.docker_client.containers.get("alice")

        # Get just the fragmentation-related settings
        result = alice_container.exec_run(
            "grep -E '(fragment_size|max_packet|send_vendor_id)' /etc/strongswan.conf"
        )
        if result.exit_code == 0:
            fragmentation_lines = result.output.decode().strip().split("\n")
            fragmentation_settings = {}

            for line in fragmentation_lines:
                if "=" in line:
                    key, value = line.split("=", 1)
                    fragmentation_settings[key.strip()] = value.strip()

            network_info["fragmentation_settings"] = fragmentation_settings

    except Exception as e:
        network_info["fragmentation_settings"]["error"] = str(e)

    # Write to files
    try:
        # JSON format
        json_file = Path(output_dir) / "network_configuration.json"
        with open(json_file, "w") as f:
            json.dump(network_info, f, indent=2, sort_keys=True)

        # Human readable format
        txt_file = Path(output_dir) / "network_configuration.txt"
        with open(txt_file, "w") as f:
            f.write("=== Network Configuration Analysis ===\n")
            f.write(f"Generated: {network_info['timestamp']}\n\n")

            # Host network
            f.write("[Host Network Interfaces]\n")
            if "interfaces" in network_info["host_network"]:
                for iface, info in network_info["host_network"]["interfaces"].items():
                    f.write(f"  {iface}: MTU {info['mtu']}\n")
            else:
                f.write(
                    f"  Error: {network_info['host_network'].get('error', 'No data')}\n"
                )
            f.write("\n")

            # Container network
            f.write("[Container Network Configuration]\n")
            for container, info in network_info["container_network"].items():
                f.write(f"  {container.upper()}:\n")
                if "interfaces" in info:
                    for iface, iface_info in info["interfaces"].items():
                        f.write(f"    {iface}: MTU {iface_info['mtu']}\n")
                if "routing" in info:
                    f.write(f"    Routing: {info['routing'][:100]}...\n")
                if "error" in info:
                    f.write(f"    Error: {info['error']}\n")
            f.write("\n")

            # Docker network
            f.write("[Docker Network Configuration]\n")
            if "error" not in network_info["docker_network"]:
                for net_name, net_info in network_info["docker_network"].items():
                    f.write(f"  {net_name}:\n")
                    f.write(f"    Driver: {net_info['driver']}\n")
                    f.write(f"    MTU: {net_info['mtu']}\n")
                    f.write(f"    Containers: {', '.join(net_info['containers'])}\n")
            else:
                f.write(f"  Error: {network_info['docker_network']['error']}\n")
            f.write("\n")

            # Fragmentation settings
            f.write("[StrongSwan Fragmentation Settings]\n")
            if "error" not in network_info["fragmentation_settings"]:
                for key, value in network_info["fragmentation_settings"].items():
                    f.write(f"  {key}: {value}\n")
            else:
                f.write(f"  Error: {network_info['fragmentation_settings']['error']}\n")

        print(f"Network configuration logged to: {json_file}")

    except Exception as e:
        print(f"Warning: Could not write network configuration files: {e}")

    return network_info


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
