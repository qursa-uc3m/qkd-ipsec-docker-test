#!/usr/bin/env python3
# Copyright (C) 2024-2025 Javier Blanco-Romero @fj-blanco (UC3M, QURSA project)
#
# alice/shared/utils.py

import subprocess


def run_cmd(
    cmd, capture_output=False, start_new_session=False, input_data=None, output_dir=None
):
    """Run a command with proper environment sourcing"""
    # Prepend source command to ensure environment is loaded
    full_cmd = f"source /set_env.sh && {cmd}"

    if cmd == "/charon":
        # Use tee to capture the output to the log file
        if output_dir:
            tee_cmd = f"{full_cmd} 2>&1 | tee -a {output_dir}/alice_log.txt"
        else:
            tee_cmd = f"{full_cmd} 2>&1"
        return subprocess.Popen(["bash", "-c", tee_cmd], start_new_session=True)
    # Run through bash to handle the source command
    elif capture_output:
        return subprocess.run(
            ["bash", "-c", full_cmd], capture_output=True, text=True, input=input_data
        )
    elif start_new_session:
        return subprocess.Popen(
            ["bash", "-c", full_cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    else:
        return subprocess.run(["bash", "-c", full_cmd])
