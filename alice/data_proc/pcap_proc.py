#!/usr/bin/env python3
# Copyright (C) 2024-2025 Javier Blanco-Romero @fj-blanco (UC3M, QURSA project)
#
# PCAP processing

import subprocess
import pandas as pd
import numpy as np
import os
from collections import defaultdict


def extract_ike_packets(pcap_file, log_message):
    """Extract basic IKE packet info using tshark"""
    if not os.path.exists(pcap_file):
        log_message(f"Error: PCAP file not found: {pcap_file}")
        return []

    if os.path.getsize(pcap_file) == 0:
        log_message(f"Warning: PCAP file is empty: {pcap_file}")
        return []

    log_message(f"Extracting IKE data from {pcap_file} using tshark...")

    cmd = [
        "tshark",
        "-r",
        pcap_file,
        "-t",
        "e",
        "-Y",
        "isakmp",
        "-T",
        "fields",
        "-e",
        "frame.time_epoch",
        "-e",
        "isakmp.exchangetype",
        "-e",
        "isakmp.messageid",
        "-e",
        "frame.len",
        "-E",
        "separator=,",
        "-E",
        "occurrence=f",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            log_message(f"tshark error: {result.stderr}")
            return []

        lines = result.stdout.strip().split("\n")
        if not lines or lines == [""]:
            log_message("No IKE packets found by tshark")
            return []

        packets = []
        exchange_types = {
            34: "ike_sa_init",
            35: "ike_auth",
            36: "create_child_sa",
            37: "informational",
            43: "ike_intermediate",
        }

        for i, line in enumerate(lines):
            if line.strip():
                try:
                    parts = line.split(",")
                    if len(parts) >= 4:
                        timestamp = float(parts[0]) if parts[0] else 0
                        exchange_type_num = int(parts[1]) if parts[1] else 0

                        # Handle hex message IDs
                        message_id_str = parts[2] if parts[2] else "0"
                        message_id = (
                            int(message_id_str, 16)
                            if message_id_str.startswith("0x")
                            else int(message_id_str)
                        )

                        frame_length = int(parts[3]) if parts[3] else 0
                        exchange_type = exchange_types.get(
                            exchange_type_num, f"unknown_{exchange_type_num}"
                        )

                        packets.append(
                            {
                                "timestamp": timestamp,
                                "exchange_type": exchange_type,
                                "message_id": message_id,
                                "frame_length": frame_length,
                            }
                        )

                except (ValueError, IndexError) as e:
                    log_message(f"Warning: Could not parse packet {i+1}: {line} - {e}")
                    continue

        log_message(f"Found {len(packets)} IKE packets")
        return packets

    except Exception as e:
        log_message(f"Error running tshark: {e}")
        return []


def get_total_pcap_bytes(pcap_file, log_message):
    """Get total bytes in PCAP file (all traffic)"""
    if not os.path.exists(pcap_file):
        log_message(f"Error: PCAP file not found: {pcap_file}")
        return 0

    log_message(f"Getting total PCAP bytes from {pcap_file}...")

    cmd = [
        "tshark",
        "-r",
        pcap_file,
        "-T",
        "fields",
        "-e",
        "frame.len",
        "-E",
        "separator=,",
        "-E",
        "occurrence=f",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            log_message(f"tshark error: {result.stderr}")
            return 0

        lines = result.stdout.strip().split("\n")
        if not lines or lines == [""]:
            log_message("No packets found in PCAP")
            return 0

        total_bytes = 0
        packet_count = 0

        for line in lines:
            if line.strip():
                try:
                    frame_length = int(line.strip()) if line.strip() else 0
                    total_bytes += frame_length
                    packet_count += 1
                except ValueError:
                    continue

        log_message(
            f"Total PCAP contains {packet_count} packets, {total_bytes:,} bytes"
        )
        return total_bytes

    except Exception as e:
        log_message(f"Error getting total PCAP bytes: {e}")
        return 0


def calculate_ike_handshake_latencies(packets, filename, log_message, iters=1):
    """Calculate IKE handshake latency from first IKE_SA_INIT to last message before first IKE_AUTH per iteration"""
    if not packets:
        log_message("No packets for latency calculation")
        return []

    # Convert packets to DataFrame
    df_packets = pd.DataFrame(packets)
    df_packets = df_packets.sort_values("timestamp").reset_index(drop=True)

    # Find IKE_SA_INIT packets to identify iteration boundaries
    ike_sa_init_packets = df_packets[df_packets["exchange_type"] == "ike_sa_init"]

    if len(ike_sa_init_packets) == 0:
        log_message("Error: No IKE_SA_INIT packets found")
        return []

    # Each iteration should have 2 IKE_SA_INIT packets (request + response)
    expected_sa_init_count = iters * 2
    actual_sa_init_count = len(ike_sa_init_packets)

    if actual_sa_init_count != expected_sa_init_count:
        log_message(
            f"Warning: Expected {expected_sa_init_count} IKE_SA_INIT packets ({iters} iterations × 2), found {actual_sa_init_count}"
        )
        # Adjust iterations based on actual packets found
        iters = actual_sa_init_count // 2
        log_message(f"Adjusting to {iters} iterations based on packet count")

    if iters == 0:
        log_message("Error: No complete iterations found")
        return []

    # Group IKE_SA_INIT packets by pairs (request + response)
    ike_sa_init_timestamps = ike_sa_init_packets["timestamp"].values

    ike_handshake_latencies = []

    for iter_num in range(iters):
        # Each iteration starts with the first IKE_SA_INIT packet of the pair
        iter_start_time = ike_sa_init_timestamps[iter_num * 2]  # First packet of pair

        # Define iteration boundary
        if iter_num < iters - 1:
            # Next iteration starts with the next IKE_SA_INIT pair
            iter_end_boundary = ike_sa_init_timestamps[(iter_num + 1) * 2]
            iter_mask = (df_packets["timestamp"] >= iter_start_time) & (
                df_packets["timestamp"] < iter_end_boundary
            )
        else:
            # Last iteration - include all remaining packets
            iter_mask = df_packets["timestamp"] >= iter_start_time

        iter_packets = df_packets[iter_mask]

        # Find first IKE_SA_INIT and first IKE_AUTH in this iteration
        iter_ike_sa_init = iter_packets[iter_packets["exchange_type"] == "ike_sa_init"]
        iter_ike_auth = iter_packets[iter_packets["exchange_type"] == "ike_auth"]

        if len(iter_ike_sa_init) == 0:
            log_message(f"Warning: Missing IKE_SA_INIT in iteration {iter_num + 1}")
            ike_handshake_latencies.append(0.0)
            continue

        # Get first IKE_SA_INIT timestamp
        first_ike_sa_init = iter_ike_sa_init["timestamp"].min()

        # Find last message before first IKE_AUTH
        if len(iter_ike_auth) > 0:
            first_ike_auth = iter_ike_auth["timestamp"].min()
            # Get all packets before first IKE_AUTH (including IKE_SA_INIT and IKE_INTERMEDIATE)
            pre_auth_packets = iter_packets[iter_packets["timestamp"] < first_ike_auth]

            if len(pre_auth_packets) > 0:
                last_pre_auth_timestamp = pre_auth_packets["timestamp"].max()

                # Show breakdown of included exchanges
                pre_auth_exchanges = pre_auth_packets["exchange_type"].value_counts()
                exchange_info = ", ".join(
                    [f"{ex}: {count}" for ex, count in pre_auth_exchanges.items()]
                )
                log_message(f"  Included exchanges: {exchange_info}")
            else:
                log_message(
                    f"Warning: No packets found before IKE_AUTH in iteration {iter_num + 1}"
                )
                last_pre_auth_timestamp = first_ike_sa_init
        else:
            # No IKE_AUTH found - use last packet in iteration (fallback)
            log_message(
                f"Warning: No IKE_AUTH found in iteration {iter_num + 1}, using last packet"
            )
            last_pre_auth_timestamp = iter_packets["timestamp"].max()

        # Calculate handshake latency: first IKE_SA_INIT to last pre-auth message
        handshake_latency = last_pre_auth_timestamp - first_ike_sa_init
        ike_handshake_latencies.append(handshake_latency)

        log_message(
            f"Iteration {iter_num + 1}: {handshake_latency:.6f}s ({handshake_latency*1000:.2f}ms)"
        )
        log_message(f"  First IKE_SA_INIT: {first_ike_sa_init:.6f}")
        log_message(f"  Last pre-AUTH message: {last_pre_auth_timestamp:.6f}")

    # Save simple summary
    summary_data = []
    for i, latency in enumerate(ike_handshake_latencies):
        summary_data.append(
            {
                "iteration": i + 1,
                "handshake_latency_seconds": latency,
                "handshake_latency_ms": latency * 1000,
            }
        )

    pd.DataFrame(summary_data).to_csv(filename, index=False)
    log_message(f"Saved handshake summary to {filename}")

    return ike_handshake_latencies


def analyze_packets(packets, prop, log_message, total_pcap_bytes=0, iters=1):
    """Analyze packet counts and bytes"""
    if not packets:
        log_message("No packets to analyze")
        return create_empty_combined_stats(prop)

    # Count packets by exchange type
    exchange_counts = defaultdict(int)
    exchange_bytes = defaultdict(int)
    total_ike_bytes = 0

    for packet in packets:
        exchange_type = packet["exchange_type"]
        frame_length = packet["frame_length"]
        exchange_counts[exchange_type] += 1
        exchange_bytes[exchange_type] += frame_length
        total_ike_bytes += frame_length

    # Normalize by iterations
    total_ike_bytes = total_ike_bytes // iters
    total_pcap_bytes = total_pcap_bytes // iters
    total_message_count = len(packets) // iters

    # Log analysis results
    log_message(
        f"Analysis complete: {total_message_count} IKE packets per iteration, {total_ike_bytes:,} IKE bytes per iteration"
    )
    if total_pcap_bytes > 0:
        non_ike_bytes = total_pcap_bytes - total_ike_bytes
        ike_percentage = (total_ike_bytes / total_pcap_bytes) * 100

        log_message(f"Total PCAP per iteration: {total_pcap_bytes:,} bytes")
        log_message(
            f"IKE traffic per iteration: {total_ike_bytes:,} bytes ({ike_percentage:.1f}%)"
        )
        log_message(
            f"Non-IKE traffic per iteration: {non_ike_bytes:,} bytes ({100-ike_percentage:.1f}%)"
        )

    # Create combined stats
    combined_stats = {
        "proposal": prop,
        "total_ike_bytes": total_ike_bytes,
        "total_pcap_bytes": total_pcap_bytes,
        "non_ike_bytes": (
            total_pcap_bytes - total_ike_bytes if total_pcap_bytes > 0 else 0
        ),
        "ike_percentage": (
            (total_ike_bytes / total_pcap_bytes * 100) if total_pcap_bytes > 0 else 0
        ),
        "total_message_count": total_message_count,
        "ike_latency_avg": 0,  # Will be updated with IKE handshake latency
        "ike_latency_std": 0,  # Will be updated with IKE handshake latency
    }

    # Add exchange stats (normalized)
    exchanges = [
        "ike_sa_init",
        "ike_auth",
        "ike_intermediate",
        "informational",
        "create_child_sa",
    ]
    for exchange in exchanges:
        count = exchange_counts.get(exchange, 0) // iters
        bytes_val = exchange_bytes.get(exchange, 0) // iters
        combined_stats.update(
            {
                f"{exchange}_count": count,
                f"{exchange}_bytes": bytes_val,
            }
        )

    return combined_stats


def update_stats_with_ike_latencies(combined_stats, ike_handshake_latencies):
    """Update combined stats with IKE handshake latencies"""
    if ike_handshake_latencies:
        latencies_array = np.array(ike_handshake_latencies)
        combined_stats["ike_latency_avg"] = np.mean(latencies_array)
        combined_stats["ike_latency_std"] = (
            np.std(latencies_array, ddof=1) if len(latencies_array) > 1 else 0
        )


def create_empty_combined_stats(prop):
    """Create empty combined stats structure"""
    combined_stats = {
        "proposal": prop,
        "total_ike_bytes": 0,
        "total_pcap_bytes": 0,
        "non_ike_bytes": 0,
        "ike_percentage": 0,
        "total_message_count": 0,
        "ike_latency_avg": 0,
        "ike_latency_std": 0,
    }

    exchanges = [
        "ike_sa_init",
        "ike_auth",
        "ike_intermediate",
        "informational",
        "create_child_sa",
    ]
    for exchange in exchanges:
        combined_stats.update(
            {
                f"{exchange}_count": 0,
                f"{exchange}_bytes": 0,
            }
        )

    return combined_stats


def process_ike_data(pcap_file, output_dir, prop, log_message, iters=1):
    """Main processing function"""
    log_message(f"Processing PCAP for proposal: {prop} with {iters} iterations")

    # Get total PCAP bytes first
    total_pcap_bytes = get_total_pcap_bytes(pcap_file, log_message)

    # Extract IKE packets
    packets = extract_ike_packets(pcap_file, log_message)

    # Analyze packets for counts and bytes
    combined_stats = analyze_packets(
        packets, prop, log_message, total_pcap_bytes, iters
    )

    # Calculate IKE handshake latencies per iteration
    ike_handshake_latencies = calculate_ike_handshake_latencies(
        packets, f"{output_dir}/handshake_summary_{prop}.csv", log_message, iters
    )

    # Update combined stats with latency data
    update_stats_with_ike_latencies(combined_stats, ike_handshake_latencies)

    # Log final results
    ike_latency_ms = combined_stats.get("ike_latency_avg", 0) * 1000
    ike_latency_std_ms = combined_stats.get("ike_latency_std", 0) * 1000
    log_message(
        f"Results for {prop}: {combined_stats['total_message_count']} IKE packets per iteration, "
        f"{combined_stats['total_ike_bytes']:,} IKE bytes per iteration, "
        f"{combined_stats['total_pcap_bytes']:,} total bytes per iteration, "
        f"IKE handshake latency: {ike_latency_ms:.2f}±{ike_latency_std_ms:.2f}ms"
    )

    return combined_stats, {}  # No exchange-level latencies returned


def generate_pcap_report(
    output_dir, proposals, df_plugin_timing, df_bytes, log_message
):
    """Generate PCAP analysis report"""
    all_latency_measurements = []

    with open(f"{output_dir}/report.txt", "w") as f:
        f.write("StrongSwan QKD Plugin Performance Test Results\n" + "=" * 50 + "\n\n")

        for prop in proposals:
            byte_data = df_bytes[df_bytes["proposal"] == prop]
            if byte_data.empty:
                f.write(f"Proposal: {prop}\nNo data available\n\n")
                continue

            row = byte_data.iloc[0]
            f.write(f"Proposal: {prop}\n" + "=" * (len(prop) + 10) + "\n\n")

            # Basic metrics
            total_packets = row.get("total_message_count", 0)
            total_ike_bytes = row.get("total_ike_bytes", 0)
            total_pcap_bytes = row.get("total_pcap_bytes", 0)
            non_ike_bytes = row.get("non_ike_bytes", 0)
            ike_percentage = row.get("ike_percentage", 0)
            ike_latency_avg = row.get("ike_latency_avg", 0)
            ike_latency_std = row.get("ike_latency_std", 0)

            f.write(f"Total IKE Packets (per iteration): {total_packets}\n")
            f.write(f"Total IKE Bytes (per iteration): {total_ike_bytes:,}\n")
            f.write(f"Total PCAP Bytes (per iteration): {total_pcap_bytes:,}\n")
            f.write(f"Non-IKE Bytes (per iteration): {non_ike_bytes:,}\n")
            f.write(f"IKE Percentage: {ike_percentage:.1f}%\n")
            f.write(
                f"IKE Handshake Latency: {ike_latency_avg:.6f} ± {ike_latency_std:.6f} seconds "
                f"({ike_latency_avg*1000:.2f} ± {ike_latency_std*1000:.2f} ms)\n\n"
            )

            # Exchange breakdown (counts and bytes only)
            f.write("Exchange Type Breakdown:\n")
            exchanges = [
                ("ike_sa_init", "IKE_SA_INIT"),
                ("ike_intermediate", "IKE_INTERMEDIATE"),
                ("ike_auth", "IKE_AUTH"),
                ("informational", "INFORMATIONAL"),
                ("create_child_sa", "CREATE_CHILD_SA"),
            ]

            for exchange, display_name in exchanges:
                count = row.get(f"{exchange}_count", 0)
                if count > 0:
                    bytes_val = row.get(f"{exchange}_bytes", 0)
                    f.write(f"  {display_name}: {count} packets, {bytes_val:,} bytes\n")

            # Plugin timing
            plugin_data = (
                df_plugin_timing[df_plugin_timing["proposal"] == prop]
                if not df_plugin_timing.empty
                else pd.DataFrame()
            )
            if not plugin_data.empty:
                plugin_row = plugin_data.iloc[0]
                f.write(
                    f"\nPlugin Timing:\n  Total Operations: {plugin_row.get('total_algorithms', 0)}\n"
                )
                f.write(f"  Total Time: {plugin_row.get('total_time_ms', 0):.2f} ms\n")

            f.write("\n" + "=" * 80 + "\n\n")

            # Collect latency summary data
            summary_file = f"{output_dir}/handshake_summary_{prop}.csv"
            try:
                if os.path.exists(summary_file):
                    summary_df = pd.read_csv(summary_file)
                    if not summary_df.empty:
                        if "proposal" not in summary_df.columns:
                            summary_df["proposal"] = prop
                        all_latency_measurements.append(summary_df)
                        log_message(
                            f"Collected {len(summary_df)} handshake measurements for {prop}"
                        )
                else:
                    log_message(f"No handshake summary file found for {prop}")
            except Exception as e:
                log_message(f"Error reading handshake summary for {prop}: {e}")

        # Overall summary
        if not df_bytes.empty:
            f.write("OVERALL SUMMARY\n===============\n\n")

            # Use pandas aggregation functions
            summary_stats = df_bytes.agg(
                {
                    "total_ike_bytes": "sum",
                    "total_pcap_bytes": "sum",
                    "total_message_count": "sum",
                    "ike_latency_avg": "mean",
                    "ike_latency_std": "mean",
                }
            )

            f.write(
                f"Total IKE bytes processed (per iteration): {summary_stats['total_ike_bytes']:,}\n"
            )
            f.write(
                f"Total PCAP bytes processed (per iteration): {summary_stats['total_pcap_bytes']:,}\n"
            )
            f.write(
                f"Total messages processed (per iteration): {summary_stats['total_message_count']:,}\n"
            )
            f.write(
                f"Average IKE handshake latency across all proposals: "
                f"{summary_stats['ike_latency_avg']*1000:.2f} ± {summary_stats['ike_latency_std']*1000:.2f} ms\n"
            )

    # Save aggregated measurements
    if all_latency_measurements:
        aggregated_df = pd.concat(all_latency_measurements, ignore_index=True)
        aggregated_file = f"{output_dir}/pcap_aggregated_measurements.csv"
        aggregated_df.to_csv(aggregated_file, index=False)
        proposal_counts = aggregated_df["proposal"].value_counts()
        log_message(
            f"Aggregated {len(aggregated_df)} handshake latency measurements from {len(proposal_counts)} proposals"
        )
        log_message(f"Aggregated measurements saved to: {aggregated_file}")
    else:
        log_message("No handshake latency measurements found to aggregate")

    log_message(f"Complete report generated: {output_dir}/report.txt")
