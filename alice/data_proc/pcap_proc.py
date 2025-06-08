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
        "frame.len",  # Frame length (might be fragment)
        "-e",
        "isakmp.length",  # IKE message size
        "-e",
        "ip.frag_offset",  # Optional fragmentation info
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

        size_corrections = 0
        large_messages = []

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

                        # Get ISAKMP length (TRUE IKE message size)
                        isakmp_length = (
                            int(parts[4]) if len(parts) > 4 and parts[4] else 0
                        )

                        # Get fragmentation info if available
                        frag_offset = parts[5] if len(parts) > 5 else "0"
                        is_fragmented = frag_offset != "0" and frag_offset != ""

                        # PRIORITY: Use ISAKMP length when available (most accurate)
                        if isakmp_length > 0:
                            actual_size = isakmp_length
                            if isakmp_length != frame_length:
                                size_corrections += 1
                                log_message(
                                    f"  Size corrected: Frame={frame_length}B → True IKE={isakmp_length}B"
                                )
                        else:
                            actual_size = frame_length

                        exchange_type = exchange_types.get(
                            exchange_type_num, f"unknown_{exchange_type_num}"
                        )

                        packet_info = {
                            "timestamp": timestamp,
                            "exchange_type": exchange_type,
                            "message_id": message_id,
                            "frame_length": actual_size,  # Use TRUE IKE size for backward compatibility
                            "raw_frame_length": frame_length,  # Raw frame length
                            "isakmp_length": isakmp_length,  # True ISAKMP size
                            "fragmented": is_fragmented
                            or (isakmp_length > frame_length),
                        }

                        packets.append(packet_info)

                        # Track large messages
                        if actual_size > 1400:
                            large_messages.append(f"{exchange_type}: {actual_size}B")

                except (ValueError, IndexError) as e:
                    log_message(f"Warning: Could not parse packet {i+1}: {line} - {e}")
                    continue

        log_message(f"Found {len(packets)} IKE packets")

        # Report size corrections (fragmentation handling)
        if size_corrections > 0:
            log_message(
                f"✓ Corrected {size_corrections} packet sizes (fragmentation detected)"
            )

        if large_messages:
            log_message(f"Large IKE messages (>1400B): {len(large_messages)} found")
            for msg in large_messages[:3]:  # Show first 3
                log_message(f"  {msg}")
            if len(large_messages) > 3:
                log_message(f"  ... and {len(large_messages) - 3} more")

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
        "-e",
        "ip.frag_offset",
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
        fragment_count = 0

        for line in lines:
            if line.strip():
                try:
                    parts = line.split(",")
                    frame_length = int(parts[0]) if parts[0] else 0
                    frag_offset = parts[1] if len(parts) > 1 else "0"

                    total_bytes += frame_length
                    packet_count += 1

                    if frag_offset and frag_offset != "0":
                        fragment_count += 1

                except ValueError:
                    continue

        log_message(
            f"Total PCAP contains {packet_count} packets, {total_bytes:,} bytes"
        )
        return total_bytes

    except Exception as e:
        log_message(f"Error getting total PCAP bytes: {e}")
        return 0


def calculate_ike_handshake_latencies(
    packets, filename, log_message, iters=1, pcap_file=None
):
    """Calculate IKE handshake latency"""
    if not packets:
        log_message("No packets for latency calculation")
        return []

    df_packets = pd.DataFrame(packets).sort_values("timestamp").reset_index(drop=True)

    # Fragment correction
    fragment_corrections = {}
    if pcap_file and os.path.exists(pcap_file):
        try:
            result = subprocess.run(
                [
                    "tshark",
                    "-r",
                    pcap_file,
                    "-t",
                    "e",
                    "-Y",
                    "(udp.port == 500 or udp.port == 4500) and (ip.frag_offset == 0 or isakmp)",
                    "-T",
                    "fields",
                    "-e",
                    "frame.time_epoch",
                    "-e",
                    "ip.frag_offset",
                    "-e",
                    "isakmp.exchangetype",
                    "-e",
                    "ip.id",
                    "-E",
                    "separator=,",
                    "-E",
                    "occurrence=f",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )

            if result.returncode == 0:
                first_frags, reassembled = {}, {}
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        parts = line.split(",")
                        if len(parts) >= 4:
                            timestamp = float(parts[0]) if parts[0] else 0
                            if parts[1] == "0":  # First fragment
                                first_frags[parts[3]] = timestamp
                            elif parts[2] in [
                                "34",
                                "43",
                            ]:  # IKE_SA_INIT or IKE_INTERMEDIATE
                                reassembled[parts[3]] = timestamp

                for ip_id in first_frags:
                    if ip_id in reassembled and reassembled[ip_id] > first_frags[ip_id]:
                        fragment_corrections[reassembled[ip_id]] = first_frags[ip_id]

                if fragment_corrections:
                    log_message(
                        f"Applied {len(fragment_corrections)} fragment timing corrections"
                    )
        except Exception as e:
            log_message(f"Fragment correction failed: {e}")

    # Apply corrections
    for i, row in df_packets.iterrows():
        if row["timestamp"] in fragment_corrections:
            df_packets.at[i, "timestamp"] = fragment_corrections[row["timestamp"]]

    df_packets = df_packets.sort_values("timestamp").reset_index(drop=True)

    # Find IKE_SA_INIT packets - these mark handshake boundaries
    ike_sa_init_packets = df_packets[df_packets["exchange_type"] == "ike_sa_init"]
    log_message(f"Found {len(ike_sa_init_packets)} IKE_SA_INIT packets")

    if len(ike_sa_init_packets) == 0:
        log_message("Error: No IKE_SA_INIT packets found")
        return []

    # Expected: 2 IKE_SA_INIT per iteration (request + response)
    expected_sa_init_count = iters * 2
    actual_sa_init_count = len(ike_sa_init_packets)

    log_message(
        f"Expected {expected_sa_init_count} IKE_SA_INIT packets for {iters} iterations, found {actual_sa_init_count}"
    )

    # Calculate actual iterations based on packet pairs
    actual_iterations = min(iters, actual_sa_init_count // 2)
    if actual_iterations == 0:
        log_message("Error: No complete iterations found")
        return []

    latencies = []
    ike_sa_init_timestamps = ike_sa_init_packets["timestamp"].values

    # Process each iteration (pair of IKE_SA_INIT packets)
    for iter_num in range(actual_iterations):
        iter_start_time = ike_sa_init_timestamps[
            iter_num * 2
        ]  # First IKE_SA_INIT of pair

        # Define iteration boundary
        if iter_num < actual_iterations - 1:
            iter_end_boundary = ike_sa_init_timestamps[
                (iter_num + 1) * 2
            ]  # Next iteration start
        else:
            iter_end_boundary = float("inf")  # Last iteration

        log_message(
            f"Processing iteration {iter_num + 1}: start={iter_start_time:.6f}, boundary={iter_end_boundary}"
        )

        # Get all packets in this iteration
        iter_packets = df_packets[
            (df_packets["timestamp"] >= iter_start_time)
            & (df_packets["timestamp"] < iter_end_boundary)
        ]

        log_message(f"  Iteration {iter_num + 1} has {len(iter_packets)} packets")

        # Find first IKE_AUTH in this iteration
        iter_ike_auth = iter_packets[iter_packets["exchange_type"] == "ike_auth"]

        if len(iter_ike_auth) > 0:
            first_ike_auth = iter_ike_auth["timestamp"].min()
            log_message(f"  Found IKE_AUTH at {first_ike_auth:.6f}")

            # Get all packets before first IKE_AUTH
            pre_auth_packets = iter_packets[iter_packets["timestamp"] < first_ike_auth]

            if len(pre_auth_packets) > 0:
                end_time = pre_auth_packets["timestamp"].max()
                exchanges = pre_auth_packets["exchange_type"].value_counts()
                log_message(f"  Pre-AUTH exchanges: {dict(exchanges)}")
            else:
                log_message(f"  Warning: No pre-AUTH packets found")
                end_time = iter_start_time
        else:
            log_message(f"  No IKE_AUTH found in iteration, using all packets")
            end_time = (
                iter_packets["timestamp"].max()
                if len(iter_packets) > 0
                else iter_start_time
            )

        latency = end_time - iter_start_time
        latencies.append(latency)
        log_message(
            f"  Iteration {iter_num + 1}: {latency*1000:.2f}ms (start: {iter_start_time:.6f}, end: {end_time:.6f})"
        )

    log_message(f"Calculated {len(latencies)} handshake latencies")

    # Save results
    if latencies:
        summary_data = [
            {
                "iteration": i + 1,
                "handshake_latency_seconds": lat,
                "handshake_latency_ms": lat * 1000,
            }
            for i, lat in enumerate(latencies)
        ]
        pd.DataFrame(summary_data).to_csv(filename, index=False)
        log_message(f"Saved {len(summary_data)} measurements to {filename}")
    else:
        log_message("No latencies calculated - creating empty file")
        pd.DataFrame(
            columns=["iteration", "handshake_latency_seconds", "handshake_latency_ms"]
        ).to_csv(filename, index=False)

    return latencies


def analyze_packets(packets, prop, log_message, total_pcap_bytes=0, iters=1):
    """Analyze packet counts and bytes"""
    if not packets:
        log_message("No packets to analyze")
        return create_empty_combined_stats(prop)

    # Count packets by exchange type
    exchange_counts = defaultdict(int)
    exchange_bytes = defaultdict(int)
    total_ike_bytes = 0
    fragmented_count = 0

    for packet in packets:
        exchange_type = packet["exchange_type"]
        # Use frame_length
        ike_size = packet["frame_length"]

        exchange_counts[exchange_type] += 1
        exchange_bytes[exchange_type] += ike_size
        total_ike_bytes += ike_size

        if packet.get("fragmented", False):
            fragmented_count += 1

    # Normalize by iterations
    total_ike_bytes = total_ike_bytes // iters
    total_pcap_bytes = total_pcap_bytes // iters
    total_message_count = len(packets) // iters
    fragmented_count = fragmented_count // iters

    # Log analysis results
    log_message(f"Analysis complete: {total_message_count} IKE messages per iteration")
    log_message(f"Total TRUE IKE message bytes per iteration: {total_ike_bytes:,}")

    if fragmented_count > 0:
        log_message(
            f"Fragmented messages: {fragmented_count}/{total_message_count} ({fragmented_count/total_message_count*100:.1f}%)"
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

    # Log exchange breakdown with TRUE sizes
    log_message("Exchange type breakdown (TRUE IKE message sizes per iteration):")
    for exchange_type, count in exchange_counts.items():
        norm_count = count // iters
        norm_bytes = exchange_bytes[exchange_type] // iters
        if norm_count > 0:
            avg_size = norm_bytes / norm_count
            log_message(
                f"  {exchange_type}: {norm_count} messages, {norm_bytes:,} bytes (avg: {avg_size:.1f}B/msg)"
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
        "fragmented_message_count": fragmented_count,
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
        "fragmented_message_count": 0,
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
        packets,
        f"{output_dir}/handshake_summary_{prop}.csv",
        log_message,
        iters,
        pcap_file,
    )

    # Update combined stats with latency data
    update_stats_with_ike_latencies(combined_stats, ike_handshake_latencies)

    # Log final results
    ike_latency_ms = combined_stats.get("ike_latency_avg", 0) * 1000
    ike_latency_std_ms = combined_stats.get("ike_latency_std", 0) * 1000
    log_message(
        f"Results for {prop}: {combined_stats['total_message_count']} IKE messages per iteration, "
        f"{combined_stats['total_ike_bytes']:,} IKE bytes per iteration, "
        f"{combined_stats['total_pcap_bytes']:,} total bytes per iteration, "
        f"IKE handshake latency: {ike_latency_ms:.2f}±{ike_latency_std_ms:.2f}ms"
    )

    return combined_stats, {}


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

            f.write(f"Total IKE Messages (per iteration): {total_packets}\n")
            f.write(f"Total IKE Message Bytes (per iteration): {total_ike_bytes:,}\n")
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
                    avg_size = bytes_val / count if count > 0 else 0
                    f.write(
                        f"  {display_name}: {count} messages, {bytes_val:,} bytes (avg: {avg_size:.1f} bytes/msg)\n"
                    )

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
                f"Total IKE message bytes processed (per iteration): {summary_stats['total_ike_bytes']:,}\n"
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
