#!/usr/bin/env python3
# Copyright (C) 2024-2025 Javier Blanco-Romero @fj-blanco (UC3M, QURSA project)
#
# PCAP processing

import subprocess
import pandas as pd
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


def calculate_latencies_and_save(packets, filename, log_message):
    """Calculate latencies and save detailed CSV"""
    if not packets:
        log_message("No packets for latency calculation")
        return {}

    # Group packets by exchange type and message_id (SINGLE grouping operation)
    exchange_groups = defaultdict(list)
    for packet in packets:
        key = (packet["exchange_type"], packet["message_id"])
        exchange_groups[key].append(packet)

    latency_results = {}
    latency_data = []  # For CSV output

    for (exchange_type, message_id), group_packets in exchange_groups.items():
        if len(group_packets) < 2:
            continue

        # Sort by timestamp
        group_packets.sort(key=lambda x: x["timestamp"])

        # Calculate latencies for pairs (SINGLE pairing operation)
        latencies = []
        for i in range(0, len(group_packets) - 1, 2):
            if i + 1 < len(group_packets):
                req_time = group_packets[i]["timestamp"]
                resp_time = group_packets[i + 1]["timestamp"]
                latency = resp_time - req_time
                latencies.append(latency)

                # Add to CSV data while we're here
                latency_data.append(
                    {
                        "exchange_type": exchange_type,
                        "pair_number": (i // 2) + 1,
                        "latency_seconds": latency,
                        "latency_ms": latency * 1000,
                        "request_timestamp": req_time,
                        "response_timestamp": resp_time,
                        "message_id": message_id,
                    }
                )

        # Calculate statistics
        if latencies:
            if exchange_type not in latency_results:
                latency_results[exchange_type] = {
                    "latencies": [],
                    "latency_avg": 0,
                    "latency_min": 0,
                    "latency_max": 0,
                    "latency_std": 0,
                    "count": 0,
                }

            existing_latencies = latency_results[exchange_type]["latencies"]
            all_latencies = existing_latencies + latencies
            avg = sum(all_latencies) / len(all_latencies)
            std = (
                (sum((x - avg) ** 2 for x in all_latencies) / len(all_latencies)) ** 0.5
                if len(all_latencies) > 1
                else 0
            )

            latency_results[exchange_type] = {
                "latencies": all_latencies,
                "latency_avg": avg,
                "latency_min": min(all_latencies),
                "latency_max": max(all_latencies),
                "latency_std": std,
                "count": len(all_latencies),
            }

    # Save CSV file
    if latency_data:
        pd.DataFrame(latency_data).to_csv(filename, index=False)
        log_message(f"Saved {len(latency_data)} latency measurements to {filename}")

        # Log summary
        for exchange_type in set(d["exchange_type"] for d in latency_data):
            exchange_data = [
                d for d in latency_data if d["exchange_type"] == exchange_type
            ]
            avg_ms = sum(d["latency_ms"] for d in exchange_data) / len(exchange_data)
            log_message(
                f"  {exchange_type}: {len(exchange_data)} pairs, avg {avg_ms:.2f}ms"
            )
    else:
        log_message("No latency pairs created")

    return latency_results


def analyze_packets(packets, prop, log_message):
    """Simplified analysis with single latency calculation"""
    if not packets:
        log_message("No packets to analyze")
        return create_empty_combined_stats(prop), {}

    # Count packets by exchange type
    exchange_counts = defaultdict(int)
    exchange_bytes = defaultdict(int)
    total_bytes = 0

    for packet in packets:
        exchange_type = packet["exchange_type"]
        frame_length = packet["frame_length"]
        exchange_counts[exchange_type] += 1
        exchange_bytes[exchange_type] += frame_length
        total_bytes += frame_length

    # Log what we found
    log_message(f"Analysis complete: {len(packets)} packets, {total_bytes} bytes")
    for exchange_type, count in exchange_counts.items():
        log_message(
            f"  {exchange_type}: {count} packets, {exchange_bytes[exchange_type]} bytes"
        )

    # Calculate total stats from exchange latencies
    total_latency_avg = total_latency_std = 0

    # Create combined stats
    combined_stats = {
        "proposal": prop,
        "total_ike_bytes": total_bytes,
        "total_message_count": len(packets),
        "total_latency_avg": total_latency_avg,
        "total_latency_std": total_latency_std,
    }

    # Add exchange stats
    exchanges = [
        "ike_sa_init",
        "ike_auth",
        "ike_intermediate",
        "informational",
        "create_child_sa",
    ]
    for exchange in exchanges:
        count = exchange_counts.get(exchange, 0)
        bytes_val = exchange_bytes.get(exchange, 0)
        combined_stats.update(
            {
                f"{exchange}_count": count,
                f"{exchange}_bytes": bytes_val,
                f"{exchange}_latency_avg": 0,
                f"{exchange}_latency_min": 0,
                f"{exchange}_latency_max": 0,
                f"{exchange}_latency_std": 0,
                f"{exchange}_latency_count": 0,
            }
        )

    return combined_stats


def update_stats_with_latencies(combined_stats, latency_results):
    """Update combined stats with calculated latencies"""
    if latency_results:
        valid_latencies = [
            data["latency_avg"]
            for data in latency_results.values()
            if data["latency_avg"] > 0
        ]
        if valid_latencies:
            total_avg = sum(valid_latencies) / len(valid_latencies)
            total_std = (
                (
                    sum((x - total_avg) ** 2 for x in valid_latencies)
                    / len(valid_latencies)
                )
                ** 0.5
                if len(valid_latencies) > 1
                else 0
            )
            combined_stats["total_latency_avg"] = total_avg
            combined_stats["total_latency_std"] = total_std

    # Update individual exchange stats
    for exchange_type, latency_data in latency_results.items():
        combined_stats[f"{exchange_type}_latency_avg"] = latency_data.get(
            "latency_avg", 0
        )
        combined_stats[f"{exchange_type}_latency_min"] = latency_data.get(
            "latency_min", 0
        )
        combined_stats[f"{exchange_type}_latency_max"] = latency_data.get(
            "latency_max", 0
        )
        combined_stats[f"{exchange_type}_latency_std"] = latency_data.get(
            "latency_std", 0
        )
        combined_stats[f"{exchange_type}_latency_count"] = latency_data.get("count", 0)


def create_empty_combined_stats(prop):
    """Create empty combined stats structure"""
    combined_stats = {
        "proposal": prop,
        "total_ike_bytes": 0,
        "total_message_count": 0,
        "total_latency_avg": 0,
        "total_latency_std": 0,
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
                f"{exchange}_latency_avg": 0,
                f"{exchange}_latency_min": 0,
                f"{exchange}_latency_max": 0,
                f"{exchange}_latency_std": 0,
                f"{exchange}_latency_count": 0,
            }
        )

    return combined_stats


def process_ike_data(pcap_file, output_dir, prop, log_message):
    """main processing function"""
    log_message(f"Processing PCAP for proposal: {prop}")

    # Extract packets
    packets = extract_ike_packets(pcap_file, log_message)

    # Analyze packets (basic stats only)
    combined_stats = analyze_packets(packets, prop, log_message)

    # Calculate latencies AND save CSV in one operation
    latency_results = calculate_latencies_and_save(
        packets, f"{output_dir}/detailed_latencies_{prop}.csv", log_message
    )

    # Update combined stats with latency data
    update_stats_with_latencies(combined_stats, latency_results)

    # Simple counters for backward compatibility
    exchanges = [
        "ike_sa_init",
        "ike_intermediate",
        "ike_auth",
        "informational",
        "create_child_sa",
    ]

    # Log results
    total_latency_ms = combined_stats.get("total_latency_avg", 0) * 1000
    total_latency_std_ms = combined_stats.get("total_latency_std", 0) * 1000
    log_message(
        f"Results for {prop}: {combined_stats['total_message_count']} packets, "
        f"{combined_stats['total_ike_bytes']:,} bytes, "
        f"{total_latency_ms:.2f}±{total_latency_std_ms:.2f}ms real avg latency"
    )

    return combined_stats, latency_results


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
            total_bytes = row.get("total_ike_bytes", 0)
            total_latency_avg = row.get("total_latency_avg", 0)
            total_latency_std = row.get("total_latency_std", 0)

            f.write(f"Total IKE Packets: {total_packets}\n")
            f.write(f"Total IKE Bytes: {total_bytes:,}\n")
            f.write(
                f"Total Average Latency: {total_latency_avg:.6f} ± {total_latency_std:.6f} seconds "
                f"({total_latency_avg*1000:.2f} ± {total_latency_std*1000:.2f} ms)\n\n"
            )

            # Exchange breakdown
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
                    latency_avg = row.get(f"{exchange}_latency_avg", 0)
                    latency_std = row.get(f"{exchange}_latency_std", 0)
                    latency_count = row.get(f"{exchange}_latency_count", 0)

                    f.write(f"  {display_name}: {count} packets, {bytes_val:,} bytes")
                    if latency_count > 0:
                        f.write(
                            f", {latency_count} req/resp pairs, avg {latency_avg*1000:.2f}±{latency_std*1000:.2f}ms"
                        )
                    f.write("\n")

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

            # Collect detailed latency data
            detailed_file = f"{output_dir}/detailed_latencies_{prop}.csv"
            try:
                if os.path.exists(detailed_file):
                    detail_df = pd.read_csv(detailed_file)
                    if not detail_df.empty:
                        if "proposal" not in detail_df.columns:
                            detail_df["proposal"] = prop
                        all_latency_measurements.append(detail_df)
                        log_message(
                            f"Collected {len(detail_df)} detailed measurements for {prop}"
                        )
                else:
                    log_message(f"No detailed latencies file found for {prop}")
            except Exception as e:
                log_message(f"Error reading detailed latencies for {prop}: {e}")

        # Overall summary
        if not df_bytes.empty:
            f.write("OVERALL SUMMARY\n===============\n\n")
            total_bytes_all = df_bytes["total_ike_bytes"].sum()
            total_msgs_all = df_bytes["total_message_count"].sum()
            latency_avg_all = df_bytes["total_latency_avg"].mean()
            latency_std_all = df_bytes["total_latency_std"].mean()

            f.write(f"Total bytes processed: {total_bytes_all:,}\n")
            f.write(f"Total messages processed: {total_msgs_all}\n")
            f.write(
                f"Average handshake latency across all proposals: {latency_avg_all*1000:.2f} ± {latency_std_all*1000:.2f} ms\n"
            )

    # Save aggregated measurements
    if all_latency_measurements:
        aggregated_df = pd.concat(all_latency_measurements, ignore_index=True)
        aggregated_file = f"{output_dir}/pcap_aggregated_measurements.csv"
        aggregated_df.to_csv(aggregated_file, index=False)
        proposal_counts = aggregated_df["proposal"].value_counts()
        log_message(
            f"Aggregated {len(aggregated_df)} latency measurements from {len(proposal_counts)} proposals"
        )
        log_message(f"Aggregated measurements saved to: {aggregated_file}")
    else:
        log_message("No detailed latency measurements found to aggregate")

    log_message(f"Complete report generated: {output_dir}/report.txt")
