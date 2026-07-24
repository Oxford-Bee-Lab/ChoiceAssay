from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from choice_assay.etl.rerun_processing import detect_default_roots, run_rerun_processing


def normalize_id(value: str) -> str:
    """Normalize IDs by lowercasing and removing non-hex characters."""
    return re.sub(r"[^0-9a-fA-F]", "", value).lower()


def read_interface_mac(interface: str) -> str:
    """Read MAC address for a network interface and return it without separators."""
    addr_path = Path("/sys/class/net") / interface / "address"
    if not addr_path.exists():
        msg = (
            f"Could not find interface '{interface}' at {addr_path}. "
            "Use --reprocessor-id to provide an explicit reprocessor ID."
        )
        raise RuntimeError(msg)

    mac_raw = addr_path.read_text(encoding="utf-8").strip()
    mac = normalize_id(mac_raw)
    if not mac:
        msg = (
            f"Failed to parse MAC from '{mac_raw}' for interface '{interface}'. "
            "Use --reprocessor-id to provide an explicit reprocessor ID."
        )
        raise RuntimeError(msg)

    return mac


def load_assigned_device_ids(assignments_csv: Path, reprocessor_id: str) -> list[str]:
    """Return device IDs assigned to this reprocessor."""
    if not assignments_csv.exists():
        raise FileNotFoundError(f"Assignments file not found: {assignments_csv}")

    df = pd.read_csv(assignments_csv)
    required_columns = {"device_id", "reprocessor"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Assignments file missing required columns: {sorted(missing)}")

    working = df.loc[:, ["device_id", "reprocessor"]].copy()
    working["device_id"] = working["device_id"].fillna("").astype(str).str.strip().str.lower()
    working["reprocessor"] = working["reprocessor"].fillna("").astype(str).map(normalize_id)

    assigned = working.loc[
        (working["reprocessor"] == reprocessor_id) & (working["device_id"] != ""),
        "device_id",
    ]
    return sorted(set(assigned.tolist()))


def parse_args() -> argparse.Namespace:
    nas_root, _ = detect_default_roots()
    default_video_src_dir = nas_root / "azure" / "choice_assay" / "expidite-choiceassay-trapcam"
    default_output_dir = nas_root / "results" / "choice_assay_rerun"
    default_files_list = Path(__file__).with_name("files_to_process.csv")
    default_assignments = Path(__file__).with_name("assignments.csv")

    parser = argparse.ArgumentParser(
        description=(
            "Run rerun processing only for device IDs assigned to this reprocessor "
            "(identified by wlan0 MAC with separators removed)."
        )
    )
    parser.add_argument("--assignments-csv", type=Path, default=default_assignments)
    parser.add_argument("--files-to-process", type=Path, default=default_files_list)
    parser.add_argument("--video-src-dir", type=Path, default=default_video_src_dir)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument(
        "--interface",
        type=str,
        default="wlan0",
        help="Network interface used to detect this Pi's MAC address.",
    )
    parser.add_argument(
        "--reprocessor-id",
        type=str,
        default=None,
        help=(
            "Optional explicit reprocessor ID (MAC with or without separators). "
            "If omitted, derived from --interface."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print assigned device IDs without running processing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    reprocessor_id = read_interface_mac(args.interface)
    if not reprocessor_id:
        raise ValueError("Resolved reprocessor ID is empty. Provide a valid --reprocessor-id.")

    assigned_device_ids = load_assigned_device_ids(args.assignments_csv, reprocessor_id)

    print(f"Reprocessor ID: {reprocessor_id}")
    print(f"Assignments file: {args.assignments_csv.resolve()}")
    print(f"Assigned device_ids: {len(assigned_device_ids)}")

    if not assigned_device_ids:
        print("No device assignments found for this reprocessor. Nothing to do.")
        return

    for device_id in assigned_device_ids:
        print(f"  - {device_id}")

    if args.dry_run:
        print("Dry run enabled; skipping processing.")
        return

    totals = {
        "processed_success": 0,
        "missing_in_prefetch": 0,
        "missing_in_processing": 0,
        "queued": 0,
    }

    for device_id in assigned_device_ids:
        print(f"\nStarting rerun for device_id='{device_id}'")
        stats = run_rerun_processing(
            files_to_process=args.files_to_process,
            video_src_dir=args.video_src_dir,
            output_dir=args.output_dir,
            file_filter=device_id,
        )
        for key in totals:
            totals[key] += stats.get(key, 0)

    print("\nAll assigned reruns completed.")
    print(
        "Totals across all assigned device_ids: "
        f"processed_success={totals['processed_success']}, "
        f"missing_in_prefetch={totals['missing_in_prefetch']}, "
        f"missing_in_processing={totals['missing_in_processing']}, "
        f"queued={totals['queued']}"
    )


if __name__ == "__main__":
    main()
