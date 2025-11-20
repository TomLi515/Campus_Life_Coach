"""
Convert .xls files with separate accelerometer/gyroscope sheets into unified CSVs.

Usage:
    python convert_xls_to_unified_csv.py --input_dir ./raw_data/user1 \
                                          --output_dir ./processed_data/user1 \
                                          --device phone \
                                          --subject user1
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import re


def extract_activity_from_filename(filename):
    """Extract activity label from filename like 'walk1.xls' or 'sit_trial2.xls'."""
    filename = filename.lower().replace(".xls", "").replace(".xlsx", "")

    # Match common patterns
    for activity in ["walk", "run", "sit", "stand", "lie", "lying"]:
        if activity in filename:
            if activity == "lying":
                return "lie"
            return activity

    raise ValueError(f"Could not extract activity from filename: {filename}")


def convert_xls_to_unified_csv(xls_path, device, subject, output_path):
    """
    Convert .xls with multiple sheets to single CSV with all IMU data.

    Args:
        xls_path: Path to .xls file
        device: "phone" or "watch"
        subject: Subject ID (e.g., "user1")
        output_path: Path for output CSV
    """
    # Read accelerometer sheet (sheet 0)
    acc_df = pd.read_excel(xls_path, sheet_name=0)

    # Read gyroscope sheet (sheet 1)
    gyro_df = pd.read_excel(xls_path, sheet_name=1)

    # Handle different possible column name formats
    acc_time_col = [c for c in acc_df.columns if "time" in c.lower()][0]
    gyro_time_col = [c for c in gyro_df.columns if "time" in c.lower()][0]

    # Extract X, Y, Z columns (should be in order after Time)
    acc_cols = [c for c in acc_df.columns if c != acc_time_col][:3]
    gyro_cols = [c for c in gyro_df.columns if c != gyro_time_col][:3]

    # Handle mismatched lengths by truncating to shorter length
    min_len = min(len(acc_df), len(gyro_df))

    if len(acc_df) != len(gyro_df):
        print(
            f"  ⚠️ Length mismatch: acc={len(acc_df)}, gyro={len(gyro_df)}, using first {min_len} samples"
        )

    # Merge on time (assuming timestamps align)
    merged_df = pd.DataFrame(
        {
            "Time": acc_df[acc_time_col].values[:min_len],
            "Acc_X": acc_df[acc_cols[0]].values[:min_len],
            "Acc_Y": acc_df[acc_cols[1]].values[:min_len],
            "Acc_Z": acc_df[acc_cols[2]].values[:min_len],
            "Gyro_X": gyro_df[gyro_cols[0]].values[:min_len],
            "Gyro_Y": gyro_df[gyro_cols[1]].values[:min_len],
            "Gyro_Z": gyro_df[gyro_cols[2]].values[:min_len],
        }
    )

    # Add metadata
    activity = extract_activity_from_filename(xls_path.name)
    merged_df["Device"] = device
    merged_df["Subject"] = subject
    merged_df["Activity"] = activity

    # Save to CSV
    merged_df.to_csv(output_path, index=False)
    print(f"✓ Converted {xls_path.name} → {output_path.name}")

    return merged_df


def convert_separate_csvs_to_unified(
    acc_csv_path, gyro_csv_path, device, subject, output_path
):
    """
    Convert separate accelerometer and gyroscope CSVs into unified format.

    Args:
        acc_csv_path: Path to accelerometer CSV
        gyro_csv_path: Path to gyroscope CSV
        device: "phone" or "watch"
        subject: Subject ID
        output_path: Path for output CSV
    """
    acc_df = pd.read_csv(acc_csv_path)
    gyro_df = pd.read_csv(gyro_csv_path)

    # Extract time columns
    acc_time_col = [c for c in acc_df.columns if "time" in c.lower()][0]
    gyro_time_col = [c for c in gyro_df.columns if "time" in c.lower()][0]

    # Extract X, Y, Z data columns
    acc_cols = [c for c in acc_df.columns if c != acc_time_col][:3]
    gyro_cols = [c for c in gyro_df.columns if c != gyro_time_col][:3]

    # Merge
    merged_df = pd.DataFrame(
        {
            "Time": acc_df[acc_time_col].values,
            "Acc_X": acc_df[acc_cols[0]].values,
            "Acc_Y": acc_df[acc_cols[1]].values,
            "Acc_Z": acc_df[acc_cols[2]].values,
            "Gyro_X": gyro_df[gyro_cols[0]].values,
            "Gyro_Y": gyro_df[gyro_cols[1]].values,
            "Gyro_Z": gyro_df[gyro_cols[2]].values,
        }
    )

    # Add metadata
    activity = extract_activity_from_filename(acc_csv_path.name)
    merged_df["Device"] = device
    merged_df["Subject"] = subject
    merged_df["Activity"] = activity

    merged_df.to_csv(output_path, index=False)
    print(
        f"✓ Converted {acc_csv_path.name} + {gyro_csv_path.name} → {output_path.name}"
    )

    return merged_df


def batch_convert_xls_directory(input_dir, output_dir, device, subject):
    """Convert all .xls files in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    xls_files = list(input_path.glob("*.xls")) + list(input_path.glob("*.xlsx"))

    print(f"\nConverting {len(xls_files)} files from {input_dir}...")
    print(f"Device: {device} | Subject: {subject}\n")

    for xls_file in sorted(xls_files):
        output_csv = output_path / f"{xls_file.stem}.csv"
        try:
            convert_xls_to_unified_csv(xls_file, device, subject, output_csv)
        except Exception as e:
            print(f"✗ Error converting {xls_file.name}: {e}")

    print(f"\n✓ Conversion complete! Files saved to {output_dir}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert .xls HAR data to unified CSV format"
    )
    parser.add_argument(
        "--input_dir", required=True, help="Directory containing .xls files"
    )
    parser.add_argument("--output_dir", required=True, help="Directory for output CSVs")
    parser.add_argument(
        "--device",
        required=True,
        choices=["phone", "watch"],
        help="Device type (phone or watch)",
    )
    parser.add_argument("--subject", required=True, help="Subject ID (e.g., user1)")

    args = parser.parse_args()

    batch_convert_xls_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        device=args.device,
        subject=args.subject,
    )


if __name__ == "__main__":
    main()
