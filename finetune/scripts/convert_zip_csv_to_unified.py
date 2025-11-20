"""
Convert zip files containing separate accelerometer.csv and gyroscope.csv 
into unified CSV format matching user1's structure.

Each zip is named like: walk_1.zip, sit_2.zip, etc.
Inside each zip: accelerometer.csv, gyroscope.csv, meta/ (ignored)

Usage:
    python convert_zip_csv_to_unified.py --input_dir ./user2/phone/zip_files \
                                          --output_dir ./user2/phone/unified_csv \
                                          --device phone \
                                          --subject user2
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import zipfile
import re
import tempfile
import os


def extract_activity_from_zipname(zip_name):
    """Extract activity from zip filename like 'walk_1.zip' or 'Sit_2.zip'."""
    # Remove .zip extension and convert to lowercase
    name = zip_name.lower().replace('.zip', '')
    
    # Extract activity name (everything before the last underscore and number)
    # Handles: walk_1, Walk_1, sitting_trial_2, etc.
    for activity in ['walk', 'run', 'sit', 'stand', 'lie', 'lying']:
        if name.startswith(activity):
            if activity == 'lying':
                return 'lie'
            return activity
    
    raise ValueError(f"Could not extract activity from zip filename: {zip_name}")


def convert_zip_to_unified_csv(zip_path, device, subject, output_path):
    """
    Extract accelerometer.csv and gyroscope.csv from zip and merge them.
    
    Args:
        zip_path: Path to .zip file
        device: "phone" or "watch"
        subject: Subject ID (e.g., "user2")
        output_path: Path for output CSV
    """
    
    # Create a temporary directory to extract files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find accelerometer.csv and gyroscope.csv (case-insensitive)
        temp_path = Path(temp_dir)
        acc_file = None
        gyro_file = None
        
        # Search recursively for the CSV files
        for file in temp_path.rglob("*"):
            if file.is_file() and 'accelerometer' in file.name.lower():
                acc_file = file
            elif file.is_file() and 'gyroscope' in file.name.lower():
                gyro_file = file
        
        if acc_file is None or gyro_file is None:
            raise FileNotFoundError(f"Could not find accelerometer.csv or gyroscope.csv in {zip_path.name}")
        
        # Read the CSV files
        acc_df = pd.read_csv(acc_file)
        gyro_df = pd.read_csv(gyro_file)
    
    # Handle different possible column name formats
    # Common formats: 'Time (s)', 'time', 'timestamp', 'Time'
    acc_time_col = [c for c in acc_df.columns if 'time' in c.lower()][0]
    gyro_time_col = [c for c in gyro_df.columns if 'time' in c.lower()][0]
    
    # Extract X, Y, Z columns (everything except time column, take first 3)
    acc_cols = [c for c in acc_df.columns if c != acc_time_col][:3]
    gyro_cols = [c for c in gyro_df.columns if c != gyro_time_col][:3]
    
    # Handle mismatched lengths
    if len(acc_df) != len(gyro_df):
        print(f"  ⚠️ Length mismatch: acc={len(acc_df)}, gyro={len(gyro_df)}")
        
        # Use time-based interpolation for better alignment
        if len(acc_df) > len(gyro_df):
            common_time = acc_df[acc_time_col].values
            # Interpolate gyroscope to match accelerometer timestamps
            gyro_interp = pd.DataFrame({
                gyro_time_col: common_time
            })
            for col in gyro_cols:
                gyro_interp[col] = np.interp(common_time, 
                                             gyro_df[gyro_time_col].values,
                                             gyro_df[col].values)
            acc_aligned = acc_df
            gyro_aligned = gyro_interp
        else:
            common_time = gyro_df[gyro_time_col].values
            # Interpolate accelerometer to match gyroscope timestamps
            acc_interp = pd.DataFrame({
                acc_time_col: common_time
            })
            for col in acc_cols:
                acc_interp[col] = np.interp(common_time,
                                           acc_df[acc_time_col].values,
                                           acc_df[col].values)
            acc_aligned = acc_interp
            gyro_aligned = gyro_df
        
        print(f"  ✓ Interpolated to {len(common_time)} samples")
    else:
        acc_aligned = acc_df
        gyro_aligned = gyro_df
        common_time = acc_df[acc_time_col].values
    
    # Create unified dataframe
    merged_df = pd.DataFrame({
        'Time': common_time,
        'Acc_X': acc_aligned[acc_cols[0]].values,
        'Acc_Y': acc_aligned[acc_cols[1]].values,
        'Acc_Z': acc_aligned[acc_cols[2]].values,
        'Gyro_X': gyro_aligned[gyro_cols[0]].values,
        'Gyro_Y': gyro_aligned[gyro_cols[1]].values,
        'Gyro_Z': gyro_aligned[gyro_cols[2]].values,
    })
    
    # Add metadata
    activity = extract_activity_from_zipname(zip_path.name)
    merged_df['Device'] = device
    merged_df['Subject'] = subject
    merged_df['Activity'] = activity
    
    # Save to CSV with descriptive name matching the zip file
    merged_df.to_csv(output_path, index=False)
    print(f"✓ Converted {zip_path.name} → {output_path.name}")
    
    return merged_df


def batch_convert_zip_directory(input_dir, output_dir, device, subject):
    """Convert all zip files in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    zip_files = list(input_path.glob("*.zip"))
    
    if len(zip_files) == 0:
        print(f"⚠️ No .zip files found in {input_dir}")
        return
    
    print(f"\nConverting {len(zip_files)} zip files from {input_dir}...")
    print(f"Device: {device} | Subject: {subject}\n")
    
    successful = 0
    failed = 0
    
    for zip_file in sorted(zip_files):
        # Create output filename: walk_1.zip → walk_1.csv
        output_csv = output_path / f"{zip_file.stem}.csv"
        
        try:
            convert_zip_to_unified_csv(zip_file, device, subject, output_csv)
            successful += 1
        except Exception as e:
            print(f"✗ Error converting {zip_file.name}: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"✓ Conversion complete!")
    print(f"  Successful: {successful}/{len(zip_files)}")
    print(f"  Failed: {failed}/{len(zip_files)}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert zip files with separate accelerometer/gyroscope CSVs to unified format"
    )
    parser.add_argument("--input_dir", required=True, 
                        help="Directory containing .zip files (e.g., walk_1.zip, sit_2.zip)")
    parser.add_argument("--output_dir", required=True, 
                        help="Directory for output unified CSVs")
    parser.add_argument("--device", required=True, choices=["phone", "watch"], 
                        help="Device type (phone or watch)")
    parser.add_argument("--subject", required=True, 
                        help="Subject ID (e.g., user2)")
    
    args = parser.parse_args()
    
    batch_convert_zip_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        device=args.device,
        subject=args.subject
    )


if __name__ == "__main__":
    main()