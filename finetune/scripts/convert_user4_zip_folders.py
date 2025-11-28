"""
Convert user4's zip file containing folders (walk_1, walk_2, etc.) 
with separate accelerometer.csv and gyroscope.csv into unified CSV format.

Each folder structure:
    walk_1/
        accelerometer.csv
        gyroscope.csv
    walk_2/
        accelerometer.csv
        gyroscope.csv
    ...

Usage:
    python convert_user4_zip_folders.py --zip_file ../data/user4/data.zip \
                                         --output_dir ../data/user4/phone/unified_csv \
                                         --device phone \
                                         --subject user4
                                         
    Or if already extracted:
    python convert_user4_zip_folders.py --input_dir ../data/user4/extracted_folders \
                                         --output_dir ../data/user4/phone/unified_csv \
                                         --device phone \
                                         --subject user4
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import zipfile
import tempfile
import shutil
import re


def extract_activity_and_instance(folder_name):
    """
    Extract activity and instance from folder name.
    
    Examples:
        walk_1 → (walk, 1)
        run_2 → (run, 2)
        sit_10 → (sit, 10)
        stand_3 → (stand, 3)
        lie_5 → (lie, 5)
    """
    # Convert to lowercase
    name = folder_name.lower()
    
    # Try pattern: activity_number
    match = re.match(r'([a-z]+)_(\d+)', name)
    if match:
        activity = match.group(1)
        instance = int(match.group(2))
        
        # Normalize activity names
        if activity in ['lying', 'lay']:
            activity = 'lie'
        
        return activity, instance
    
    raise ValueError(f"Could not extract activity and instance from folder: {folder_name}")


def convert_folder_to_unified_csv(folder_path, device, subject, output_path):
    """
    Read accelerometer.csv and gyroscope.csv from folder and merge them.
    
    Args:
        folder_path: Path to folder containing accelerometer.csv and gyroscope.csv
        device: "phone" or "watch"
        subject: Subject ID (e.g., "user4")
        output_path: Path for output CSV
    """
    
    # Find accelerometer.csv and gyroscope.csv
    acc_file = folder_path / "accelerometer.csv"
    gyro_file = folder_path / "gyroscope.csv"
    
    if not acc_file.exists():
        raise FileNotFoundError(f"accelerometer.csv not found in {folder_path}")
    
    if not gyro_file.exists():
        raise FileNotFoundError(f"gyroscope.csv not found in {folder_path}")
    
    # Read CSV files
    acc_df = pd.read_csv(acc_file)
    gyro_df = pd.read_csv(gyro_file)
    
    # Handle different possible column name formats
    # Common formats: 'Time (s)', 'time', 'timestamp', 'Time', 'seconds_elapsed'
    acc_time_col = None
    gyro_time_col = None
    
    for col in acc_df.columns:
        if 'time' in col.lower():
            acc_time_col = col
            break
    
    for col in gyro_df.columns:
        if 'time' in col.lower():
            gyro_time_col = col
            break
    
    if acc_time_col is None:
        raise ValueError(f"No time column found in accelerometer.csv")
    
    if gyro_time_col is None:
        raise ValueError(f"No time column found in gyroscope.csv")
    
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
    
    # Extract activity and instance from folder name
    activity, instance = extract_activity_and_instance(folder_path.name)
    
    # Add metadata
    merged_df['Device'] = device
    merged_df['Subject'] = subject
    merged_df['Activity'] = activity
    
    # Save to CSV
    merged_df.to_csv(output_path, index=False)
    
    print(f"✓ Converted {folder_path.name} → {output_path.name} ({len(merged_df)} samples)")
    
    return merged_df


def process_zip_file(zip_path, output_dir, device, subject):
    """
    Extract zip file to temporary directory and process all folders.
    """
    print(f"Extracting zip file: {zip_path}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract zip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        temp_path = Path(temp_dir)
        
        # Find all subdirectories (activity folders)
        # Handle cases where zip might have a root folder
        folders = []
        for item in temp_path.rglob("*"):
            if item.is_dir():
                # Check if this folder contains accelerometer.csv and gyroscope.csv
                if (item / "accelerometer.csv").exists() and (item / "gyroscope.csv").exists():
                    folders.append(item)
        
        if len(folders) == 0:
            print("⚠️ No folders with accelerometer.csv and gyroscope.csv found in zip")
            return
        
        print(f"Found {len(folders)} activity folders\n")
        
        # Process each folder
        process_folders(folders, output_dir, device, subject)


def process_folders(folders, output_dir, device, subject):
    """
    Process a list of folders containing accelerometer.csv and gyroscope.csv.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(folders)} folders...")
    print(f"Device: {device} | Subject: {subject}\n")
    
    successful = 0
    failed = 0
    activity_counts = {}
    
    for folder in sorted(folders):
        try:
            # Extract activity and instance
            activity, instance = extract_activity_and_instance(folder.name)
            
            # Create output filename matching other users' format
            # Capitalize first letter to match user1 format (Walk1.csv, Run2.csv)
            output_filename = f"{activity.capitalize()}{instance}.csv"
            output_csv = output_path / output_filename
            
            # Convert
            convert_folder_to_unified_csv(folder, device, subject, output_csv)
            
            successful += 1
            activity_counts[activity] = activity_counts.get(activity, 0) + 1
            
        except Exception as e:
            print(f"✗ Error converting {folder.name}: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"✓ Conversion complete!")
    print(f"  Successful: {successful}/{len(folders)}")
    print(f"  Failed: {failed}/{len(folders)}")
    print(f"\n  Activity breakdown:")
    for activity, count in sorted(activity_counts.items()):
        print(f"    {activity.capitalize()}: {count} files")
    print(f"\n  Output directory: {output_dir}")
    print(f"{'='*60}\n")


def process_directory(input_dir, output_dir, device, subject):
    """
    Process folders that are already extracted.
    """
    input_path = Path(input_dir)
    
    # Find all subdirectories containing accelerometer.csv and gyroscope.csv
    folders = []
    for item in input_path.iterdir():
        if item.is_dir():
            if (item / "accelerometer.csv").exists() and (item / "gyroscope.csv").exists():
                folders.append(item)
    
    if len(folders) == 0:
        print(f"⚠️ No folders with accelerometer.csv and gyroscope.csv found in {input_dir}")
        return
    
    process_folders(folders, output_dir, device, subject)


def main():
    parser = argparse.ArgumentParser(
        description="Convert user4's folder-based data to unified CSV format"
    )
    
    # Input source (either zip or directory)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--zip_file", help="Path to zip file containing activity folders")
    input_group.add_argument("--input_dir", help="Path to directory containing activity folders (if already extracted)")
    
    # Output and metadata
    parser.add_argument("--output_dir", required=True, 
                        help="Output directory for unified CSVs")
    parser.add_argument("--device", required=True, choices=["phone", "watch"], 
                        help="Device type (phone or watch)")
    parser.add_argument("--subject", required=True, 
                        help="Subject ID (e.g., user4)")
    
    args = parser.parse_args()
    
    if args.zip_file:
        # Process from zip file
        zip_path = Path(args.zip_file)
        if not zip_path.exists():
            print(f"Error: Zip file not found: {args.zip_file}")
            return
        
        process_zip_file(zip_path, args.output_dir, args.device, args.subject)
    
    else:
        # Process from directory
        input_path = Path(args.input_dir)
        if not input_path.exists():
            print(f"Error: Directory not found: {args.input_dir}")
            return
        
        process_directory(args.input_dir, args.output_dir, args.device, args.subject)


if __name__ == "__main__":
    main()