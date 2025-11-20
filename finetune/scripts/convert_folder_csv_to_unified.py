"""
Convert folders containing wristmotion.csv into unified CSV format.

Each folder is named like: lay-1-2025-11-19_17-43-58, run-2-2025-11-19_17-45-12, etc.
Inside each folder: wristmotion.csv (and other files we ignore)

The wristmotion.csv contains:
- rotationRateX/Y/Z → Gyroscope data
- accelerationX/Y/Z → Accelerometer data (user acceleration)
- gravityX/Y/Z → Gravity component (we ignore this)
- Other columns → Ignore

Usage:
    python convert_folder_csv_to_unified.py --input_dir ./user3/watch/raw_folders \
                                             --output_dir ./user3/watch/unified_csv \
                                             --device watch \
                                             --subject user3
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import re


def extract_activity_and_trial_from_foldername(folder_name):
    """
    Extract activity and trial number from folder name.
    
    Examples:
        'lay-1-2025-11-19_17-43-58' → activity='lie', trial=1
        'run-2-2025-11-19_17-45-12' → activity='run', trial=2
        'walk-10-2025-11-19_18-00-00' → activity='walk', trial=10
        'sit-3-2025-11-19_17-50-30' → activity='sit', trial=3
        'stand-5-2025-11-19_17-55-45' → activity='stand', trial=5
    """
    # Convert to lowercase for matching
    folder_lower = folder_name.lower()
    
    # Extract activity and trial number using regex
    # Pattern: (activity)-(trial_number)-rest_of_name
    match = re.match(r'(walk|run|sit|stand|lay|lie|lying)-(\d+)', folder_lower)
    
    if not match:
        raise ValueError(f"Could not extract activity and trial from folder: {folder_name}")
    
    activity = match.group(1)
    trial = int(match.group(2))
    
    # Normalize activity names
    if activity in ['lay', 'lying']:
        activity = 'lie'
    
    return activity, trial


def convert_wristmotion_to_unified_csv(folder_path, device, subject, output_path):
    """
    Extract relevant columns from wristmotion.csv and save as unified format.
    
    Args:
        folder_path: Path to folder containing wristmotion.csv
        device: "phone" or "watch"
        subject: Subject ID (e.g., "user3")
        output_path: Path for output CSV
    """
    
    # Find wristmotion.csv in the folder
    wristmotion_file = folder_path / "wristmotion.csv"
    
    if not wristmotion_file.exists():
        raise FileNotFoundError(f"wristmotion.csv not found in {folder_path.name}")
    
    # Read the CSV
    df = pd.read_csv(wristmotion_file)
    
    # Check if required columns exist
    required_cols = {
        'acc': ['accelerationX', 'accelerationY', 'accelerationZ'],
        'gyro': ['rotationRateX', 'rotationRateY', 'rotationRateZ'],
        'time': ['time', 'seconds_elapsed']
    }
    
    # Find time column (could be 'time' or 'seconds_elapsed')
    time_col = None
    for col in required_cols['time']:
        if col in df.columns:
            time_col = col
            break
    
    if time_col is None:
        raise ValueError(f"No time column found in {wristmotion_file}")
    
    # Check if acceleration and rotation rate columns exist
    for col in required_cols['acc'] + required_cols['gyro']:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {wristmotion_file}")
    
    # Create unified dataframe with standard column names
    # Note: iPhone acceleration is in g units, rotation rate is in rad/s
    # This matches the expected format from other datasets
    unified_df = pd.DataFrame({
        'Time': df[time_col].values,
        'Acc_X': df['accelerationX'].values,
        'Acc_Y': df['accelerationY'].values,
        'Acc_Z': df['accelerationZ'].values,
        'Gyro_X': df['rotationRateX'].values,
        'Gyro_Y': df['rotationRateY'].values,
        'Gyro_Z': df['rotationRateZ'].values,
    })
    
    # Extract activity and trial number from folder name
    activity, trial = extract_activity_and_trial_from_foldername(folder_path.name)
    
    # Add metadata
    unified_df['Device'] = device
    unified_df['Subject'] = subject
    unified_df['Activity'] = activity
    
    # Save to CSV with descriptive name: lie1.csv, run2.csv, etc.
    unified_df.to_csv(output_path, index=False)
    
    print(f"✓ Converted {folder_path.name} → {output_path.name} ({len(unified_df)} samples, activity={activity})")
    
    return unified_df, activity, trial


def batch_convert_folder_directory(input_dir, output_dir, device, subject):
    """Convert all folders containing wristmotion.csv in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all subdirectories that might contain wristmotion.csv
    folders = [f for f in input_path.iterdir() if f.is_dir()]
    
    if len(folders) == 0:
        print(f"⚠️ No subdirectories found in {input_dir}")
        return
    
    print(f"\nProcessing {len(folders)} folders from {input_dir}...")
    print(f"Device: {device} | Subject: {subject}\n")
    
    successful = 0
    failed = 0
    activity_counts = {}
    
    for folder in sorted(folders):
        try:
            # Extract activity and trial number
            activity, trial = extract_activity_and_trial_from_foldername(folder.name)
            
            # Create output filename: lie1.csv, run2.csv, etc.
            # Capitalize first letter to match user1 format (Lie1.csv, Run2.csv)
            output_filename = f"{activity.capitalize()}{trial}.csv"
            output_csv = output_path / output_filename
            
            # Convert
            _, act, _ = convert_wristmotion_to_unified_csv(folder, device, subject, output_csv)
            
            successful += 1
            activity_counts[act] = activity_counts.get(act, 0) + 1
            
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


def main():
    parser = argparse.ArgumentParser(
        description="Convert folders with wristmotion.csv to unified CSV format"
    )
    parser.add_argument("--input_dir", required=True, 
                        help="Directory containing activity folders (e.g., lay-1-2025-11-19_17-43-58)")
    parser.add_argument("--output_dir", required=True, 
                        help="Directory for output unified CSVs")
    parser.add_argument("--device", required=True, choices=["phone", "watch"], 
                        help="Device type (phone or watch)")
    parser.add_argument("--subject", required=True, 
                        help="Subject ID (e.g., user3)")
    
    args = parser.parse_args()
    
    batch_convert_folder_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        device=args.device,
        subject=args.subject
    )


if __name__ == "__main__":
    main()