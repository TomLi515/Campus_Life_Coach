"""
Organize and rename CSV files from all users into a single directory.

Output format: user_device_activity_instance.csv
Example: user1_phone_walk_1.csv, user2_phone_sit_3.csv, user3_watch_lie_5.csv

Usage:
    python organize_and_rename_csvs.py --input_root ../data \
                                        --output_dir ../data/all_users_organized
"""

import shutil
from pathlib import Path
import argparse
import re


def extract_activity_and_instance(filename):
    """
    Extract activity and instance number from various filename formats.
    
    Handles:
        - Walk1.csv → (walk, 1)
        - walk_1.csv → (walk, 1)
        - Lie10.csv → (lie, 10)
        - sit_trial_3.csv → (sit, 3)
    """
    # Remove extension
    name = filename.replace('.csv', '').lower()
    
    # Try to match pattern: activity + number
    # Pattern 1: Walk1, sit2, Run10, etc.
    match = re.match(r'([a-z]+)(\d+)', name)
    if match:
        activity = match.group(1)
        instance = int(match.group(2))
        return activity, instance
    
    # Pattern 2: walk_1, sit_2, run_trial_3, etc.
    match = re.search(r'([a-z]+)_.*?(\d+)', name)
    if match:
        activity = match.group(1)
        instance = int(match.group(2))
        return activity, instance
    
    # Pattern 3: Just activity name (assume instance 1)
    for activity in ['walk', 'run', 'sit', 'stand', 'lie', 'lying', 'lay']:
        if activity in name:
            # Try to find any number
            numbers = re.findall(r'\d+', name)
            instance = int(numbers[0]) if numbers else 1
            return activity if activity not in ['lying', 'lay'] else 'lie', instance
    
    raise ValueError(f"Could not extract activity and instance from: {filename}")


def find_user_csv_files(user_dir, user_id):
    """
    Find all CSV files for a user across phone and watch directories.
    
    Returns:
        List of tuples: (csv_path, device_type)
    """
    csv_files = []
    user_path = Path(user_dir)
    
    # Check for different directory structures
    possible_structures = [
        # Structure 1: user1/phone/*.csv, user1/watch/*.csv
        ('phone', user_path / 'phone'),
        ('watch', user_path / 'watch'),
        
        # Structure 2: user1/phone/unified_csv/*.csv
        ('phone', user_path / 'phone' / 'unified_csv'),
        ('watch', user_path / 'watch' / 'unified_csv'),
        
        # Structure 3: user1/*.csv (directly in user folder, determine device from CSV)
        ('auto', user_path),
    ]
    
    for device, dir_path in possible_structures:
        if not dir_path.exists():
            continue
        
        for csv_file in dir_path.glob('*.csv'):
            # Skip if already processed (contains device info in metadata)
            csv_files.append((csv_file, device, user_id))
    
    return csv_files


def determine_device_from_csv(csv_path):
    """
    Read CSV and determine device from 'Device' column if available.
    """
    try:
        import pandas as pd
        df = pd.read_csv(csv_path, nrows=1)
        if 'Device' in df.columns:
            return df['Device'].iloc[0].lower()
    except:
        pass
    
    return None


def organize_and_rename(input_root, output_dir, dry_run=False):
    """
    Copy and rename all CSV files to output directory.
    
    Args:
        input_root: Root directory containing user folders (user1, user2, etc.)
        output_dir: Output directory for organized CSVs
        dry_run: If True, only print what would be done
    """
    input_path = Path(input_root)
    output_path = Path(output_dir)
    
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all user directories
    user_dirs = [d for d in input_path.iterdir() if d.is_dir() and d.name.startswith('user')]
    
    if len(user_dirs) == 0:
        print(f"⚠️ No user directories found in {input_root}")
        print("Expected structure: user1/, user2/, user3/, user4/")
        return
    
    print(f"Found {len(user_dirs)} user directories: {[d.name for d in user_dirs]}\n")
    
    total_files = 0
    renamed_files = 0
    skipped_files = 0
    errors = []
    
    for user_dir in sorted(user_dirs):
        user_id = user_dir.name  # e.g., "user1", "user2"
        print(f"Processing {user_id}...")
        
        # Find all CSV files for this user
        csv_files = find_user_csv_files(user_dir, user_id)
        
        if len(csv_files) == 0:
            print(f"  ⚠️ No CSV files found for {user_id}")
            continue
        
        print(f"  Found {len(csv_files)} CSV files")
        
        for csv_path, device_hint, _ in csv_files:
            total_files += 1
            
            try:
                # Determine device
                if device_hint == 'auto':
                    device = determine_device_from_csv(csv_path)
                    if device is None:
                        print(f"  ⚠️ Skipping {csv_path.name}: Cannot determine device")
                        skipped_files += 1
                        continue
                else:
                    device = device_hint
                
                # Extract activity and instance
                activity, instance = extract_activity_and_instance(csv_path.name)
                
                # Normalize activity name
                if activity in ['lying', 'lay']:
                    activity = 'lie'
                
                # Create new filename: user_device_activity_instance.csv
                new_filename = f"{user_id}_{device}_{activity}_{instance}.csv"
                new_path = output_path / new_filename
                
                # Check for duplicates
                if new_path.exists() and not dry_run:
                    print(f"  ⚠️ Duplicate: {new_filename} already exists, skipping {csv_path.name}")
                    skipped_files += 1
                    continue
                
                # Copy and rename
                if dry_run:
                    print(f"  [DRY RUN] {csv_path.name} → {new_filename}")
                else:
                    shutil.copy2(csv_path, new_path)
                    print(f"  ✓ {csv_path.name} → {new_filename}")
                
                renamed_files += 1
                
            except Exception as e:
                error_msg = f"  ✗ Error processing {csv_path.name}: {e}"
                print(error_msg)
                errors.append(error_msg)
        
        print()
    
    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total files found: {total_files}")
    print(f"Successfully renamed: {renamed_files}")
    print(f"Skipped: {skipped_files}")
    print(f"Errors: {len(errors)}")
    
    if not dry_run:
        print(f"\n✓ All files organized in: {output_dir}")
    else:
        print("\n[DRY RUN] No files were actually copied. Run without --dry-run to execute.")
    
    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(error)
    
    print("="*60)


def verify_naming_convention(output_dir):
    """
    Verify that all files follow the naming convention and check for issues.
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"Directory does not exist: {output_dir}")
        return
    
    csv_files = list(output_path.glob("*.csv"))
    
    print(f"\nVerifying {len(csv_files)} files in {output_dir}...\n")
    
    # Group by user and device
    from collections import defaultdict
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    pattern = re.compile(r'(user\d+)_(phone|watch)_([a-z]+)_(\d+)\.csv')
    
    for csv_file in sorted(csv_files):
        match = pattern.match(csv_file.name)
        
        if not match:
            print(f"⚠️ Does not match naming convention: {csv_file.name}")
            continue
        
        user, device, activity, instance = match.groups()
        grouped[user][device][activity].append(int(instance))
    
    # Print summary
    for user in sorted(grouped.keys()):
        print(f"{user}:")
        for device in sorted(grouped[user].keys()):
            print(f"  {device}:")
            for activity in sorted(grouped[user][device].keys()):
                instances = sorted(grouped[user][device][activity])
                count = len(instances)
                print(f"    {activity}: {count} files (instances: {instances})")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Organize and rename CSV files from all users into standardized format"
    )
    parser.add_argument("--input_root", required=True, 
                        help="Root directory containing user folders (user1, user2, etc.)")
    parser.add_argument("--output_dir", required=True, 
                        help="Output directory for organized CSVs")
    parser.add_argument("--dry-run", action="store_true", 
                        help="Preview changes without actually copying files")
    parser.add_argument("--verify", action="store_true", 
                        help="Verify naming convention of files in output directory")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_naming_convention(args.output_dir)
    else:
        organize_and_rename(args.input_root, args.output_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()