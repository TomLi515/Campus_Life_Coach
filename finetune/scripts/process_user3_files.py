import pandas as pd
import glob
import os

def convert_time_column_in_folder(base_directory):
    """
    Converts the 'Time' column in all CSV files found in the specified 
    directory to relative seconds (time since the first timestamp in the file).
    
    The script OVERWRITES the original files.
    """
    # --- Configuration ---
    # Find all CSV files directly inside the specified folder.
    search_path = os.path.join(base_directory, '*.csv')
    csv_files = glob.glob(search_path)
    
    if not csv_files:
        print(f"No CSV files found in '{base_directory}'. Please check the folder path.")
        return

    print(f"Found {len(csv_files)} files to process in '{base_directory}'.")
    print("-" * 30)

    # --- Processing Loop ---
    for file_path in csv_files:
        try:
            # 1. Read the CSV file
            df = pd.read_csv(file_path)

            # Ensure 'Time' column exists and is not empty
            if 'Time' not in df.columns or df['Time'].empty:
                print(f"Skipping {file_path}: 'Time' column not found or empty.")
                continue

            # 2. Get the first timestamp (the start time in nanoseconds)
            # This will be the new zero point for relative time.
            time_start_ns = df['Time'].iloc[0]

            # 3. Apply the conversion: (Current Time - Start Time) / 1,000,000,000
            # Converting from nanoseconds (ns) to seconds (s) relative to the start.
            df['Time'] = (df['Time'] - time_start_ns) / 1e9

            # 4. Overwrite the original CSV file with the modified data.
            # Use '%.9f' to maintain high precision for the new time values.
            df.to_csv(file_path, index=False, float_format='%.9f')

            print(f"Successfully processed and updated: {os.path.basename(file_path)}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# ==============================================================================
# ⚠️ CAUTION: This operation will overwrite your original files!
# If you want to save the new files in a separate folder, modify the saving logic (step 4).
# ==============================================================================

# Define the folder containing all your activity CSV files
BASE_DIR = '../data/user3/watch'

# Run the function
convert_time_column_in_folder(BASE_DIR)