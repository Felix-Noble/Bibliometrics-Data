# data_integration_export.py
import pandas as pd
import dask.dataframe as dd
import shutil
import numpy as np
import os
from pathlib import Path

# --- Plan Implementation: Part 1 - Data Integration and Export Module ---
# This script is responsible for connecting to the 'dash DB' and exporting
# processed data into it. It covers the following parts of the plan:
# 1.1: Database Connection Manager (via get_db_engine)
# 1.2: Data Formatting (within export_to_dash_db)
# 1.3: Export Functionality (the export_to_dash_db function itself)
# 1.4: Idempotency (using the 'if_exists' parameter)

def convert_csv_to_partitioned_parquet(input_dir, output_dir, index_col="doi", partition_col="year"):
    """
    Reads all CSV files from an input directory using Dask, and saves them
    as a single, partitioned Parquet dataset.

    Args:
        input_dir (str): Path to the directory containing source CSV files.
        output_dir (str): Path to the directory where the Parquet dataset will be saved.
        partition_col (str): The name of the column to partition the data by.
    """
    # --- 1. Setup: Ensure directories exist and handle potential errors ---

    # The output directory should not exist before writing, Dask handles creation.
    if os.path.exists(output_dir):
        print(f"Warning: Output directory '{output_dir}' already exists.")
        # In a real workflow, you might want to delete it first:
        # import shutil
        # shutil.rmtree(output_dir)
        # print("Removed existing output directory.")

    # --- 2. Use Dask to read all CSV files ---
    # The '*' is a wildcard that tells Dask to read all files ending in .csv
    # This is much more memory-efficient for very large datasets than reading
    # them all with pandas first.
    csv_path = os.path.join(input_dir, '*.csv')
    print(f"Reading all CSVs from: {csv_path}")

    try:
        # Assuming the partition column exists and can be inferred.
        # For date columns, it's good practice to specify the dtype.
        # If your partition column is a date, Dask might need help parsing it.
        # ddf = dd.read_csv(csv_path, parse_dates=[partition_col])
        ddf = dd.read_csv(csv_path, dtype={'reference_DOIs': 'object'})
        ddf = ddf.dropna(subset=[index_col])
        ddf = ddf.set_index(index_col)

    except Exception as e:
        print(f"Error reading CSV files with Dask: {e}")
        return

    # --- 3. Check if the partition column exists ---
    if partition_col not in ddf.columns:
        print(f"Error: Partition column '{partition_col}' not found in the data.")
        print(f"Available columns are: {list(ddf.columns)}")
        return

    print(f"Data loaded. Total rows (estimated): {len(ddf)}")
    print(f"Columns: {list(ddf.columns)}")

    # --- 4. Write to Partitioned Parquet ---
    # Dask handles all the complexity of creating the directory structure
    # and writing the data in parallel.
    print(f"\nWriting data to partitioned Parquet format in '{output_dir}'...")
    print(f"Partitioning by column: '{partition_col}'")

    try:
        # This is the key command. Dask does all the heavy lifting.
        # ddf[partition_col] = ddf[partition_col].astype(int)
        ddf.to_parquet(
            output_dir,
            engine='pyarrow',
            write_index=False,
            partition_on=[partition_col]
        )
        print(f"\n✅ Success! Data converted and saved to '{output_dir}'")
    except Exception as e:
        print(f"An error occurred during Parquet conversion: {e}")

## RUN ##
def run_main_data_integration_export(config=None):
  # placeholders dor config imports
    journals = ["Nature", "American Chemical Society"]
    output_parent_dir = Path(os.getcwd()) / "DataBase"
    os.makedirs(output_parent_dir, exist_ok=True)
    for journal in journals:

        input_dir_main = Path(os.getcwd()) / "data" / journal
        input_dir_ref = Path(os.getcwd()) / "data" / journal / "references"
        input_dir_cit = Path(os.getcwd()) / "data" / journal / "citations"

        output_dir_main = output_parent_dir / journal / "main"
        output_dir_ref = output_parent_dir / journal / "references"
        output_dir_cit = output_parent_dir / journal / "citations"


        convert_csv_to_partitioned_parquet(input_dir_main, output_dir_main, index_col="publication_date")

    ### do this for the reference dir also
if __name__ == "__main__":
    from ..utils.load_config import get_integration_config
    config = get_integration_config()

    print(config)
