# data_integration_export.py
import pandas as pd
import dask.dataframe as dd
import shutil
import numpy as np
import os
from pathlib import Path
import logging
from typing import List

logger = logging.getLogger(Path(__file__).stem)

# --- Plan Implementation: Part 1 - Data Integration and Export Module ---
# This script is responsible for connecting to the 'dash DB' and exporting
# processed data into it. It covers the following parts of the plan:
# 1.1: Database Connection Manager (via get_db_engine)
# 1.2: Data Formatting (within export_to_dash_db)
# 1.3: Export Functionality (the export_to_dash_db function itself)
# 1.4: Idempotency (using the 'if_exists' parameter)

def export_csv_to_dask(input_dirs: List[Path], 
                       output_dir: Path, 
                       index_col: str ="doi", 
                       partition_col: str = "",
                       publication_date_col: str ="publication_date"):
    """
    Reads all CSV files from an input directory using Dask, and saves them
    as a single, partitioned Parquet dataset.

    Args:
        input_dirs : Paths to the directories containing source CSV files.
        output_dir : Path to the directory where the Parquet dataset will be saved.
        index_col : The name of the column to index by
        partition_col : The name of the column to partition the data by.
    """
    # --- 1. Setup: Ensure directories exist and handle potential errors ---

    # The output directory should not exist before writing, Dask handles creation.
    if os.path.exists(output_dir):
        logger.warning(f"Warning: Output directory '{output_dir}' already exists.")
    if not partition_col and not publication_date_col:
        raise ValueError("Expected at least one of parition_col or publication_date_col to not be empty")
    # --- 2. Read all CSV files ---
    csv_paths = []
    for path in input_dirs:
        csv_paths += list(path.glob("*.csv"))
        logger.info(f"Indexing all CSVs from: {path}")
    logger.debug(f"Found CSV paths: {csv_paths}")
    
    try:
        # Assuming the partition column exists and can be inferred.
        # For date columns, it's good practice to specify the dtype.
        # If your partition column is a date, Dask might need help parsing it.
        # ddf = dd.read_csv(csv_path, parse_dates=[partition_col])
        ddf = dd.read_csv(csv_paths, 
                          dtype={'citation_count_OpenAlex': 'float64',
                                            "referenced_works_OpenAlex": 'object'})
        ddf = ddf.dropna(subset=[index_col])
        if not partition_col and publication_date_col:
            partition_col = "publication_year"

            ddf[publication_date_col] = dd.to_datetime(ddf[publication_date_col])
            ddf["publication_year"] = ddf[publication_date_col].dt.year.astype(int)
            # ddf["publication_year"] = ddf["publication_year"].astype(int)
            
        ddf = ddf.set_index(index_col)

    except Exception as e:
        logger.error(f"{e}")
        return

    # --- 3. Check if the partition column exists ---
    if partition_col not in ddf.columns:
        logger.error(f"Error: Partition column '{partition_col}' not found in the data.")
        logger.error(f"Available columns are: {list(ddf.columns)}")
        return

    logger.info(f"Data loaded. Total rows (estimated): {len(ddf)}")
    logger.info(f"Columns: {list(ddf.columns)}")

    # --- 4. Write to Partitioned Parquet ---
    # Dask handles all the complexity of creating the directory structure
    # and writing the data in parallel.
    logger.info(f"\nWriting data to partitioned Parquet format in '{output_dir}'...")
    logger.info(f"Partitioning by column: '{partition_col}'")

    try:
        # This is the key command. Dask does all the heavy lifting.
        # ddf[partition_col] = ddf[partition_col].astype(int)
        ddf.to_parquet(
            output_dir,
            engine='pyarrow',
            write_index=True,
            partition_on=[partition_col]
        )
        logger.info(f"\n✅ Success! Data converted and saved to '{output_dir}'")
    except Exception as e:
        logger.error(f"An error occurred during Parquet conversion: {e}")

## RUN ##
def run_main_data_integration_export(config):
  # placeholders dor config imports
    journals = ["Nature", "Science"] # placeholder
    output_dir = config["output_dir"]
    input_dir = config["input_dir"]
    os.makedirs(output_dir, exist_ok=True)
    directories = []
    for journal in journals:

        input_dir_main = input_dir /  journal
        input_dir_ref = input_dir / journal / "references"
        input_dir_cit = input_dir / journal / "citations"
        directories.extend([input_dir_main, input_dir_cit, input_dir_ref])

    output_dir_main = output_dir / "database"
    output_dir_secondary = output_dir / "database_idx_date"

    # create main db
    export_csv_to_dask(directories, output_dir_main, 
                       index_col="doi", 
                       publication_date_col="publication_date")

    ### do this for the reference dir also
if __name__ == "__main__":
    from utils.load_config import get_integration_config
    from utils.setup_logging import setup_logger
    config = get_integration_config()
    setup_logger(logger, config["log"])

    run_main_data_integration_export(config)


